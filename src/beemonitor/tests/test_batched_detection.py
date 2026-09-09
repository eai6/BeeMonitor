"""Batched detection, and the seam it crosses.

Two things are pinned here. First, detect_batch keeps detections separated PER
FRAME — detect() flattens every Results into one list, so a list passed to it
would merge frames silently, which is the trap this method exists to avoid.
Second, the lookback replay produces exactly what the one-at-a-time loop did;
only the number of forward passes changes.
"""

import unittest

import numpy as np

from beemonitor.core.profiling import PROFILER
from beemonitor.detection.base_detector import Detection
from beemonitor.detection.yolo_detector import YOLODetector
from beemonitor.tracking.bee_tracking import TRACKER_ROW_FIELDS, to_tracker_rows


class _Boxes:
    """n detections, each at a distinct, identifiable position."""

    def __init__(self, n, offset):
        self.n = n
        self.offset = offset
        self.cls = [0] * n
        self.conf = [0.5 + i / 100 for i in range(n)]

    def __len__(self):
        return self.n

    @property
    def xyxy(self):
        class _Arr:
            def __init__(self, v):
                self.v = v

            def cpu(self):
                return self

            def numpy(self):
                return self.v

        return [_Arr(np.array([self.offset + i, 0.0, self.offset + i + 10, 10.0]))
                for i in range(self.n)]


class _Result:
    names = {0: "bee"}

    def __init__(self, n, offset):
        self.boxes = _Boxes(n, offset)


class _Model:
    """Returns one Results per input image; records the batch sizes it saw."""

    def __init__(self, per_frame=(1, 2, 3)):
        self.per_frame = per_frame
        self.batch_sizes = []

    def __call__(self, images, **kwargs):
        images = images if isinstance(images, list) else [images]
        self.batch_sizes.append(len(images))
        return [_Result(self.per_frame[i % len(self.per_frame)], offset=100 * i)
                for i in range(len(images))]


def _frame():
    return np.zeros((8, 8, 3), dtype=np.uint8)


class DetectBatchTests(unittest.TestCase):
    def test_detections_stay_separated_per_frame(self):
        model = _Model(per_frame=(1, 2, 3))
        detector = YOLODetector(model)

        out = detector.detect_batch([_frame(), _frame(), _frame()])

        self.assertEqual([len(d) for d in out], [1, 2, 3])

    def test_one_forward_pass_for_the_whole_batch(self):
        model = _Model()
        YOLODetector(model).detect_batch([_frame()] * 6)

        self.assertEqual(model.batch_sizes, [6])

    def test_detect_is_the_single_frame_case_of_detect_batch(self):
        model = _Model(per_frame=(2,))
        detector = YOLODetector(model)

        single = detector.detect(_frame())
        batched = detector.detect_batch([_frame()])[0]

        self.assertEqual(len(single), len(batched))
        self.assertEqual([d.bbox for d in single], [d.bbox for d in batched])

    def test_an_empty_batch_calls_nothing(self):
        model = _Model()

        self.assertEqual(YOLODetector(model).detect_batch([]), [])
        self.assertEqual(model.batch_sizes, [])

    def test_class_filtering_applies_per_frame(self):
        model = _Model(per_frame=(2,))
        detector = YOLODetector(model, tracking_classes=["wasp"])  # nothing matches

        self.assertEqual(detector.detect_batch([_frame(), _frame()]), [[], []])

    def test_inference_time_is_attributed_to_every_frame_in_the_batch(self):
        """calls counts frames, not model invocations — otherwise batching would
        look like it made the GPU stage cheaper per call while doing the same work."""
        PROFILER.reset()
        YOLODetector(_Model()).detect_batch([_frame()] * 5)

        self.assertEqual(PROFILER.snapshot()["inference"]["calls"], 5)
        PROFILER.reset()


class TrackerRowTests(unittest.TestCase):
    """The detector->tracker seam, which BeeTracker.update reads positionally."""

    def test_rows_match_the_documented_layout(self):
        det = Detection(bbox=(1.0, 2.0, 3.0, 4.0), centroid=(2.0, 3.0),
                        confidence=0.9, label="bee", source="yolo")

        row = to_tracker_rows([det])[0]

        self.assertEqual(len(row), len(TRACKER_ROW_FIELDS))
        self.assertEqual(row, [1.0, 2.0, 3.0, 4.0, 0.9, "yolo", "bee"])

    def test_the_label_becomes_the_taxon(self):
        det = Detection(bbox=(0, 0, 1, 1), centroid=(0, 0), confidence=0.5,
                        label="wasp", source="yolo")

        self.assertEqual(to_tracker_rows([det])[0][-1], "wasp")

    def test_empty_in_empty_out(self):
        self.assertEqual(to_tracker_rows([]), [])


class LookbackReplayTests(unittest.TestCase):
    """When motion starts, the buffered frames are replayed through YOLO.

    They all need detection and none depends on another, so they go in one
    forward pass — but the tracker must still be updated one frame at a time, in
    order, because its state is sequential.
    """

    def _tracker(self, buffered_frames):
        from beemonitor.tracking.bee_tracking import BeeTracking

        bt = object.__new__(BeeTracking)      # __init__ loads YOLO weights
        bt.roi = None
        bt.enable_two_mode = True
        bt.mode = "motion_detection"
        bt.processing_lookback = False
        bt.frame_buffer = [(i, _frame()) for i in range(buffered_frames)]
        bt.lookback_frames = buffered_frames
        bt.frames_since_motion = 0
        bt.motion_cooldown = 10
        bt.identifier = None
        bt.species_classifier = None
        bt.save_crops = False
        bt._mask_roi = lambda f: f
        bt.detect_motion = lambda f: True     # motion -> switch to tracking

        self.model = _Model(per_frame=(1,))
        bt.yolo_detector = YOLODetector(self.model)

        self.updates = []

        class _Tracker:
            def update(inner, detections, frame_num):
                self.updates.append((frame_num, len(detections)))
                return []

        bt.tracker = _Tracker()
        return bt

    def test_the_buffer_is_replayed_in_one_forward_pass(self):
        bt = self._tracker(buffered_frames=5)

        bt.process_frame(_frame(), frame_num=5)

        # One batched call for the 5 buffered frames, then the current frame.
        self.assertEqual(self.model.batch_sizes, [5, 1])

    def test_the_tracker_is_still_updated_frame_by_frame_in_order(self):
        bt = self._tracker(buffered_frames=4)

        result = bt.process_frame(_frame(), frame_num=4)

        # In motion_detection mode the current frame is appended to the buffer
        # before the motion check, and the buffer is capped at lookback_frames —
        # so frame 0 ages out and frame 4 is replayed AND then processed as the
        # current frame. Pre-existing behaviour, unchanged by batching; pinned
        # here so a future change to the replay has to be deliberate.
        self.assertEqual([n for n, _ in self.updates], [1, 2, 3, 4, 4])
        self.assertEqual([r["frame_num"] for r in result["lookback_results"]],
                         [1, 2, 3, 4])

    def test_every_buffered_frame_keeps_its_own_detections(self):
        bt = self._tracker(buffered_frames=3)

        bt.process_frame(_frame(), frame_num=3)

        # One detection per frame — not three merged onto the first, which is
        # what a naive list-through-detect() would have produced.
        self.assertEqual([n for _, n in self.updates], [1, 1, 1, 1])

    def test_the_buffer_is_cleared_after_replay(self):
        bt = self._tracker(buffered_frames=3)

        bt.process_frame(_frame(), frame_num=3)

        self.assertEqual(bt.frame_buffer, [])
        self.assertFalse(bt.processing_lookback)

    def test_an_empty_buffer_costs_no_extra_call(self):
        bt = self._tracker(buffered_frames=0)

        bt.process_frame(_frame(), frame_num=0)

        self.assertEqual(self.model.batch_sizes, [1])   # the current frame only


class DetectorInterfaceTests(unittest.TestCase):
    """Every detector the tracker can be pointed at must answer detect_batch.

    The tracker's lookback replay calls it, and detector_kind chooses which
    class is behind that name at runtime. Adding detect_batch to YOLODetector
    alone broke SAM 3 tracking in production — the step failed with
    "'Sam3Detector' object has no attribute 'detect_batch'" — so the contract
    lives on BaseDetector and this test walks every implementation.
    """

    def _detectors(self):
        from beemonitor.detection.base_detector import BaseDetector
        from beemonitor.detection.blob_detector import BlobDetector
        from beemonitor.detection.sam3_detector import Sam3Detector
        from beemonitor.detection.sift_detector import SIFTDetector
        from beemonitor.detection.yolo_detector import YOLODetector

        self.assertTrue(issubclass(Sam3Detector, BaseDetector))
        return [Sam3Detector, BlobDetector, SIFTDetector, YOLODetector]

    def test_every_detector_exposes_detect_batch(self):
        for cls in self._detectors():
            self.assertTrue(hasattr(cls, "detect_batch"), cls.__name__)

    def test_the_default_returns_one_list_per_frame(self):
        """The fallback must keep frames separate, like the batched override."""
        from beemonitor.detection.sam3_detector import Sam3Detector

        detector = Sam3Detector(prompt="bee")
        detector._ensure_model = lambda: None
        calls = []

        def fake_segment(pil, prompt):
            # Boxes must not overlap: detect() runs NMS, which would fold
            # identical ones into a single detection and hide the separation
            # this test is checking.
            x = 100.0 * len(calls)
            calls.append(prompt)
            return [(x, 0.0, x + 10.0, 10.0, 0.9)]

        detector._segment = fake_segment
        out = detector.detect_batch([_frame(), _frame(), _frame()])

        self.assertEqual(len(out), 3)
        self.assertEqual([len(d) for d in out], [1, 1, 1])
        self.assertEqual([d[0].bbox[0] for d in out], [0.0, 100.0, 200.0])

    def test_an_empty_batch_is_empty_on_the_default_too(self):
        from beemonitor.detection.blob_detector import BlobDetector

        self.assertEqual(BaseDetectorProbe().detect_batch([]), [])


class BaseDetectorProbe:
    """Minimal concrete detector, to exercise the default straight."""

    from beemonitor.detection.base_detector import BaseDetector as _B
    detect_batch = _B.detect_batch

    def detect(self, frame, **kwargs):
        return ["one"]


if __name__ == "__main__":
    unittest.main()
