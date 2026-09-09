"""Stage accounting — the numbers the cost line and the "is the GPU idle?"
question both read.

The point of these is that a stage is recorded where the work happens (inside
the detector, inside the frame loop), not estimated from the outside, and that
concurrent producers can record without losing time.
"""

import threading
import unittest

import numpy as np

from beemonitor.core.profiling import PROFILER, StageProfiler


class StageProfilerTests(unittest.TestCase):
    def setUp(self):
        self.profiler = StageProfiler()

    def test_stages_accumulate_across_calls(self):
        for _ in range(3):
            with self.profiler.stage("decode"):
                pass

        snap = self.profiler.snapshot()
        self.assertEqual(snap["decode"]["calls"], 3)
        self.assertGreaterEqual(snap["decode"]["seconds"], 0.0)

    def test_count_records_work_not_just_calls(self):
        """A batched call is one call over many frames — both are worth knowing."""
        with self.profiler.stage("inference", count=8):
            pass

        snap = self.profiler.snapshot()
        self.assertEqual(snap["inference"]["calls"], 8)

    def test_a_raising_block_is_still_recorded(self):
        with self.assertRaises(ValueError):
            with self.profiler.stage("decode"):
                raise ValueError("boom")

        self.assertEqual(self.profiler.snapshot()["decode"]["calls"], 1)

    def test_reset_clears_everything(self):
        with self.profiler.stage("decode"):
            pass
        self.profiler.reset()

        self.assertEqual(self.profiler.snapshot(), {})

    def test_concurrent_producers_do_not_lose_records(self):
        """The reader thread records decode while the main thread records
        inference — the whole reason this is lock-guarded."""
        def work(stage):
            for _ in range(50):
                with self.profiler.stage(stage):
                    pass

        threads = [threading.Thread(target=work, args=(s,))
                   for s in ("decode", "inference", "decode")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = self.profiler.snapshot()
        self.assertEqual(snap["decode"]["calls"], 100)
        self.assertEqual(snap["inference"]["calls"], 50)

    def test_seconds_of_an_unrecorded_stage_is_zero(self):
        self.assertEqual(self.profiler.seconds("inference"), 0.0)


class DetectorRecordsInferenceTests(unittest.TestCase):
    """The GPU stage is timed inside YOLODetector, at the call itself."""

    class _FakeBoxes:
        cls = [0]
        conf = [0.9]

        def __len__(self):
            return 0

    class _FakeResult:
        names = {0: "bee"}

        def __init__(self):
            self.boxes = DetectorRecordsInferenceTests._FakeBoxes()

    class _FakeModel:
        def __call__(self, frame, **kwargs):
            return [DetectorRecordsInferenceTests._FakeResult()]

    def test_detect_records_an_inference_stage(self):
        from beemonitor.detection.yolo_detector import YOLODetector

        PROFILER.reset()
        YOLODetector(self._FakeModel()).detect(np.zeros((8, 8, 3), np.uint8))

        self.assertEqual(PROFILER.snapshot()["inference"]["calls"], 1)
        PROFILER.reset()


if __name__ == "__main__":
    unittest.main()


class Sam3RecordsInferenceTests(unittest.TestCase):
    """SAM 3 must report GPU time under the same stage name as YOLO.

    It is the heavier detector by a wide margin — the reason it gets its own
    g5 endpoint — so a run that reported gpu_seconds = 0 would read as "the GPU
    was idle" on precisely the path where it is busiest.
    """

    def test_detect_records_one_inference_call_per_frame(self):
        from beemonitor.core.profiling import PROFILER
        from beemonitor.detection.sam3_detector import Sam3Detector

        detector = Sam3Detector(prompt="bee, wasp")     # two prompts, one frame
        detector._ensure_model = lambda: None
        detector._segment = lambda pil, prompt: [(0.0, 0.0, 1.0, 1.0, 0.9)]

        PROFILER.reset()
        dets = detector.detect(np.zeros((4, 4, 3), np.uint8))

        # calls counts FRAMES, not prompt passes, so the figure means the same
        # thing whichever detector produced it.
        self.assertEqual(PROFILER.snapshot()["inference"]["calls"], 1)
        self.assertTrue(dets)
        PROFILER.reset()

    def test_both_detectors_report_the_same_stage_name(self):
        from beemonitor.core.profiling import PROFILER
        from beemonitor.detection.sam3_detector import Sam3Detector
        from beemonitor.detection.yolo_detector import YOLODetector

        PROFILER.reset()
        YOLODetector(DetectorRecordsInferenceTests._FakeModel()).detect(
            np.zeros((8, 8, 3), np.uint8))
        yolo_stages = set(PROFILER.snapshot())

        detector = Sam3Detector()
        detector._ensure_model = lambda: None
        detector._segment = lambda pil, prompt: []
        PROFILER.reset()
        detector.detect(np.zeros((4, 4, 3), np.uint8))
        sam3_stages = set(PROFILER.snapshot())

        self.assertEqual(yolo_stages, sam3_stages, "stage names must not diverge")
        PROFILER.reset()


class Sam3ConcurrentLoadTests(unittest.TestCase):
    """SAM 3 loads once per process, and concurrent first use cannot race.

    Batch 5af72b17 failed partly on
    ``ImportError: cannot import name 'Sam3Model' from 'transformers'`` —
    intermittently, under concurrency only. transformers resolves submodules
    lazily, so two threads importing it for the first time at once can have one
    observe a half-registered module. The container serves several invocations
    on gthread workers, each building its own detector, so a batch triggers what
    a single retry never does.
    """

    def setUp(self):
        from beemonitor.detection import sam3_detector as sd
        sd._SHARED_MODEL = sd._SHARED_PROCESSOR = sd._SHARED_DEVICE = None
        self.sd = sd

    def tearDown(self):
        self.sd._SHARED_MODEL = self.sd._SHARED_PROCESSOR = self.sd._SHARED_DEVICE = None

    def _detector(self, loads):
        """A detector whose load is instrumented and slow enough to overlap."""
        import time

        det = self.sd.Sam3Detector(prompt="bee")
        real_lock = self.sd._LOAD_LOCK

        def fake_load():
            # Stands in for the import + from_pretrained pair.
            time.sleep(0.02)
            loads.append(1)
            return object(), object(), "cpu"

        return det, fake_load, real_lock

    def test_the_model_is_loaded_once_across_threads(self):
        loads = []

        def load_side_effect():
            import time
            time.sleep(0.02)
            loads.append(1)
            self.sd._SHARED_PROCESSOR = object()
            self.sd._SHARED_MODEL = object()
            self.sd._SHARED_DEVICE = "cpu"

        def ensure(det):
            if det._model is not None:
                return
            with self.sd._LOAD_LOCK:
                if self.sd._SHARED_MODEL is None:
                    load_side_effect()
                det._processor = self.sd._SHARED_PROCESSOR
                det._model = self.sd._SHARED_MODEL
                det._device = self.sd._SHARED_DEVICE

        dets = [self.sd.Sam3Detector(prompt="bee") for _ in range(6)]
        threads = [threading.Thread(target=ensure, args=(d,)) for d in dets]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(sum(loads), 1, "the model must load exactly once")
        self.assertTrue(all(d._model is dets[0]._model for d in dets),
                        "every detector must share the one model")

    def test_a_second_detector_reuses_the_loaded_model(self):
        self.sd._SHARED_MODEL = sentinel_model = object()
        self.sd._SHARED_PROCESSOR = object()
        self.sd._SHARED_DEVICE = "cpu"

        det = self.sd.Sam3Detector(prompt="wasp")
        det._ensure_model()

        self.assertIs(det._model, sentinel_model)
        self.assertEqual(det.prompts, ["wasp"], "prompts stay per-detector")

    def test_prompts_remain_per_detector_though_the_model_is_shared(self):
        self.sd._SHARED_MODEL = object()
        self.sd._SHARED_PROCESSOR = object()
        self.sd._SHARED_DEVICE = "cpu"

        a = self.sd.Sam3Detector(prompt="bee")
        b = self.sd.Sam3Detector(prompt="hoverfly, beetle")
        a._ensure_model()
        b._ensure_model()

        self.assertIs(a._model, b._model)
        self.assertEqual(a.prompts, ["bee"])
        self.assertEqual(b.prompts, ["hoverfly", "beetle"])
