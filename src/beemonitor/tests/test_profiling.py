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
