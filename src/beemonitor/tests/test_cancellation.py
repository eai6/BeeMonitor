"""A cancelled clip stops instead of running to the end (memory/49).

SageMaker async inference can't abort a request, so before this a cancelled
clip still used its ~11 GPU minutes and its result was thrown away.
"""
import threading
import time
import unittest
from unittest.mock import patch

from beemonitor import cancellation
from beemonitor.tracking import bee_tracking
from beemonitor.tests.test_frame_pipeline import FakeCapture


class MarkerTests(unittest.TestCase):
    def test_every_chunk_of_a_job_shares_its_marker(self):
        self.assertEqual(cancellation.marker_key("pl_ab12"), "cancel/pl_ab12")
        self.assertEqual(cancellation.marker_key("pl_ab12-c3"), "cancel/pl_ab12")

    def test_nothing_is_cancelled_outside_a_watch(self):
        cancellation.watch(None)
        self.assertFalse(cancellation.requested())
        cancellation.check()  # does not raise


class WatcherTests(unittest.TestCase):
    def test_a_marker_present_up_front_is_seen_before_any_work(self):
        w = cancellation.Watcher("pl_x", lambda key: key == "cancel/pl_x")
        self.assertTrue(w.cancelled_already())

    def test_a_marker_that_appears_mid_clip_sets_the_flag(self):
        present = threading.Event()
        w = cancellation.Watcher("pl_x", lambda key: present.is_set(), interval=0.01)
        self.assertFalse(w.cancelled_already())
        with w:
            self.assertFalse(cancellation.requested())
            present.set()
            deadline = time.time() + 2
            while not cancellation.requested() and time.time() < deadline:
                time.sleep(0.01)
            self.assertTrue(cancellation.requested())
            with self.assertRaises(cancellation.JobCancelled):
                cancellation.check()
        self.assertFalse(cancellation.requested())  # cleared on exit

    def test_a_failing_lookup_never_cancels(self):
        def boom(key):
            raise RuntimeError("S3 down")
        w = cancellation.Watcher("pl_x", boom)
        self.assertFalse(w.cancelled_already())

    def test_the_flag_belongs_to_the_thread_running_the_clip(self):
        w = cancellation.Watcher("pl_x", lambda key: True)
        w.cancelled_already()
        seen = []
        with w:
            t = threading.Thread(target=lambda: seen.append(cancellation.requested()))
            t.start()
            t.join()
            self.assertTrue(cancellation.requested())
        self.assertEqual(seen, [False])


class FrameLoopTests(unittest.TestCase):
    def _tracker(self, seen):
        from beemonitor.tracking.bee_tracking import BeeTracking

        tracker = object.__new__(BeeTracking)  # no YOLO weights needed
        tracker.save_crops = False
        tracker.track_crop_counts = {}
        tracker.detections_df = None

        def fake_process_frame(frame, frame_num, visualize=False):
            seen.append(frame_num)
            return {"frame_num": frame_num, "detections": [], "tracks": [],
                    "mode": "motion_detection", "lookback_results": []}

        tracker.process_frame = fake_process_frame
        tracker.initialize_video = lambda *a, **k: None
        tracker._build_detections_df = lambda results: None
        return tracker

    def _run(self, depth):
        seen = []
        tracker = self._tracker(seen)
        event = threading.Event()
        event.set()  # cancelled from the start
        cancellation.watch(event)
        try:
            with patch.object(bee_tracking, "FRAME_QUEUE_DEPTH", depth), \
                 patch.object(bee_tracking.cv2, "VideoCapture",
                              lambda _path: FakeCapture(1000)):
                with self.assertRaises(cancellation.JobCancelled):
                    tracker.process_video("clip.mp4")
        finally:
            cancellation.watch(None)
        return seen

    def test_a_cancelled_clip_stops_at_the_next_check(self):
        # The loop looks every 100 frames, so it stops after 100, not 1000.
        self.assertEqual(len(self._run(depth=8)), 100)
        self.assertEqual(len(self._run(depth=0)), 100)

    def test_the_reader_thread_is_stopped(self):
        self._run(depth=8)
        alive = [t for t in threading.enumerate() if t.name == "beemonitor-frame-reader"]
        self.assertEqual(alive, [])


if __name__ == "__main__":
    unittest.main()
