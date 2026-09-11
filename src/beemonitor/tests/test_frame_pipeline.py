"""The reader thread must be invisible to everything downstream.

process_frame, BeeTracker and the MOG2 background model are all stateful and
order-dependent: each must see every frame exactly once, in order. Overlapping
decode with inference changes only WHEN a frame is decoded, never which frame
arrives when — so the threaded path and the inline path have to produce
byte-identical results. That equivalence is the entire safety argument for
Phase 4, so it is tested directly rather than inferred.
"""

import queue
import threading
import unittest
from unittest.mock import patch

import numpy as np

from beemonitor.tracking import bee_tracking


class FakeCapture:
    """Stands in for cv2.VideoCapture: hands out N distinguishable frames."""

    def __init__(self, count, fail_at=None):
        self.count = count
        self.fail_at = fail_at
        self.reads = 0
        self.released = False

    def release(self):
        self.released = True

    def read(self):
        if self.fail_at is not None and self.reads == self.fail_at:
            raise RuntimeError("decoder exploded")
        if self.reads >= self.count:
            return False, None
        # Frame i is filled with the value i, so identity survives the queue.
        frame = np.full((2, 2, 3), self.reads % 251, dtype=np.uint8)
        self.reads += 1
        return True, frame


def collect(depth, count, start=0, end=None, fail_at=None):
    with patch.object(bee_tracking, "FRAME_QUEUE_DEPTH", depth):
        cap = FakeCapture(count, fail_at=fail_at)
        return [(n, int(f[0, 0, 0])) for n, f in
                bee_tracking._iter_frames(cap, start, end)]


class FrameIterationTests(unittest.TestCase):
    def test_threaded_and_inline_paths_agree(self):
        self.assertEqual(collect(depth=8, count=50), collect(depth=0, count=50))

    def test_a_queue_smaller_than_the_video_still_yields_every_frame(self):
        """The consumer must not be able to lose frames to a full queue."""
        frames = collect(depth=2, count=40)

        self.assertEqual(len(frames), 40)
        self.assertEqual([n for n, _ in frames], list(range(40)))

    def test_frame_numbers_stay_absolute_when_starting_mid_video(self):
        """Chunked long videos rely on this: event timestamps and cross-chunk
        CSV merges are computed from the absolute frame number."""
        for depth in (0, 8):
            frames = collect(depth=depth, count=10, start=1000)
            self.assertEqual([n for n, _ in frames][:3], [1000, 1001, 1002], depth)

    def test_end_frame_is_exclusive_on_both_paths(self):
        for depth in (0, 8):
            frames = collect(depth=depth, count=100, start=0, end=10)
            self.assertEqual(len(frames), 10, depth)
            self.assertEqual(frames[-1][0], 9, depth)

    def test_an_empty_video_yields_nothing_and_does_not_hang(self):
        for depth in (0, 8):
            self.assertEqual(collect(depth=depth, count=0), [])

    def test_a_decoder_that_raises_ends_the_stream_instead_of_hanging(self):
        """The reader's finally-block posts the sentinel on every exit path."""
        frames = collect(depth=4, count=50, fail_at=20)

        self.assertEqual(len(frames), 20)
        self.assertEqual([n for n, _ in frames], list(range(20)))

    def test_abandoning_the_generator_stops_the_reader(self):
        """A consumer that raises mid-loop must not leave a thread wedged on a
        full queue — the generator's finally drains it."""
        before = threading.active_count()

        with patch.object(bee_tracking, "FRAME_QUEUE_DEPTH", 2):
            frames = bee_tracking._iter_frames(FakeCapture(10_000), 0, None)
            next(frames)
            next(frames)
            frames.close()

        for _ in range(50):
            if threading.active_count() <= before:
                break
            threading.Event().wait(0.1)
        self.assertLessEqual(threading.active_count(), before)

    def test_decode_time_is_attributed_on_both_paths(self):
        from beemonitor.core.profiling import PROFILER

        for depth in (0, 8):
            PROFILER.reset()
            collect(depth=depth, count=25)
            self.assertEqual(PROFILER.snapshot()["decode"]["calls"], 26, depth)
        PROFILER.reset()


class ProcessVideoEquivalenceTests(unittest.TestCase):
    """The loop level: process_frame must see the same calls, in the same
    order, whichever path fed it."""

    def _run(self, depth, count):
        from beemonitor.tracking.bee_tracking import BeeTracking

        # Built without __init__ — that loads YOLO weights, and none of this
        # exercises the detector. Only what process_video itself touches.
        tracker = object.__new__(BeeTracking)
        tracker.save_crops = False
        tracker.track_crop_counts = {}
        tracker.detections_df = None
        seen = []

        def fake_process_frame(frame, frame_num, visualize=False):
            seen.append((frame_num, int(frame[0, 0, 0])))
            return {"frame_num": frame_num, "detections": [], "tracks": [],
                    "mode": "motion_detection", "lookback_results": []}

        tracker.process_frame = fake_process_frame
        tracker.initialize_video = lambda *a, **k: None
        tracker._build_detections_df = lambda results: None

        with patch.object(bee_tracking, "FRAME_QUEUE_DEPTH", depth), \
             patch.object(bee_tracking.cv2, "VideoCapture",
                          lambda _path: FakeCapture(count)):
            df = tracker.process_video("clip.mp4")
        return seen, df

    def test_the_tracker_sees_an_identical_call_sequence(self):
        threaded, threaded_df = self._run(depth=8, count=60)
        inline, inline_df = self._run(depth=0, count=60)

        self.assertEqual(threaded, inline)
        self.assertEqual(threaded, [(i, i) for i in range(60)])
        self.assertEqual(len(threaded_df), len(inline_df))

    def test_no_frame_is_processed_twice(self):
        seen, _ = self._run(depth=4, count=200)

        self.assertEqual(len(seen), len(set(n for n, _ in seen)))


if __name__ == "__main__":
    unittest.main()
