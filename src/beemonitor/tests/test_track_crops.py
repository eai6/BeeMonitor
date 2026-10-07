"""Per-track crops: padded, one per detected frame, tentative frames kept.

Crops are what a species or marker model reads later, and the point of the
system is small bees — so a crop must include the margin around the body, and
every frame the bee was seen in, not just the first few as it flies in.
"""

import os
import tempfile
import unittest

import numpy as np

from beemonitor.tracking.bee_tracking import BeeTracking, padded_box
from beemonitor.tracking.mot.bee_tracker import Track


class PaddedBoxTests(unittest.TestCase):
    def test_grows_by_a_fraction_of_the_box(self):
        # 100x40 box, 25% -> 25 px each side horizontally, 16 px floor vertically
        self.assertEqual(padded_box((200, 200, 300, 240), 1000, 1000, 0.25, 16),
                         (175, 184, 325, 256))

    def test_small_boxes_get_the_pixel_floor(self):
        self.assertEqual(padded_box((100, 100, 110, 110), 1000, 1000, 0.25, 16),
                         (84, 84, 126, 126))

    def test_clamped_to_the_frame(self):
        self.assertEqual(padded_box((0, 0, 50, 50), 60, 60, 0.25, 16), (0, 0, 60, 60))

    def test_empty_or_bad_box_is_none(self):
        self.assertIsNone(padded_box((500, 500, 600, 600), 100, 100))
        self.assertIsNone(padded_box(("a", 0, 1, 1), 100, 100))


class _FakeTracker:
    def __init__(self, tracks):
        self.tracks = tracks


def _tracking(tmp, tracks, crops_per_track=0, keep_sharpest=0):
    # Built without __init__ — that loads YOLO weights; crops don't need it.
    bt = object.__new__(BeeTracking)
    bt.save_crops = True
    bt.crop_output_dir = tmp
    bt.crops_per_track = crops_per_track
    bt.crops_keep_sharpest = keep_sharpest
    bt._crop_heaps = {}
    bt._crop_sharpness = {}
    bt.crop_padding = 0.25
    bt.crop_min_padding_px = 16
    bt.track_crop_counts = {}
    bt.tracker = _FakeTracker(tracks)
    return bt


def _track(min_hits=3):
    return Track((100, 100, 140, 130, 0.9, "yolo", "bee"), frame_num=0,
                 min_hits=min_hits, max_age=10)


def _saved(tmp):
    out = {}
    for root, _, files in os.walk(tmp):
        for f in files:
            out.setdefault(os.path.basename(root), []).append(f)
    return {k: sorted(v) for k, v in out.items()}


class SaveTrackCropsTests(unittest.TestCase):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def test_tentative_frames_are_written_once_the_track_is_confirmed(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=3)
            bt = _tracking(tmp, [track])
            bt._save_track_crops(self.frame, 0)          # hits=1, tentative
            track.update((102, 101, 142, 131, 0.9, "yolo", "bee"), 1)
            bt._save_track_crops(self.frame, 1)          # hits=2, tentative
            self.assertEqual(_saved(tmp), {})
            track.update((104, 102, 144, 132, 0.9, "yolo", "bee"), 2)
            bt._save_track_crops(self.frame, 2)          # hits=3, confirmed
            name = f"track_{track.id:04d}"
            self.assertEqual(_saved(tmp), {name: [
                "frame_000000.jpg", "frame_000001.jpg", "frame_000002.jpg"]})

    def test_a_predicted_frame_gets_no_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=1)
            bt = _tracking(tmp, [track])
            bt._save_track_crops(self.frame, 0)
            track.predict()                              # missed this frame
            bt._save_track_crops(self.frame, 1)
            self.assertEqual(_saved(tmp), {f"track_{track.id:04d}": ["frame_000000.jpg"]})

    def test_crops_are_padded(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=1)
            _tracking(tmp, [track])._save_track_crops(self.frame, 0)
            path = os.path.join(tmp, f"track_{track.id:04d}", "frame_000000.jpg")
            h, w = cv2.imread(path).shape[:2]
            # 40x30 box + 16 px floor on every side
            self.assertEqual((w, h), (40 + 32, 30 + 32))

    def test_cap_still_honoured_when_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=1)
            bt = _tracking(tmp, [track], crops_per_track=2)
            for f in range(5):
                track.update((100, 100, 140, 130, 0.9, "yolo", "bee"), f)
                bt._save_track_crops(self.frame, f)
            self.assertEqual(len(_saved(tmp)[f"track_{track.id:04d}"]), 2)



class KeepSharpestTests(unittest.TestCase):
    """Without identification a track keeps only its N sharpest crops."""

    @staticmethod
    def _frame(sharp):
        # Checkerboard where the box is (sharp) or flat grey (blurry).
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        if sharp:
            board = (np.indices((80, 80)).sum(axis=0) % 2 * 255).astype(np.uint8)
            frame[80:160, 80:160] = board[..., None]
        return frame

    def test_only_the_sharpest_stay_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=1)
            bt = _tracking(tmp, [track], keep_sharpest=2)
            sharp_frames = {3, 5}
            for f in range(8):
                track.update((100, 100, 140, 130, 0.9, "yolo", "bee"), f)
                bt._save_track_crops(self._frame(f in sharp_frames), f)
            self.assertEqual(_saved(tmp)[f"track_{track.id:04d}"],
                             ["frame_000003.jpg", "frame_000005.jpg"])
            self.assertEqual(bt.track_crop_counts[track.id], 2)

    def test_scores_are_written_beside_the_crops(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=1)
            bt = _tracking(tmp, [track], keep_sharpest=1)
            for f in range(3):
                track.update((100, 100, 140, 130, 0.9, "yolo", "bee"), f)
                bt._save_track_crops(self._frame(f == 1), f)
            bt._write_crop_sharpness()
            with open(os.path.join(tmp, "sharpness.csv")) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual([r["crop"] for r in rows],
                             [os.path.join(f"track_{track.id:04d}", "frame_000001.jpg")])
            self.assertGreater(float(rows[0]["sharpness"]), 0)

    def test_zero_keeps_every_crop(self):
        with tempfile.TemporaryDirectory() as tmp:
            track = _track(min_hits=1)
            bt = _tracking(tmp, [track], keep_sharpest=0)
            for f in range(5):
                track.update((100, 100, 140, 130, 0.9, "yolo", "bee"), f)
                bt._save_track_crops(self._frame(False), f)
            self.assertEqual(len(_saved(tmp)[f"track_{track.id:04d}"]), 5)


if __name__ == "__main__":
    unittest.main()
