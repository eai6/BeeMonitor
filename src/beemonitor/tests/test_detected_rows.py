"""Which tracking rows were detected and which were predicted (memory/60).

A confirmed track keeps a row every frame while it is lost — the tracker's
prediction — until it dies. The platform must know which rows those are: a
track's real start and end are its first and last DETECTED rows, and a
predicted row has no detector confidence to report.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from beemonitor.core.analysis_results import AnalysisResults
from beemonitor.tracking.mot.bee_tracker import BeeTracker, Track
from beemonitor.tracking.mot.external import ExternalTracker


def box(cx, cy, size=50, conf=0.8):
    h = size / 2
    return (cx - h, cy - h, cx + h, cy + h, conf, "yolo", "bee")


class BeeTrackerDetectedTests(unittest.TestCase):
    def setUp(self):
        Track.reset_id_counter()
        self.t = BeeTracker(fps=25.0, frame_height=1080, max_age_seconds=1.0,
                            min_hits_seconds=0.0, max_resurrection_seconds=0, bee_size=50)

    def test_matched_rows_are_detected_and_carry_their_confidence(self):
        for f in range(5):
            out = self.t.update([box(300 + 5 * f, 300, conf=0.7)], f)
        self.assertEqual(out[0]["detected"], 1)
        self.assertAlmostEqual(out[0]["confidence"], 0.7)

    def test_rows_while_lost_are_predicted_with_no_confidence(self):
        for f in range(5):
            self.t.update([box(300, 300)], f)
        for f in range(5, 10):
            out = self.t.update([], f)
            self.assertEqual(len(out), 1)          # still reported while it coasts
            self.assertEqual(out[0]["detected"], 0)
            self.assertIsNone(out[0]["confidence"])
        out = self.t.update([box(300, 300, conf=0.6)], 10)  # seen again
        self.assertEqual(out[0]["detected"], 1)
        self.assertAlmostEqual(out[0]["confidence"], 0.6)

    def test_active_tracks_say_the_same(self):
        for f in range(3):
            self.t.update([box(300, 300)], f)
        self.t.update([], 3)
        self.assertEqual([r["detected"] for r in self.t.get_active_tracks()], [0])

    def test_backfilled_frames_were_detections(self):
        Track.reset_id_counter()
        t = BeeTracker(fps=25.0, frame_height=1080, max_age_seconds=3.0,
                       min_hits_seconds=0.2, max_resurrection_seconds=0, bee_size=50)
        backfilled = []
        for f in range(8):
            t.update([box(300 + 5 * f, 300)], f)
            backfilled += [r for _bf, r in t.backfill]
        self.assertTrue(backfilled)
        self.assertEqual({r["detected"] for r in backfilled}, {1})


class ExternalTrackerDetectedTests(unittest.TestCase):
    def test_reported_rows_say_detected(self):
        t = ExternalTracker("bytetrack", {}, fps=20, frame_size=(640, 480))
        t.frame = np.zeros((480, 640, 3), np.uint8)
        out = []
        for f in range(5):
            out = t.update([[10 + f * 5, 100, 50 + f * 5, 140, 0.9, "yolo", "bee"]], f)
        self.assertTrue(out)
        self.assertEqual({r["detected"] for r in out}, {1})
        self.assertAlmostEqual(out[0]["confidence"], 0.9, places=5)


class NoEventsFileTests(unittest.TestCase):
    """The worker computes no events; then no events file is written at all."""

    def test_events_none_writes_tracks_but_no_events_file(self):
        tracks = pd.DataFrame([{"frame": 0, "track_id": 1, "detected": 1}])
        with tempfile.TemporaryDirectory() as out:
            r = AnalysisResults(events=None, tracks=tracks, nests={}, video_path="clip.mp4")
            r.to_csv(out)
            self.assertFalse((Path(out) / "clip_events.csv").exists())
            self.assertTrue((Path(out) / "clip_tracking_results.csv").exists())
            self.assertEqual(r.get_statistics()["total_events"], 0)
            self.assertEqual(r.get_statistics()["total_tracks"], 1)


if __name__ == "__main__":
    unittest.main()
