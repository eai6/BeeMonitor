"""Raw detections export from the tracking pass.

Every frame's detector output was already being accumulated in
``process_video``'s ``results`` and then discarded — the flatten loop only read
``result['tracks']``. These tests pin the second flatten loop that turns it into
a real table, and in particular that a detection the tracker never associated
into a confirmed track still appears (that gap is the whole reason the raw
table exists).

Runs on CPU with no model, by calling the flattener directly on hand-built
frame results.
"""

import unittest

import pandas as pd

from beemonitor.tracking.bee_tracking import BeeTracking


def _det(x1, y1, x2, y2, conf=0.9, source="yolo", taxon="bee"):
    """One entry in the flat format process_frame builds."""
    return [x1, y1, x2, y2, conf, source, taxon]


class DetectionsExportTests(unittest.TestCase):
    def setUp(self):
        # __init__ loads a YOLO model, which we neither have nor need here —
        # the flattener is a pure function of the frame results.
        self.flatten = BeeTracking._build_detections_df.__get__(
            object.__new__(BeeTracking), BeeTracking)

    def test_one_row_per_detection_per_frame(self):
        results = [
            {"frame_num": 0, "mode": "tracking",
             "detections": [_det(0, 0, 10, 10), _det(20, 20, 30, 30)], "tracks": []},
            {"frame_num": 1, "mode": "tracking",
             "detections": [_det(0, 0, 10, 10)], "tracks": []},
        ]

        df = self.flatten(results)

        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["frame"]), [0, 0, 1])
        self.assertEqual(list(BeeTracking.DETECTION_COLUMNS), list(df.columns))

    def test_centroids_are_derived_from_the_box(self):
        df = self.flatten([
            {"frame_num": 7, "mode": "tracking",
             "detections": [_det(10, 20, 30, 60)], "tracks": []},
        ])

        self.assertEqual(df.iloc[0]["cx"], 20.0)
        self.assertEqual(df.iloc[0]["cy"], 40.0)

    def test_includes_detections_that_never_became_tracks(self):
        """The tracked table would drop this row; the raw table must keep it."""
        results = [
            {"frame_num": 0, "mode": "tracking",
             "detections": [_det(0, 0, 10, 10), _det(500, 500, 510, 510, conf=0.3)],
             "tracks": [{"track_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
                         "cx": 5, "cy": 5}]},
        ]

        df = self.flatten(results)

        self.assertEqual(len(df), 2)
        self.assertIn(0.3, list(df["confidence"]))

    def test_metadata_columns_are_carried(self):
        df = self.flatten([
            {"frame_num": 0, "mode": "motion_detection",
             "detections": [_det(0, 0, 1, 1, conf=0.42, source="blob", taxon="wasp")],
             "tracks": []},
        ])

        row = df.iloc[0]
        self.assertEqual(row["confidence"], 0.42)
        self.assertEqual(row["source"], "blob")
        self.assertEqual(row["taxon"], "wasp")
        self.assertEqual(row["mode"], "motion_detection")

    def test_lookback_frames_are_included_and_sorted(self):
        """Lookback results are appended out of order by process_video."""
        results = [
            {"frame_num": 5, "mode": "tracking", "detections": [_det(0, 0, 1, 1)],
             "tracks": []},
            {"frame_num": 2, "mode": "lookback", "detections": [_det(0, 0, 1, 1)],
             "tracks": []},
        ]

        df = self.flatten(results)

        self.assertEqual(list(df["frame"]), [2, 5])

    def test_short_rows_are_tolerated(self):
        """A detector emitting only a box shouldn't break the export."""
        df = self.flatten([
            {"frame_num": 0, "mode": "tracking",
             "detections": [[0, 0, 10, 10], [1, 2]], "tracks": []},
        ])

        self.assertEqual(len(df), 1)  # the 2-element row is skipped
        self.assertEqual(df.iloc[0]["confidence"], 0.0)
        self.assertEqual(df.iloc[0]["taxon"], "bee")

    def test_empty_input_returns_the_typed_empty_frame(self):
        for results in ([], [{"frame_num": 0, "mode": "tracking",
                              "detections": [], "tracks": []}]):
            df = self.flatten(results)
            self.assertTrue(df.empty)
            self.assertEqual(list(df.columns), list(BeeTracking.DETECTION_COLUMNS))


class AnalysisResultsExportTests(unittest.TestCase):
    def test_detections_csv_is_written_only_when_present(self):
        import tempfile
        from pathlib import Path

        from beemonitor.core.analysis_results import AnalysisResults

        detections = pd.DataFrame([{"frame": 0, "x1": 0, "y1": 0, "x2": 1, "y2": 1}])
        tracks = pd.DataFrame([{"frame": 0, "track_id": 1}])

        with tempfile.TemporaryDirectory() as out:
            AnalysisResults(events=pd.DataFrame(), tracks=tracks, nests={},
                            video_path="clip.mp4", detections=detections).to_csv(out)
            self.assertTrue((Path(out) / "clip_detections.csv").exists())

        with tempfile.TemporaryDirectory() as out:
            AnalysisResults(events=pd.DataFrame(), tracks=tracks, nests={},
                            video_path="clip.mp4").to_csv(out)
            self.assertFalse((Path(out) / "clip_detections.csv").exists())


if __name__ == "__main__":
    unittest.main()
