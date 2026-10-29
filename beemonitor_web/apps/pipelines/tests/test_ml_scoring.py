"""Optional ML scoring of events (memory/60).

The feature code is a copy of the GPU worker's (importing the worker's package
pulls in torch); these hold the copy to the same numbers, and check that
scoring only ever adds a column — it never drops a row.
"""

import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase

from apps.pipelines import ml_scoring

CENTROIDS = [(100.0, 200.0), (104.0, 199.0), (110.0, 196.0), (121.0, 190.0),
             (130.0, 181.0), (133.0, 180.0), (150.0, 170.0)]
FRAMES = list(range(10, 17))
NEST = (90.0, 190.0, 110.0, 210.0)


class FeatureParityTests(SimpleTestCase):
    def test_the_copy_matches_the_workers_features(self):
        try:
            from beemonitor.processing.event_processor import EventProcessor
        except Exception as exc:  # noqa: BLE001 - the web image has no torch
            self.skipTest(f"worker package not importable here: {exc}")

        bboxes = [(x - 5, y - 5, x + 5, y + 5) for x, y in CENTROIDS]
        for action, is_entry in (("Entry", True), ("Exit", False)):
            worker = EventProcessor._extract_trajectory_features(
                EventProcessor.__new__(EventProcessor),
                (7, CENTROIDS, bboxes, FRAMES), action, "n1", {"nests": {"n1": NEST}})
            ours = ml_scoring.trajectory_features(CENTROIDS, FRAMES, NEST, is_entry)

            self.assertEqual(list(ours), list(worker))
            for key in worker:
                self.assertAlmostEqual(float(ours[key]), float(worker[key]), places=9, msg=key)

    def test_twenty_features_in_the_trained_order(self):
        feats = ml_scoring.trajectory_features(CENTROIDS, FRAMES, NEST, True)

        self.assertEqual(len(feats), 20)
        self.assertEqual(feats["is_entry"], 1)
        self.assertEqual(feats["trajectory_length"], 7)


class _Forest:
    def predict_proba(self, df):
        import numpy as np

        # Probability = the share of path that is displacement, a stand-in.
        return np.column_stack([np.zeros(len(df)), np.linspace(0.2, 0.8, len(df))])


def _tracking_csv():
    path = Path(tempfile.mkdtemp()) / "tracking.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "track_id", "x1", "y1", "x2", "y2",
                                           "cx", "cy", "detected"])
        w.writeheader()
        for f in range(10):
            x = 200 + 30 * f
            w.writerow({"frame": f, "track_id": 1, "x1": x - 10, "y1": 190, "x2": x + 10,
                        "y2": 210, "cx": x, "cy": 200, "detected": 1})
    return str(path)


class ScoreTests(SimpleTestCase):
    REFS = [{"id": "reference_1", "box": (0.09, 0.17, 0.12, 0.2), "points": None}]
    EXTRA = {"video": SimpleNamespace(width=1920, height=1080)}

    def _rows(self):
        return [{"frame": 0, "track_id": 1, "action": "exit", "reference": "reference_1"},
                {"frame": 0, "track_id": 1, "action": "stay", "reference": "reference_1"},
                {"frame": 9, "track_id": 1, "action": "enter", "reference": "reference_1"}]

    def test_scores_exits_and_entries_and_drops_nothing(self):
        rows = self._rows()
        with mock.patch.object(ml_scoring, "_model",
                               return_value=(_Forest(), list(ml_scoring.trajectory_features(
                                   CENTROIDS, FRAMES, NEST, True)))):
            note = ml_scoring.score(rows, {"tracking_csv_path": _tracking_csv()},
                                    self.REFS, self.EXTRA)

        self.assertIsNone(note)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r.get("ml_confidence") for r in rows], [0.2, None, 0.8])

    def test_an_unloadable_model_says_so_and_changes_nothing(self):
        rows = self._rows()
        with mock.patch.object(ml_scoring, "_model", return_value=None):
            note = ml_scoring.score(rows, {"tracking_csv_path": _tracking_csv()},
                                    self.REFS, self.EXTRA)

        self.assertIn("could not be loaded", note)
        self.assertFalse(any("ml_confidence" in r for r in rows))

    def test_the_real_classifier_loads_and_scores(self):
        if ml_scoring._model() is None:
            self.skipTest("classifier or scikit-learn not available here")
        rows = self._rows()

        ml_scoring.score(rows, {"tracking_csv_path": _tracking_csv()}, self.REFS, self.EXTRA)

        for r in (rows[0], rows[2]):
            self.assertGreaterEqual(r["ml_confidence"], 0.0)
            self.assertLessEqual(r["ml_confidence"], 1.0)
