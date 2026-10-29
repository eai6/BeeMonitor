"""One row per track, with species and marker (apps/pipelines/tracks.py), and
the CSV exporters built on the same identity (memory/60)."""

from datetime import datetime, timezone

from django.test import SimpleTestCase

from apps.pipelines import tracks

BY_TRACK = {
    "1": {"taxon": "Osmia lignaria", "taxon_confidence": 0.62, "taxon_votes": 30,
          "taxon_vote_share": 0.9, "marker": "4", "marker_votes": 12, "marker_vote_share": 0.8},
    "2": {"taxon": "Megachile rotundata", "taxon_confidence": 0.12, "taxon_votes": 5,
          "taxon_vote_share": 0.5},
}
FRAMES = [{"track_id": "1", "frame_number": "10", "detected": "1"},
          {"track_id": "1", "frame_number": "40", "detected": "1"},
          # The tracker kept predicting track 1 for 10 more frames after losing it.
          {"track_id": "1", "frame_number": "50", "detected": "0"},
          {"track_id": "2.0", "frame_number": "25", "detected": "1"},
          {"track_id": "3", "frame_number": "5", "detected": "1"}]
START = datetime(2026, 6, 1, 10, 0, tzinfo=timezone.utc)


class TrackRowsTests(SimpleTestCase):
    def test_one_row_per_track_with_detected_span_species_and_marker(self):
        rows = {r["track_id"]: r for r in tracks.track_rows(FRAMES, BY_TRACK, fps=10)}

        self.assertEqual(set(rows), {1, 2, 3})
        one = rows[1]
        self.assertEqual(list(one), tracks.TRACK_FIELDS)
        # Last DETECTED frame is 40, not the predicted 50.
        self.assertEqual((one["start_clip_sec"], one["end_clip_sec"], one["duration_sec"]),
                         (1.0, 4.0, 3.0))
        self.assertEqual((one["species"], one["marker_id"]), ("Osmia lignaria", "4"))

    def test_times_are_timestamps_when_the_clip_start_is_known(self):
        one = tracks.track_rows(FRAMES, BY_TRACK, fps=10, start=START)[0]

        self.assertEqual((one["start_time"], one["end_time"]),
                         ("2026-06-01T10:00:01.000+00:00", "2026-06-01T10:00:04.000+00:00"))

    def test_an_unknown_start_leaves_timestamps_blank(self):
        one = tracks.track_rows(FRAMES, BY_TRACK, fps=10)[0]

        self.assertEqual((one["start_time"], one["end_time"]), ("", ""))

    def test_no_floor_the_species_and_its_confidence_are_reported(self):
        two = tracks.track_rows(FRAMES, BY_TRACK, fps=10)[1]

        self.assertEqual((two["species"], two["species_confidence"]), ("Megachile rotundata", 0.12))
        self.assertNotIn("species_best_guess", two)

    def test_a_csv_from_before_the_detected_column_uses_every_row(self):
        old = [{"track_id": "1", "frame": "10"}, {"track_id": "1", "frame": "50"}]

        one = tracks.track_rows(old, {}, fps=10)[0]

        self.assertEqual(one["end_clip_sec"], 5.0)

    def test_interactions_name_both_insects(self):
        rows = tracks.primitive_with_identity("interactions", [
            {"start_frame": 0, "end_frame": 5, "track_id": 1, "with": 2, "relation": "near"},
            {"start_frame": 0, "end_frame": 5, "track_id": 1, "with": "reference_1",
             "relation": "in_reference"}], BY_TRACK, fps=10)

        self.assertEqual((rows[0]["species"], rows[0]["with_species"]),
                         ("Osmia lignaria", "Megachile rotundata"))
        self.assertEqual(rows[1]["with_species"], "")


class TrackingExportTests(SimpleTestCase):
    RAW = [{"frame": "10", "track_id": "1", "detected": "1", "x1": "1", "y1": "2", "x2": "3",
            "y2": "4", "cx": "2", "cy": "3", "confidence": "0.9", "source": "yolo",
            "mode": "tracking", "taxon": "bee", "bee_id": "", "bee_id_method": ""},
           {"frame": "11", "track_id": "1", "detected": "0", "x1": "1", "y1": "2", "x2": "3",
            "y2": "4", "cx": "2.5", "cy": "3", "confidence": "0.9", "source": "yolo",
            "mode": "tracking", "taxon": "bee", "bee_id": "", "bee_id_method": ""}]

    def test_columns_are_the_tracking_contract(self):
        rows = tracks.tracking_export(self.RAW, BY_TRACK, fps=10, start=START)

        self.assertEqual(list(rows[0]), tracks.TRACKING_FIELDS)
        self.assertEqual((rows[0]["clip_sec"], rows[0]["timestamp"]),
                         (1.0, "2026-06-01T10:00:01.000+00:00"))
        self.assertEqual(rows[0]["species"], "Osmia lignaria")

    def test_a_predicted_row_reports_no_confidence(self):
        rows = tracks.tracking_export(self.RAW, BY_TRACK, fps=10)

        self.assertEqual([r["detected"] for r in rows], [1, 0])
        self.assertEqual([r["confidence"] for r in rows], ["0.9", ""])

    def test_detections_rename_taxon_to_class_and_drop_internal_columns(self):
        rows = tracks.detections_export(
            [{"frame": "4", "x1": "1", "y1": "1", "x2": "2", "y2": "2", "cx": "1.5",
              "cy": "1.5", "confidence": "0.7", "source": "yolo", "taxon": "bee",
              "mode": "tracking"}], fps=10)

        self.assertEqual(list(rows[0]), tracks.DETECTION_FIELDS)
        self.assertEqual(rows[0]["class"], "bee")
