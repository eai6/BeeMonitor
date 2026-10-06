"""One row per track, with species and marker (apps/pipelines/tracks.py)."""

from django.test import SimpleTestCase

from apps.pipelines import tracks

BY_TRACK = {
    "1": {"taxon": "Osmia lignaria", "taxon_confidence": 0.62, "taxon_votes": 30,
          "taxon_vote_share": 0.9, "marker": "4", "marker_votes": 12, "marker_vote_share": 0.8},
    "2": {"taxon": "Megachile rotundata", "taxon_confidence": 0.12, "taxon_votes": 5,
          "taxon_vote_share": 0.5},
}
FRAMES = [{"track_id": "1", "frame_number": "10", "class": "bee"},
          {"track_id": "1", "frame_number": "40", "class": "bee"},
          {"track_id": "2.0", "frame_number": "25", "class": "bee"},
          {"track_id": "3", "frame_number": "5", "class": "wasp"}]


class TrackRowsTests(SimpleTestCase):
    def test_one_row_per_track_with_span_species_and_marker(self):
        rows = {r["track_id"]: r for r in tracks.track_rows(FRAMES, BY_TRACK, 0.25, fps=10)}
        self.assertEqual(set(rows), {1, 2, 3})
        one = rows[1]
        self.assertEqual((one["first_frame"], one["last_frame"], one["frames_seen"]), (10, 40, 2))
        self.assertEqual((one["start_sec"], one["duration_sec"]), (1.0, 3.0))
        self.assertEqual((one["species"], one["marker_id"]), ("Osmia lignaria", "4"))

    def test_the_species_floor_keeps_the_best_guess(self):
        two = tracks.track_rows(FRAMES, BY_TRACK, 0.25)[1]
        self.assertEqual((two["species"], two["species_best_guess"]),
                         ("unidentified", "Megachile rotundata"))
        self.assertEqual(tracks.track_rows(FRAMES, BY_TRACK, 0.0)[1]["species"],
                         "Megachile rotundata")

    def test_tracking_rows_take_identity_and_drop_the_raw_vote(self):
        rows = tracks.with_identity([{"track_id": "1", "frame": 3, "taxon": "x"}], BY_TRACK)
        self.assertNotIn("taxon", rows[0])
        self.assertEqual(rows[0]["species"], "Osmia lignaria")

    def test_interactions_name_both_insects(self):
        rows = tracks.primitive_with_identity("interactions", [
            {"a": 1, "a_kind": "organism", "b": 2, "b_kind": "organism"},
            {"a": 1, "a_kind": "organism", "b": "tube 1", "b_kind": "reference"}],
            BY_TRACK, 0.25)
        self.assertEqual((rows[0]["species"], rows[0]["b_species"]),
                         ("Osmia lignaria", "unidentified"))
        self.assertNotIn("b_species", rows[1])

    def test_no_identification_leaves_rows_untouched(self):
        rows = [{"track_id": "1"}]
        self.assertIs(tracks.with_identity(rows, {}), rows)

    def test_floor_comes_from_the_identify_species_node(self):
        steps = [{"block_type": "identify.species", "config": {"min_mean_confidence": "0.3"}}]
        self.assertEqual(tracks.species_floor(steps), 0.3)
        self.assertEqual(tracks.species_floor([]), 0.0)
