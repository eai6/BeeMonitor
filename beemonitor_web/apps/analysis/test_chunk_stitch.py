"""A bee that flies across a chunk boundary is one bee.

Long clips are split into contiguous frame ranges so no GPU invocation exceeds
the 1 h async cap, and each chunk runs its own tracker from id 1. The ids are
namespaced on merge so they don't collide — but that left the same animal
counted twice in `unique_tracks`, and split into two short visits instead of
one long one. Unlike matching a bee across two separate clips (which needs
re-ID and is not attempted), the chunks are frame-contiguous, so this join is
decidable from box overlap at the seam.
"""

from django.test import SimpleTestCase

from apps.analysis import chunk_stitch


def row(tid, frame, x1, y1, x2, y2):
    return {"track_id": tid, "frame": str(frame),
            "x1": str(x1), "y1": str(y1), "x2": str(x2), "y2": str(y2)}


class StitchTests(SimpleTestCase):
    def test_two_halves_of_one_flight_become_one_track(self):
        rows = [row("1", 298, 100, 100, 140, 140),
                row("1", 299, 102, 100, 142, 140),
                row("1000000", 300, 104, 101, 144, 141),
                row("1000000", 301, 106, 101, 146, 141)]

        joins = chunk_stitch.stitch(rows, [300])

        self.assertEqual(joins, 1)
        self.assertEqual(chunk_stitch.distinct_track_count(rows), 1)

    def test_two_different_bees_at_the_seam_stay_apart(self):
        rows = [row("1", 299, 100, 100, 140, 140),
                row("1000000", 300, 900, 700, 940, 740)]

        joins = chunk_stitch.stitch(rows, [300])

        self.assertEqual(joins, 0)
        self.assertEqual(chunk_stitch.distinct_track_count(rows), 2)

    def test_the_best_overlap_claims_its_partner(self):
        # Two bees cross the same seam close together: each must find its own
        # half. Track "1" sits at ~(100,100) and "2" at ~(300,300), and the
        # ids on the far side are deliberately crossed relative to position.
        rows = [row("1", 299, 100, 100, 140, 140),
                row("2", 299, 300, 300, 340, 340),
                row("1000000", 300, 302, 301, 342, 341),   # belongs with "2"
                row("1000001", 300, 101, 101, 141, 141)]   # belongs with "1"

        chunk_stitch.stitch(rows, [300])

        self.assertEqual(chunk_stitch.distinct_track_count(rows), 2)
        ids = {r["x1"]: r["track_id"] for r in rows}
        # The two halves near (100,·) share an id, and so do the two near
        # (300,·) — and the two groups differ.
        self.assertEqual(ids["100"], ids["101"])
        self.assertEqual(ids["300"], ids["302"])
        self.assertNotEqual(ids["100"], ids["300"])

    def test_a_track_that_ends_well_before_the_seam_is_not_a_candidate(self):
        rows = [row("1", 200, 100, 100, 140, 140),
                row("1000000", 300, 100, 100, 140, 140)]

        self.assertEqual(chunk_stitch.stitch(rows, [300]), 0)

    def test_a_dropped_frame_at_the_seam_still_joins(self):
        rows = [row("1", 297, 100, 100, 140, 140),
                row("1000000", 302, 103, 101, 143, 141)]

        self.assertEqual(chunk_stitch.stitch(rows, [300]), 1)

    def test_several_seams_chain_into_one_track(self):
        rows = [row("1", 299, 100, 100, 140, 140),
                row("1000000", 300, 101, 100, 141, 140),
                row("1000000", 599, 102, 100, 142, 140),
                row("2000000", 600, 103, 100, 143, 140)]

        chunk_stitch.stitch(rows, [300, 600])

        self.assertEqual(chunk_stitch.distinct_track_count(rows), 1)

    def test_an_unchunked_run_is_left_exactly_as_it_was(self):
        rows = [row("1", 10, 100, 100, 140, 140)]

        self.assertEqual(chunk_stitch.stitch(rows, []), 0)
        self.assertEqual(rows[0]["track_id"], "1")

    def test_rows_without_box_columns_are_skipped_not_crashed(self):
        rows = [{"track_id": "1", "frame": "299"},
                {"track_id": "1000000", "frame": "300"}]

        self.assertEqual(chunk_stitch.stitch(rows, [300]), 0)

    def test_distinct_count_of_nothing_is_zero(self):
        self.assertEqual(chunk_stitch.distinct_track_count([]), 0)
