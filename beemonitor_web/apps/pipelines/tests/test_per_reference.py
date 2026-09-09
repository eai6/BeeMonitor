"""Analyzers report per reference, not just "something".

compute_visitation used to ask in_any_box(x, y, boxes) — one boolean for every
reference at once — so it could say a track visited SOMETHING and never WHICH.
For four flower treatments that is the whole question: they are only interesting
compared against each other.
"""

import unittest

import pandas as pd

from apps.pipelines import ops


def tracks(*rows):
    """rows of (tid, frame, x, y) as the tidy frame analyzers consume."""
    return pd.DataFrame(rows, columns=["tid", "frame", "x", "y"])


NEST_3 = {"id": 3, "box": [0.0, 0.0, 0.2, 0.2]}
NEST_7 = {"id": 7, "box": [0.8, 0.8, 1.0, 1.0]}
ROI = {"nest_layout": [NEST_3, NEST_7]}


class ReferenceIdentityTests(unittest.TestCase):
    def test_a_nest_keeps_the_id_from_the_layout(self):
        refs = ops.roi_references(ROI)

        self.assertEqual([r["id"] for r in refs], ["nest_3", "nest_7"])
        self.assertEqual(refs[0]["label"], "Nest 3")

    def test_a_named_region_uses_its_name(self):
        """Nothing lets a user name one yet — but when the editor learns to,
        the analyzers already carry it."""
        refs = ops.roi_references({"regions": [{"box": [0, 0, 1, 1], "name": "full UV"}]})

        self.assertEqual(refs[0]["label"], "full UV")

    def test_an_unnamed_region_falls_back_to_its_index(self):
        refs = ops.roi_references({"regions": [{"box": [0, 0, 1, 1]}, {"box": [0, 0, 1, 1]}]})

        self.assertEqual([r["label"] for r in refs], ["Region 1", "Region 2"])

    def test_the_hotel_is_its_own_reference(self):
        refs = ops.roi_references({"hotel_roi": [0, 0, 1, 1], "nest_layout": [NEST_3]})

        self.assertEqual([r["id"] for r in refs], ["hotel", "nest_3"])

    def test_which_reference_names_the_one_containing_a_point(self):
        refs = ops.roi_references(ROI)

        self.assertEqual(ops.which_reference(0.1, 0.1, refs)["id"], "nest_3")
        self.assertEqual(ops.which_reference(0.9, 0.9, refs)["id"], "nest_7")
        self.assertIsNone(ops.which_reference(0.5, 0.5, refs))

    def test_a_polygon_beats_its_bounding_box(self):
        """A bee on the grass beside a round trap is not inside it."""
        refs = ops.roi_references({"regions": [{
            "box": [0.0, 0.0, 1.0, 1.0],
            "points": [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]],  # diamond
        }]})

        self.assertIsNotNone(ops.which_reference(0.5, 0.5, refs))   # centre
        self.assertIsNone(ops.which_reference(0.02, 0.02, refs))    # corner


class PerReferenceVisitationTests(unittest.TestCase):
    def setUp(self):
        self.refs = ops.roi_references(ROI)

    def test_visits_are_attributed_to_the_reference_they_happened_in(self):
        df = tracks(
            *[(1, f, 0.1, 0.1) for f in range(10)],      # track 1 in nest 3
            *[(2, f, 0.9, 0.9) for f in range(10)],      # track 2 in nest 7
        )

        out = ops.compute_visitation(df, self.refs, fps=10)

        by_id = {r["id"]: r for r in out["per_reference"]}
        self.assertEqual(by_id["nest_3"]["visits"], 1)
        self.assertEqual(by_id["nest_7"]["visits"], 1)
        self.assertEqual(by_id["nest_3"]["visitors"], 1)

    def test_moving_between_references_is_two_visits_not_one(self):
        """Otherwise a bee crossing from tube 3 to tube 7 reads as one long stay
        in neither."""
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(5)],
                    *[(1, f, 0.9, 0.9) for f in range(5, 10)])

        out = ops.compute_visitation(df, self.refs, fps=10)

        self.assertEqual(out["total_visits"], 2)
        by_id = {r["id"]: r for r in out["per_reference"]}
        self.assertEqual(by_id["nest_3"]["visits"], 1)
        self.assertEqual(by_id["nest_7"]["visits"], 1)

    def test_a_long_gap_in_the_same_reference_is_a_second_visit(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(5)],
                    *[(1, f, 0.1, 0.1) for f in range(100, 105)])

        out = ops.compute_visitation(df, self.refs, fps=10, gap_frames=15)

        self.assertEqual(out["total_visits"], 2)

    def test_a_reference_nothing_visited_is_still_listed(self):
        """"Nothing visited the control" is a result; dropping the row leaves
        the reader to notice an absence."""
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)])

        out = ops.compute_visitation(df, self.refs, fps=10)

        by_id = {r["id"]: r for r in out["per_reference"]}
        self.assertIn("nest_7", by_id)
        self.assertEqual(by_id["nest_7"]["visits"], 0)
        self.assertEqual(by_id["nest_7"]["visitors"], 0)

    def test_the_busiest_reference_comes_first(self):
        df = tracks(*[(1, f, 0.9, 0.9) for f in range(10)],
                    *[(2, f, 0.9, 0.9) for f in range(20, 30)])

        out = ops.compute_visitation(df, self.refs, fps=10)

        self.assertEqual(out["per_reference"][0]["id"], "nest_7")

    def test_dwell_is_split_between_references(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)],
                    *[(1, f, 0.9, 0.9) for f in range(10, 40)])

        out = ops.compute_visitation(df, self.refs, fps=10)

        by_id = {r["id"]: r for r in out["per_reference"]}
        self.assertAlmostEqual(by_id["nest_3"]["dwell_sec"], 1.0, places=2)
        self.assertAlmostEqual(by_id["nest_7"]["dwell_sec"], 3.0, places=2)

    def test_the_totals_still_agree_with_the_breakdown(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)],
                    *[(2, f, 0.9, 0.9) for f in range(10)])

        out = ops.compute_visitation(df, self.refs, fps=10)

        self.assertEqual(out["total_visits"],
                         sum(r["visits"] for r in out["per_reference"]))
        self.assertEqual(out["unique_visitors"], 2)

    def test_points_outside_every_reference_count_for_nothing(self):
        df = tracks(*[(1, f, 0.5, 0.5) for f in range(20)])

        out = ops.compute_visitation(df, self.refs, fps=10)

        self.assertEqual(out["total_visits"], 0)
        self.assertEqual(out["unique_visitors"], 0)

    def test_bare_boxes_from_an_older_pipeline_still_work(self):
        """They just get numbers for names rather than refusing to compute."""
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)])

        out = ops.compute_visitation(df, ops.roi_shapes(ROI), fps=10)

        self.assertEqual(out["total_visits"], 1)
        self.assertEqual(out["per_reference"][0]["id"], "region_1")


if __name__ == "__main__":
    unittest.main()


class PerReferenceInteractionTests(unittest.TestCase):
    """Interactions already carried a reference_id — nothing grouped by it."""

    def _df(self, *rows):
        return pd.DataFrame(rows, columns=[
            "interaction_type", "organism_track_id", "partner_track_id",
            "reference_id", "duration_seconds", "start_frame"])

    def test_interactions_are_grouped_by_reference(self):
        df = self._df(
            ("organism-to-reference", 1, "", "nest_3", 4.0, 10),
            ("organism-to-reference", 2, "", "nest_3", 2.0, 40),
            ("organism-to-reference", 3, "", "nest_7", 1.5, 60),
        )

        out = ops.summarize_interactions(df)

        by_id = {r["id"]: r for r in out["per_reference"]}
        self.assertEqual(by_id["nest_3"]["interactions"], 2)
        self.assertEqual(by_id["nest_3"]["partners"], 2)
        self.assertEqual(by_id["nest_7"]["interactions"], 1)

    def test_worker_ids_get_the_same_labels_as_the_layout(self):
        df = self._df(("organism-to-reference", 1, "", "nest_3", 1.0, 1))

        out = ops.summarize_interactions(df)

        self.assertEqual(out["per_reference"][0]["label"], "Nest 3")

    def test_insect_to_insect_has_no_reference_and_is_not_counted_in_one(self):
        df = self._df(
            ("organism-to-organism", 1, 2, "", 3.0, 10),
            ("organism-to-reference", 1, "", "nest_3", 1.0, 20),
        )

        out = ops.summarize_interactions(df)

        self.assertEqual(len(out["per_reference"]), 1)
        self.assertEqual(out["per_reference"][0]["id"], "nest_3")
        self.assertEqual(out["organism_organism"], 1)

    def test_durations_add_up_per_reference(self):
        df = self._df(
            ("organism-to-reference", 1, "", "nest_3", 4.25, 10),
            ("organism-to-reference", 2, "", "nest_3", 1.75, 40),
        )

        out = ops.summarize_interactions(df)

        self.assertEqual(out["per_reference"][0]["duration_sec"], 6.0)

    def test_an_empty_frame_yields_an_empty_breakdown(self):
        out = ops.summarize_interactions(None)

        self.assertEqual(out["interaction_count"], 0)
        self.assertEqual(out.get("per_reference", []), [])
