"""Analyzers report per reference, not just "something".

For four flower treatments that is the whole question: they are only
interesting compared against each other. Time in each reference is the
interactions table's ``in_reference`` rows (memory/60), grouped by reference.
"""

import unittest

import pandas as pd

from apps.pipelines import ops, primitives


def tracks(*rows):
    """rows of (tid, frame, x, y) as the tidy frame analyzers consume."""
    return pd.DataFrame(rows, columns=["tid", "frame", "x", "y"])


NEST_3 = {"id": 3, "box": [0.0, 0.0, 0.2, 0.2]}
NEST_7 = {"id": 7, "box": [0.8, 0.8, 1.0, 1.0]}
ROI = {"nest_layout": [NEST_3, NEST_7]}


class ReferenceIdentityTests(unittest.TestCase):
    def test_a_nest_keeps_the_id_from_the_layout(self):
        refs = ops.roi_references(ROI)

        self.assertEqual([r["id"] for r in refs], ["reference_3", "reference_7"])
        self.assertEqual(refs[0]["label"], "Reference 3")

    def test_a_named_region_uses_its_name(self):
        """Nothing lets a user name one yet — but when the editor learns to,
        the analyzers already carry it."""
        refs = ops.roi_references({"regions": [{"box": [0, 0, 1, 1], "name": "full UV"}]})

        self.assertEqual(refs[0]["label"], "full UV")

    def test_an_unnamed_region_falls_back_to_its_index(self):
        refs = ops.roi_references({"regions": [{"box": [0, 0, 1, 1]}, {"box": [0, 0, 1, 1]}]})

        self.assertEqual([r["label"] for r in refs], ["Reference 1", "Reference 2"])

    def test_the_hotel_roi_is_not_a_reference(self):
        """It is the area tracking runs over (memory/60)."""
        refs = ops.roi_references({"hotel_roi": [0, 0, 1, 1], "nest_layout": [NEST_3]})

        self.assertEqual([r["id"] for r in refs], ["reference_3"])

    def test_which_reference_names_the_one_containing_a_point(self):
        refs = ops.roi_references(ROI)

        self.assertEqual(ops.which_reference(0.1, 0.1, refs)["id"], "reference_3")
        self.assertEqual(ops.which_reference(0.9, 0.9, refs)["id"], "reference_7")
        self.assertIsNone(ops.which_reference(0.5, 0.5, refs))

    def test_a_polygon_beats_its_bounding_box(self):
        """A bee on the grass beside a round trap is not inside it."""
        refs = ops.roi_references({"regions": [{
            "box": [0.0, 0.0, 1.0, 1.0],
            "points": [[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]],  # diamond
        }]})

        self.assertIsNotNone(ops.which_reference(0.5, 0.5, refs))   # centre
        self.assertIsNone(ops.which_reference(0.02, 0.02, refs))    # corner


class PerReferenceInteractionTests(unittest.TestCase):
    """Time in each reference, from the one episode pass."""

    def setUp(self):
        self.refs = ops.roi_references(ROI)

    def summary(self, df, **kw):
        return primitives.summarize_interactions(primitives.interactions_from_episodes(
            ops.compute_episodes(df, self.refs, **kw), fps=10))

    def test_visits_are_attributed_to_the_reference_they_happened_in(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)],
                    *[(2, f, 0.9, 0.9) for f in range(10)])

        by_id = {r["id"]: r for r in self.summary(df)["per_reference"]}

        self.assertEqual(by_id["reference_3"]["interactions"], 1)
        self.assertEqual(by_id["reference_7"]["interactions"], 1)
        self.assertEqual(by_id["reference_3"]["partners"], 1)

    def test_moving_between_references_is_two_visits_not_one(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(5)],
                    *[(1, f, 0.9, 0.9) for f in range(5, 10)])

        self.assertEqual(self.summary(df)["organism_reference"], 2)

    def test_a_long_gap_in_the_same_reference_is_a_second_visit(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(5)],
                    *[(1, f, 0.1, 0.1) for f in range(100, 105)])

        self.assertEqual(self.summary(df, gap_frames=15)["organism_reference"], 2)

    def test_the_busiest_reference_comes_first(self):
        df = tracks(*[(1, f, 0.9, 0.9) for f in range(10)],
                    *[(2, f, 0.9, 0.9) for f in range(20, 30)],
                    *[(3, f, 0.1, 0.1) for f in range(10)])

        self.assertEqual(self.summary(df)["per_reference"][0]["id"], "reference_7")

    def test_time_is_split_between_references(self):
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)],
                    *[(1, f, 0.9, 0.9) for f in range(10, 40)])

        by_id = {r["id"]: r for r in self.summary(df)["per_reference"]}

        self.assertAlmostEqual(by_id["reference_3"]["duration_sec"], 1.0, places=2)
        self.assertAlmostEqual(by_id["reference_7"]["duration_sec"], 3.0, places=2)

    def test_points_outside_every_reference_count_for_nothing(self):
        df = tracks(*[(1, f, 0.5, 0.5) for f in range(20)])

        self.assertEqual(self.summary(df)["organism_reference"], 0)

    def test_bare_boxes_from_an_older_pipeline_still_work(self):
        """They just get numbers for names rather than refusing to compute."""
        df = tracks(*[(1, f, 0.1, 0.1) for f in range(10)])

        episodes = ops.compute_episodes(df, ops.roi_shapes(ROI))

        self.assertEqual([e["reference"] for e in episodes], ["reference_1"])

    def test_insect_to_insect_has_no_reference_and_is_not_counted_in_one(self):
        rows = [{"track_id": 1, "with": 2, "relation": "near", "duration_sec": 3.0,
                 "start_frame": 10},
                {"track_id": 1, "with": "reference_3", "relation": "in_reference",
                 "duration_sec": 1.0, "start_frame": 20}]

        out = primitives.summarize_interactions(rows)

        self.assertEqual([r["id"] for r in out["per_reference"]], ["reference_3"])
        self.assertEqual(out["organism_organism"], 1)


class AnalyzerResultsTests(unittest.TestCase):
    """The batch page renders what the pipeline computed.

    It used to read events CSVs and render foraging trips whatever ran, so a
    Visitation pipeline got a page about a question it never asked.
    """

    class _Run:
        def __init__(self, context):
            self.context = context

    def test_an_analyzer_is_found_by_its_table_kind(self):
        from apps.pipelines import aggregate

        run = self._Run({"a": {"table_kind": "visitation", "total_visits": 3}})

        self.assertEqual(aggregate.analyzer_outputs(run)[0][0], "visitation")

    def test_an_old_foraging_trips_output_is_not_an_analyzer_any_more(self):
        """Trips are left to the reader (memory/60); an old run's trips output
        has no table_kind and no panel."""
        from apps.pipelines import aggregate

        run = self._Run({"a": {"artifact": "events", "trips": []}})

        self.assertEqual(aggregate.analyzer_outputs(run), [])

    def test_non_dict_context_entries_are_ignored(self):
        from apps.pipelines import aggregate

        run = self._Run({"a": True, "b": None, "c": {"table_kind": "visitation"}})

        self.assertEqual(len(aggregate.analyzer_outputs(run)), 1)

    def test_episodes_are_summed_per_reference_across_clips(self):
        """A visit is an insect-to-reference interaction, so this is the
        visitation roll-up — read off the primitive rather than off a second
        analyzer that could disagree with it."""
        from apps.pipelines import aggregate

        runs = [
            self._Run({"a": {"table_kind": "interactions", "interaction_count": 3,
                             "organism_reference": 3, "total_duration_sec": 10.0,
                             "per_reference": [
                                 {"id": "nest_3", "label": "Nest 3", "interactions": 2, "partners": 2, "duration_sec": 7.0},
                                 {"id": "nest_7", "label": "Nest 7", "interactions": 1, "partners": 1, "duration_sec": 3.0}]}}),
            self._Run({"a": {"table_kind": "interactions", "interaction_count": 2,
                             "organism_reference": 2, "total_duration_sec": 5.0,
                             "per_reference": [
                                 {"id": "nest_3", "label": "Nest 3", "interactions": 2, "partners": 1, "duration_sec": 5.0}]}}),
        ]

        results = aggregate.analyzer_results(runs)
        summary = results[0]["summary"]

        by_id = {r["id"]: r for r in summary["per_reference"]}
        self.assertEqual(summary["organism_reference"], 5)
        self.assertEqual(by_id["reference_3"]["interactions"], 4)
        self.assertEqual(by_id["reference_3"]["clips"], 2)
        self.assertEqual(by_id["reference_7"]["interactions"], 1)

    def test_the_busiest_reference_leads(self):
        from apps.pipelines import aggregate

        run = self._Run({"a": {"table_kind": "interactions", "per_reference": [
            {"id": "nest_1", "label": "Nest 1", "interactions": 1},
            {"id": "nest_9", "label": "Nest 9", "interactions": 8}]}})

        summary = aggregate.analyzer_results([run])[0]["summary"]

        self.assertEqual(summary["per_reference"][0]["id"], "reference_9")

    def test_the_retired_visitation_kind_no_longer_renders_a_panel(self):
        """A visit IS an insect-to-reference interaction. Its own panel was a
        second, worse answer to a question Interactions already answers — and
        on a pipeline with no references it showed four zeros and an apology."""
        from apps.pipelines import aggregate

        run = self._Run({"a": {"table_kind": "visitation", "total_visits": 3}})

        result = aggregate.analyzer_results([run])[0]

        self.assertEqual(result["kind"], "visitation")
        self.assertIsNone(result["summary"])

    def test_interactions_aggregate_per_reference_too(self):
        from apps.pipelines import aggregate

        runs = [self._Run({"a": {"table_kind": "interaction", "interaction_count": 2,
                                 "organism_reference": 2, "total_duration_sec": 4.0,
                                 "per_reference": [{"id": "nest_3", "label": "Nest 3",
                                                    "interactions": 2, "partners": 2,
                                                    "duration_sec": 4.0}]}})] * 2

        summary = aggregate.analyzer_results(runs)[0]["summary"]

        self.assertEqual(summary["interaction_count"], 4)
        self.assertEqual(summary["per_reference"][0]["interactions"], 4)

    def test_colony_activity_gets_no_section(self):
        """Its computation stays; a timeline belongs inside whichever analyzer
        ran rather than on a page of its own."""
        from apps.pipelines import aggregate

        run = self._Run({"a": {"table_kind": "colony_activity", "peak": 4}})

        results = aggregate.analyzer_results([run])

        self.assertEqual(results[0]["kind"], "colony_activity")
        self.assertIsNone(results[0]["summary"])

    def test_the_analyzer_that_ran_on_most_clips_leads(self):
        from apps.pipelines import aggregate

        runs = [self._Run({"a": {"table_kind": "visitation", "per_reference": []},
                           "b": {"table_kind": "interaction", "per_reference": []}}),
                self._Run({"a": {"table_kind": "visitation", "per_reference": []}})]

        results = aggregate.analyzer_results(runs)

        self.assertEqual(results[0]["kind"], "visitation")
        self.assertEqual(results[0]["clips"], 2)

    def test_a_batch_with_no_analyzer_output_yields_nothing(self):
        from apps.pipelines import aggregate

        self.assertEqual(aggregate.analyzer_results([self._Run({})]), [])
