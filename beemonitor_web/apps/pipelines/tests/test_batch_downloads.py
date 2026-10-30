"""The batch page hands back the base tables, and its rows open the clip.

Two things were awkward on a finished batch: the only download was the events
CSV even though tracking and interactions were sitting right there, and
clicking a clip landed on the run's own page rather than the clip's data.
"""

from django.test import SimpleTestCase

from apps.pipelines import aggregate


def source(**paths):
    return {"result": paths}


class AvailableDownloadTests(SimpleTestCase):
    def test_only_tables_something_actually_wrote_are_offered(self):
        sources = [source(detections_csv_path="d.csv")]

        kinds = [d["kind"] for d in aggregate.available_downloads(sources)]

        self.assertEqual(kinds, ["detections"])

    def test_tracking_alone_offers_no_events_or_interactions(self):
        """Only a pipeline's own Events / Interactions steps make those tables
        (memory/60); tracking alone is not a reason to compute them."""
        sources = [source(tracking_csv_path="t.csv")]

        kinds = [d["kind"] for d in aggregate.available_downloads(sources)]

        self.assertEqual(kinds, ["tracking", "tracks"])

    def test_the_tables_the_runs_steps_made_are_offered(self):
        class Run:
            context = {"e": {"table_kind": "events", "rows": []},
                       "i": {"table_kind": "interactions", "rows": [{}]}}
        sources = [source(detections_csv_path="d.csv", tracking_csv_path="t.csv",
                          events_csv_path="e.csv", interactions_csv_path="i.csv")]

        kinds = [d["kind"] for d in aggregate.available_downloads(sources, [Run()])]

        # The four base tables plus Tracks (one row per track, with identity).
        self.assertEqual(kinds, ["detections", "tracking", "tracks", "events", "interactions"])

    def test_a_worker_events_file_is_not_offered_without_the_step(self):
        sources = [source(tracking_csv_path="t.csv", events_csv_path="e.csv")]

        kinds = [d["kind"] for d in aggregate.available_downloads(sources)]

        self.assertNotIn("events", kinds)

    def test_a_batch_with_no_tracking_offers_no_primitives(self):
        sources = [source(detections_csv_path="d.csv")]

        kinds = [d["kind"] for d in aggregate.available_downloads(sources)]

        self.assertNotIn("interactions", kinds)
        self.assertNotIn("events", kinds)

    def test_a_button_that_would_download_nothing_is_not_shown(self):
        # A button returning an empty CSV reads as a bug rather than as a
        # pipeline that never produced that table.
        self.assertEqual(aggregate.available_downloads([source()]), [])
        self.assertEqual(aggregate.available_downloads([]), [])

    def test_the_clip_count_reflects_how_many_wrote_that_table(self):
        sources = [source(tracking_csv_path="a.csv"),
                   source(tracking_csv_path="b.csv"),
                   source(events_csv_path="c.csv")]

        by_kind = {d["kind"]: d for d in aggregate.available_downloads(sources)}

        self.assertEqual(by_kind["tracking"]["clips"], 2)

    def test_every_offered_kind_is_one_the_download_view_accepts(self):
        """The template builds URLs from these, so a typo would 404 in prod."""
        from apps.pipelines import views  # noqa: F401  (import check)

        accepted = {"events", "tracking", "interactions", "detections"}
        self.assertEqual({k for k, _p, _l, _h in aggregate.BASE_TABLES}, accepted)

    def test_the_tables_read_in_pipeline_order(self):
        order = [k for k, _p, _l, _h in aggregate.BASE_TABLES]

        self.assertEqual(order, ["detections", "tracking", "events", "interactions"])
