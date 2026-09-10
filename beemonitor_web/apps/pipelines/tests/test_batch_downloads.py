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
        sources = [source(tracking_csv_path="t.csv", events_csv_path="e.csv")]

        kinds = [d["kind"] for d in aggregate.available_downloads(sources)]

        self.assertEqual(kinds, ["tracking", "events"])

    def test_a_batch_that_wrote_everything_offers_all_four(self):
        sources = [source(detections_csv_path="d.csv", tracking_csv_path="t.csv",
                          events_csv_path="e.csv", interactions_csv_path="i.csv")]

        self.assertEqual(len(aggregate.available_downloads(sources)), 4)

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
        self.assertEqual(by_kind["events"]["clips"], 1)

    def test_every_offered_kind_is_one_the_download_view_accepts(self):
        """The template builds URLs from these, so a typo would 404 in prod."""
        from apps.pipelines import views  # noqa: F401  (import check)

        accepted = {"events", "tracking", "interactions", "detections"}
        self.assertEqual({k for k, _p, _l, _h in aggregate.BASE_TABLES}, accepted)

    def test_the_tables_read_in_pipeline_order(self):
        order = [k for k, _p, _l, _h in aggregate.BASE_TABLES]

        self.assertEqual(order, ["detections", "tracking", "events", "interactions"])
