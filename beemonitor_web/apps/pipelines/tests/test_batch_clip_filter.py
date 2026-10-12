"""Filtering a batch's clips by name, unit, day, status and what was found.

The filtering itself runs in the page (static/js/clip_filter.js) over the rows
already rendered; these check that each row carries what it is filtered on and
that the share page's rows carry no more than the share page shows.
"""
from datetime import datetime, timezone as dt_timezone

from django.test import TestCase
from django.urls import reverse

from apps.pipelines import aggregate, sharing
from apps.pipelines.tests import test_batch_cancel as cancel_fixtures
from apps.pipelines.tests import test_batch_share as share_fixtures


class BatchPageFilterTests(TestCase):
    # One finished clip and two unfinished ones.
    setUp = cancel_fixtures.BatchCancelTests.setUp
    _run = cancel_fixtures.BatchCancelTests._run

    def _page(self):
        url = reverse("pipelines:batch_detail", args=[cancel_fixtures.BATCH])
        return self.client.get(url).content.decode()

    def test_the_filter_bar_and_its_script_are_on_the_page(self):
        page = self._page()

        self.assertIn('id="clip-filter"', page)
        self.assertIn('data-filter="q"', page)
        self.assertIn("js/clip_filter.js", page)

    def test_rows_carry_what_they_are_filtered_on(self):
        page = self._page()

        self.assertEqual(page.count('data-f-status="completed"'), 1)
        self.assertEqual(page.count('data-f-status="running"'), 2)
        self.assertIn(f'data-f-video="{self.done.context["v"]["video_id"]}"', page)
        self.assertIn("data-f-day=", page)

    def test_the_status_chips_filter(self):
        page = self._page()

        self.assertIn('data-filter-status="completed"', page)
        self.assertIn('data-filter-status="running"', page)

    def test_the_clip_name_is_shown_so_a_name_search_makes_sense(self):
        page = self._page()

        self.assertIn("c0", page)

    def test_the_rerun_form_can_follow_the_filter(self):
        page = self._page()

        self.assertIn("data-rerun-form", page)
        self.assertIn("data-rerun-ids", page)


class SharePageFilterTests(TestCase):
    setUp = share_fixtures.BatchShareTests.setUp
    _clip = share_fixtures.BatchShareTests._clip

    def _page(self):
        share = sharing.turn_on(self.owner, share_fixtures.BATCH)
        return self.client.get(reverse("public_batch", args=[share.token])).content.decode()

    def test_collaborators_get_the_filter(self):
        page = self._page()

        self.assertIn('id="clip-filter"', page)
        self.assertIn('data-filter-status="completed"', page)
        self.assertIn('data-f-status="failed"', page)

    def test_share_rows_carry_nothing_the_share_page_hides(self):
        page = self._page()

        self.assertNotIn("data-f-video", page)
        self.assertNotIn("secret_path", page)
        self.assertNotIn("Clip name", page)
        self.assertNotIn("data-rerun", page)


class BatchDaysTests(TestCase):
    def test_distinct_days_newest_first(self):
        rows = [{"when": datetime(2026, 10, 7, 9, tzinfo=dt_timezone.utc)},
                {"when": datetime(2026, 10, 8, 9, tzinfo=dt_timezone.utc)},
                {"when": datetime(2026, 10, 8, 11, tzinfo=dt_timezone.utc)},
                {"when": None}]

        days = aggregate.batch_days(rows)

        self.assertEqual([d.isoformat() for d in days], ["2026-10-08", "2026-10-07"])
