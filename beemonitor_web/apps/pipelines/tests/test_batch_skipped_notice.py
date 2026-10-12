"""The clips left out of the downloads are counted by reason, not listed.

A big batch printed every left-out clip's id above the clip list: hundreds of
hex strings, a page of noise before the first row.
"""
from django.test import TestCase
from django.urls import reverse

from apps.pipelines.tests import test_batch_cancel as fixtures

BATCH = fixtures.BATCH


class SkippedNoticeTests(TestCase):
    # One finished clip and two unfinished ones, as in the cancel tests.
    setUp = fixtures.BatchCancelTests.setUp
    _run = fixtures.BatchCancelTests._run

    def test_left_out_clips_are_counted_by_reason(self):
        page = self.client.get(reverse("pipelines:batch_detail", args=[BATCH])).content.decode()

        self.assertIn("not in the downloads above", page)
        self.assertIn("2 run running", page)
        self.assertNotIn("(run running)", page)
