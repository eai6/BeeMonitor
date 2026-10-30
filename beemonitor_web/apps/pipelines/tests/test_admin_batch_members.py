"""A platform admin's own batch opens for the people they invite (memory/62).

Job.objects.for_viewer hides an admin's jobs from other users, so an admin
analysing someone else's clip stays out of that person's pages. It must not
hide the admin's own batch from its invited reviewers and viewers.
"""

import json
from unittest.mock import patch

from django.urls import reverse

from apps.pipelines.models import BatchMember
from django.test import TestCase

from apps.pipelines.tests import test_event_review as base


class AdminOwnedBatchTests(TestCase):
    # The review tests' clip, batch and helpers, without re-running their tests.
    vote = base.EventReviewTests.vote

    def setUp(self):
        base.EventReviewTests.setUp(self)
        for target in ("apps.analysis.views._generate_presigned_url", "config.storage.presigned_get"):
            p = patch(target, return_value="https://s3/x")
            p.start()
            self.addCleanup(p.stop)
        self.owner.is_superuser = True
        self.owner.save()
        BatchMember.objects.get_or_create(batch_id=self.batch, user=self.maria, defaults={"role": "reviewer"})

    def test_an_invited_reviewer_opens_and_reviews_the_admins_clip(self):
        self.client.force_login(self.maria)
        self.assertEqual(self.client.get(reverse("pipelines:batch_detail", args=[self.batch])).status_code, 200)
        page = self.client.get(reverse("analysis:results", args=[self.job.pk]))
        self.assertEqual(page.status_code, 200, page.content[:300])
        overlay = reverse("analysis:track_overlay", args=[self.job.pk])
        self.assertEqual(self.client.get(overlay, {"part": "events"}).status_code, 200)
        self.assertEqual(self.vote(self.maria, "accepted").status_code, 200)
        r = self.client.post(reverse("analysis:add_missed_event", args=[self.job.pk]),
                             json.dumps({"frame": 400, "action": "enter", "reference": "1"}),
                             content_type="application/json")
        self.assertEqual(r.status_code, 200, r.content)

    def test_a_stranger_still_cant(self):
        self.client.force_login(self.stranger)
        self.assertEqual(self.client.get(reverse("analysis:results", args=[self.job.pk])).status_code, 404)
