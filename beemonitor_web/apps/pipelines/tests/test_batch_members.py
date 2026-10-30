"""Inviting people to review or view a batch (memory/62)."""

from unittest import mock

from django.urls import reverse

from apps.pipelines.models import BatchMember, EventReview
from apps.pipelines.tests.test_event_review import EventReviewTests


class BatchMemberTests(EventReviewTests):
    def invite(self, who, role="reviewer", as_user=None):
        self.client.force_login(as_user or self.owner)
        return self.client.post(reverse("pipelines:batch_members", args=[self.batch]),
                                {"who": who, "role": role})

    def test_the_owner_invites_by_username_or_email(self):
        self.maria.email = "maria@uni.edu"
        self.maria.save()
        self.invite("maria@uni.edu")
        self.assertEqual(BatchMember.objects.get().user, self.maria)
        self.invite("stranger", role="viewer")
        self.assertEqual(BatchMember.objects.get(user=self.stranger).role, "viewer")

    def test_only_the_owner_can_invite(self):
        r = self.invite("stranger", as_user=self.maria)
        self.assertEqual(r.status_code, 404)
        self.assertFalse(BatchMember.objects.exists())

    def test_an_unknown_name_is_reported_not_invited(self):
        self.invite("nobody-here")
        self.assertFalse(BatchMember.objects.exists())

    def test_a_member_opens_the_batch_and_its_clips_read_only(self):
        self.invite("maria")
        self.client.force_login(self.maria)
        page = self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))
        self.assertEqual(page.status_code, 200)
        self.assertFalse(page.context["can_share"])
        self.assertFalse(page.context["can_rerun"])
        self.assertEqual(page.context["my_role"], "reviewer")
        with mock.patch("apps.analysis.views._generate_presigned_url", return_value="https://x"):
            clip = self.client.get(reverse("analysis:results", args=[self.job.pk]))
        self.assertEqual(clip.status_code, 200)
        self.assertTrue(clip.context["can_review"])
        self.assertEqual(self.client.get(reverse("analysis:results_csv", kwargs={
            "pk": self.job.pk, "kind": "events"})).status_code, 200)

    def test_a_viewer_watches_but_cannot_vote(self):
        self.invite("maria", role="viewer")
        with mock.patch("apps.analysis.views._generate_presigned_url", return_value="https://x"):
            self.client.force_login(self.maria)
            clip = self.client.get(reverse("analysis:results", args=[self.job.pk]))
        self.assertEqual(clip.status_code, 200)
        self.assertFalse(clip.context["can_review"])
        self.assertEqual(self.vote(self.maria, "accepted").status_code, 404)
        self.assertFalse(EventReview.objects.exists())

    def test_a_stranger_sees_nothing(self):
        self.client.force_login(self.stranger)
        self.assertEqual(self.client.get(reverse("pipelines:batch_detail", args=[self.batch])).status_code, 404)
        self.assertEqual(self.client.get(reverse("analysis:results", args=[self.job.pk])).status_code, 404)

    def test_removing_a_member_closes_it_again(self):
        self.invite("maria")
        member = BatchMember.objects.get()
        self.client.force_login(self.owner)
        self.client.post(reverse("pipelines:batch_members", args=[self.batch]),
                         {"action": "remove", "member": member.pk})
        self.client.force_login(self.maria)
        self.assertEqual(self.client.get(reverse("analysis:results", args=[self.job.pk])).status_code, 404)

    def test_shared_batches_are_listed_under_runs(self):
        self.invite("maria")
        self.client.force_login(self.maria)
        page = self.client.get(reverse("pipelines:run_list"))
        self.assertEqual([str(b["batch_id"]) for b in page.context["shared"]], [str(self.batch)])
