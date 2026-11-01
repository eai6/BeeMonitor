"""A batch's clips by site and review, metrics for the clips shown, and
taking clips out of a batch (memory/73)."""

import uuid

from django.urls import reverse

from apps.pipelines.models import ClipReview, EventReview, PipelineRun
from apps.pipelines.tests.test_insect_groups import FilterBase
from apps.videos.models import Video


class BatchSelectTests(FilterBase):
    def setUp(self):
        super().setUp()
        self.video.site_name = "mendels"
        self.video.save()
        self.other = Video.objects.create(
            user=self.user, device=self.device, title="clip2", site_name="natalies",
            storage_key="prim/clip2.mp4", file_size_bytes=1, status=Video.Status.READY)
        self.batch = uuid.uuid4()
        self.run_a, _ = self._run(self._steps_with_species(in_line=False))
        mine = self.video
        self.video = self.other
        self.run_b, _ = self._run(self._steps_with_species(in_line=False))
        self.video = mine
        PipelineRun.objects.filter(pk__in=[self.run_a.pk, self.run_b.pk]).update(batch_id=self.batch)
        self.client.force_login(self.user)

    def page(self):
        return self.client.get(reverse("pipelines:batch_detail", args=[self.batch]))

    def test_rows_carry_their_site_and_review_and_the_menus_show(self):
        ClipReview.objects.create(run=self.run_a, user=self.user)
        page = self.page()
        self.assertEqual(page.context["batch_sites"], ["mendels", "natalies"])
        self.assertContains(page, 'data-filter="site"')
        self.assertContains(page, 'data-filter="reviewed"')
        self.assertContains(page, f'data-f-run="{self.run_a.pk}" data-f-site="mendels" data-f-reviewed="yes"')
        self.assertContains(page, f'data-f-run="{self.run_b.pk}" data-f-site="natalies" data-f-reviewed="no"')
        self.assertContains(page, "data-select-toggle")

    def test_review_metrics_for_the_clips_shown(self):
        event = self.run_a.context["a"]["rows"][0]
        from apps.pipelines.review import event_key
        EventReview.objects.create(run=self.run_a, event_key=event_key(event), action="enter",
                                   reference="x", track_id="1", frame=1,
                                   user=self.user, verdict="accepted")
        url = reverse("pipelines:batch_review_metrics", args=[self.batch])
        shown = self.client.post(url, {"run": [self.run_a.pk]})
        self.assertContains(shown, "For the 1 clip the filter shows")
        self.assertContains(shown, "1 accepted")
        other = self.client.post(url, {"run": [self.run_b.pk]})
        self.assertContains(other, "Nothing reviewed among the 1 clip the filter shows")

    def test_metrics_need_access_to_the_batch(self):
        from django.contrib.auth.models import User
        self.client.force_login(User.objects.create_user("nosy", password="x"))
        res = self.client.post(reverse("pipelines:batch_review_metrics", args=[self.batch]),
                               {"run": [self.run_a.pk]})
        self.assertEqual(res.status_code, 404)

    def test_taking_a_clip_out_and_putting_it_back(self):
        ClipReview.objects.create(run=self.run_b, user=self.user)
        res = self.client.post(reverse("pipelines:batch_remove_clips", args=[self.batch]),
                               {"run": [self.run_b.pk]})
        self.assertEqual(res.status_code, 302)
        self.run_b.refresh_from_db()
        self.assertIsNone(self.run_b.batch_id)
        self.assertEqual(self.run_b.removed_from_batch, self.batch)
        self.assertEqual([r["run"].pk for r in self.page().context["rows"]], [self.run_a.pk])
        # Not listed as a run of its own either.
        runs = self.client.get(reverse("pipelines:run_list"))
        self.assertNotIn(self.run_b.pk, [r.pk for r in runs.context["solo_page"]])

        # Add clips puts the run back: its results and review, no new run.
        before = PipelineRun.objects.count()
        res = self.client.post(reverse("pipelines:run_on_videos"),
                               {"add_to": str(self.batch), "video_ids": [self.other.pk]})
        self.assertEqual(res.status_code, 302)
        self.assertEqual(PipelineRun.objects.count(), before)
        self.run_b.refresh_from_db()
        self.assertEqual((self.run_b.batch_id, self.run_b.removed_from_batch), (self.batch, None))
        self.assertTrue(ClipReview.objects.filter(run=self.run_b).exists())

    def test_a_running_clip_stays_in(self):
        PipelineRun.objects.filter(pk=self.run_b.pk).update(status=PipelineRun.Status.RUNNING)
        self.client.post(reverse("pipelines:batch_remove_clips", args=[self.batch]),
                         {"run": [self.run_b.pk]})
        self.run_b.refresh_from_db()
        self.assertEqual(self.run_b.batch_id, self.batch)

    def test_only_the_launcher_takes_clips_out(self):
        from django.contrib.auth.models import User
        self.client.force_login(User.objects.create_user("other", password="x"))
        res = self.client.post(reverse("pipelines:batch_remove_clips", args=[self.batch]),
                               {"run": [self.run_b.pk]})
        self.assertEqual(res.status_code, 404)

    def test_update_these_updates_the_selected_clips(self):
        res = self.client.post(reverse("pipelines:batch_apply_groups", args=[self.batch]),
                               {"which": "picked", "run": [self.run_a.pk]})
        self.assertEqual(res.status_code, 302)
