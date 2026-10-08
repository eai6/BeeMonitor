"""Re-running a batch's failed clips in place.

Recovering one clip the GPU dropped used to mean a new batch over all 98: a
whole batch of GPU time, and results split across two batches.
"""
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()
BATCH = "91dc8dc3-0000-4000-8000-0000000000aa"
STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "t", "block_type": "track.mot", "config": {}},
    {"id": "i", "block_type": "analyze.interactions", "config": {}},
]


class RetryFailedTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("rf", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="Pollen")
        self.done = self._run("completed", {"v": "done", "t": "done", "i": "done"},
                              {"t": {"job_id": 1}, "i": {"table_kind": "interactions"}})
        self.failed = self._run(
            "failed", {"v": "done", "t": "failed", "i": "failed"},
            {"t": {"error": "Amazon SageMaker could not get a response"},
             "i": {"error": "Upstream step failed."}})

    def _run(self, status, step_status, context):
        video = Video.objects.create(
            user=self.user, title="c", storage_key="rf/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now(), duration_seconds=600)
        steps = [dict(s) for s in STEPS]
        steps[0]["config"] = {"video_id": str(video.pk)}
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH, status=status,
            steps=steps, step_status=step_status,
            context={"v": {"artifact": "video", "video_id": video.pk}, **context})

    def _retry(self, **data):
        with mock.patch("apps.pipelines.views.engine.advance_run") as advance:
            r = self.client.post(reverse("pipelines:batch_retry_failed", args=[BATCH]), data)
        return r, advance

    def test_only_the_failed_clip_is_reset_and_it_stays_in_the_batch(self):
        r, advance = self._retry()

        self.assertEqual(r.status_code, 302)
        self.failed.refresh_from_db()
        self.done.refresh_from_db()
        self.assertEqual(self.failed.status, "running")
        self.assertEqual(str(self.failed.batch_id), BATCH)
        self.assertEqual(self.failed.step_status, {"v": "done", "t": "pending", "i": "pending"})
        self.assertNotIn("t", self.failed.context)
        self.assertIn("v", self.failed.context)          # the clip itself is kept
        self.assertEqual(self.done.status, "completed")
        advance.assert_called_once_with(self.failed.pk)
        self.assertEqual(PipelineRun.objects.count(), 2)  # no new runs

    def test_a_cause_retries_only_its_own_clips(self):
        _, advance = self._retry(video_ids=["999999"])

        self.failed.refresh_from_db()
        self.assertEqual(self.failed.status, "failed")
        advance.assert_not_called()

    def test_someone_else_cannot_retry_your_batch(self):
        self.client.force_login(User.objects.create_user("nosy", password="x"))
        _, advance = self._retry()

        self.failed.refresh_from_db()
        self.assertEqual(self.failed.status, "failed")
        advance.assert_not_called()

    def test_the_failure_panel_offers_it(self):
        html = self.client.get(reverse("pipelines:batch_detail", args=[BATCH])).content.decode()

        self.assertIn(reverse("pipelines:batch_retry_failed", args=[BATCH]), html)
        self.assertIn("Re-run this clip", html)
