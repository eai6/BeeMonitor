"""Cancelling a batch stops its unfinished clips on the GPU too (memory/49).

SageMaker can't abort an async request, so a cancel used to be bookkeeping:
the clips still ran to the end and billed. Now each cancelled job leaves a
marker the worker watches for.
"""
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis.cancelling import cancel_jobs
from apps.analysis.models import Job
from apps.pipelines import aggregate
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()
BATCH = "c4ce11ed-0000-4000-8000-0000000000cc"
STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "t", "block_type": "track.mot", "config": {}},
    {"id": "i", "block_type": "analyze.interactions", "config": {}},
]


class BatchCancelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("cx", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="Pollen")
        self.done = self._run("completed", {"v": "done", "t": "done", "i": "done"}, None)
        self.running, self.job = self._run("running", {"v": "done", "t": "running"}, "processing",
                                           with_job=True)
        self.queued, self.queued_job = self._run("running", {"v": "done", "t": "running"}, "queued",
                                                 with_job=True)

    def _run(self, status, step_status, job_status, with_job=False):
        n = PipelineRun.objects.count()
        video = Video.objects.create(
            user=self.user, title=f"c{n}", storage_key=f"cx/{n}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now(), duration_seconds=600)
        steps = [dict(s) for s in STEPS]
        steps[0]["config"] = {"video_id": str(video.pk)}
        context = {"v": {"artifact": "video", "video_id": video.pk}}
        job = None
        if with_job:
            job = Job.objects.create(user=self.user, video=video, status=job_status,
                                     modal_job_id=f"pl_job{n}")
            context["t"] = {"job_id": job.pk, "pending": True}
        run = PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH, status=status,
            steps=steps, step_status=step_status, context=context)
        return (run, job) if with_job else run

    def _cancel(self):
        s3 = mock.Mock()
        with mock.patch("config.storage.get_s3_client", return_value=s3):
            r = self.client.post(reverse("pipelines:batch_cancel", args=[BATCH]))
        return r, s3

    def test_unfinished_clips_are_cancelled_and_finished_ones_kept(self):
        r, _ = self._cancel()

        self.assertEqual(r.status_code, 302)
        for run in (self.running, self.queued):
            run.refresh_from_db()
            self.assertEqual(run.status, "failed")
            self.assertEqual(aggregate.run_error(run), "Cancelled by user.")
        self.done.refresh_from_db()
        self.assertEqual(self.done.status, "completed")
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, "cancelled")

    def test_each_cancelled_job_tells_the_gpu(self):
        _, s3 = self._cancel()

        keys = sorted(c.args[1] for c in s3.upload_stream.call_args_list)
        self.assertEqual(keys, sorted([f"cancel/{self.job.modal_job_id}",
                                       f"cancel/{self.queued_job.modal_job_id}"]))
        self.assertTrue(all(c.args[0] == "processed" for c in s3.upload_stream.call_args_list))

    def test_only_the_launcher_can_cancel(self):
        self.client.force_login(User.objects.create_user("nosy", password="x"))
        r, s3 = self._cancel()

        self.assertEqual(r.status_code, 404)
        self.running.refresh_from_db()
        self.assertEqual(self.running.status, "running")
        s3.upload_stream.assert_not_called()

    def test_cancelled_clips_are_counted_apart_from_failures_and_can_be_rerun(self):
        self._cancel()
        with mock.patch("config.storage.get_s3_client"):
            resp = self.client.get(reverse("pipelines:batch_detail", args=[BATCH]))

        outcome = resp.context["outcome"]
        self.assertEqual((outcome["cancelled"], outcome["failed"], outcome["completed"]), (2, 0, 1))
        html = resp.content.decode()
        self.assertIn("Cancelled by you", html)
        self.assertIn("Re-run these 2 clips", html)
        self.assertNotIn("Cancel batch", html)  # nothing left running

    def test_the_button_shows_while_clips_are_running(self):
        with mock.patch("config.storage.get_s3_client"):
            html = self.client.get(reverse("pipelines:batch_detail", args=[BATCH])).content.decode()

        self.assertIn("Cancel batch", html)
        self.assertIn("Cancel 2 clips", html)

    def test_a_marker_that_cannot_be_written_still_cancels_the_job(self):
        s3 = mock.Mock()
        s3.upload_stream.side_effect = RuntimeError("S3 down")
        with mock.patch("config.storage.get_s3_client", return_value=s3):
            self.assertEqual(cancel_jobs([self.job]), 1)
        self.job.refresh_from_db()
        self.assertEqual(self.job.status, "cancelled")

    def test_the_processing_page_cancel_tells_the_gpu_too(self):
        s3 = mock.Mock()
        with mock.patch("config.storage.get_s3_client", return_value=s3):
            self.client.post(reverse("analysis:cancel", args=[self.job.pk]))
        s3.upload_stream.assert_called_once()
        self.assertEqual(s3.upload_stream.call_args.args[1], f"cancel/{self.job.modal_job_id}")
