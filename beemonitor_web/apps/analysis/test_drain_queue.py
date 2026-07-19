"""Tests for the QUEUED-job drain (global SageMaker concurrency gate)."""

from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase, override_settings

from apps.analysis import views
from apps.analysis.models import Job
from apps.videos.models import Video


@override_settings(SAGEMAKER_MAX_CONCURRENT=2)
class DrainQueueTest(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="drainer")
        self.video = Video.objects.create(user=self.user, title="v",
                                          storage_key="k", file_size_bytes=1)

    def _queue(self, n):
        base = Job.objects.count()
        return [Job.objects.create(user=self.user, video=self.video,
                                   status=Job.Status.QUEUED,
                                   modal_job_id=f"q{base + i}") for i in range(n)]

    def test_drain_respects_cap_and_spawns_that_many(self):
        self._queue(5)
        with mock.patch.object(views, "spawn_gpu_job_async") as spawn:
            spawned = views._drain_queue()
        self.assertEqual(spawned, 2)                       # cap=2, 0 active → 2 slots
        self.assertEqual(spawn.call_count, 2)
        self.assertEqual(Job.objects.filter(status=Job.Status.PROCESSING).count(), 2)
        self.assertEqual(Job.objects.filter(status=Job.Status.QUEUED).count(), 3)

    def test_no_slots_when_at_capacity(self):
        Job.objects.create(user=self.user, video=self.video,
                            status=Job.Status.PROCESSING, modal_job_id="p1")
        Job.objects.create(user=self.user, video=self.video,
                            status=Job.Status.PROCESSING, modal_job_id="p2")
        self._queue(3)
        with mock.patch.object(views, "spawn_gpu_job_async") as spawn:
            spawned = views._drain_queue()
        self.assertEqual(spawned, 0)
        spawn.assert_not_called()
        self.assertEqual(Job.objects.filter(status=Job.Status.QUEUED).count(), 3)

    def test_fifo_oldest_first(self):
        jobs = self._queue(3)
        with mock.patch.object(views, "spawn_gpu_job_async"):
            views._drain_queue()
        jobs = [Job.objects.get(pk=j.pk) for j in jobs]
        # The two oldest (created first) are promoted; the newest stays QUEUED.
        self.assertEqual(jobs[0].status, Job.Status.PROCESSING)
        self.assertEqual(jobs[1].status, Job.Status.PROCESSING)
        self.assertEqual(jobs[2].status, Job.Status.QUEUED)

    def test_claim_is_idempotent(self):
        """A job already promoted is never spawned twice across drains."""
        self._queue(2)
        with mock.patch.object(views, "spawn_gpu_job_async") as spawn:
            first = views._drain_queue()   # promotes 2 (fills cap)
            second = views._drain_queue()  # no free slots now
        self.assertEqual(first, 2)
        self.assertEqual(second, 0)
        self.assertEqual(spawn.call_count, 2)  # never re-spawned
