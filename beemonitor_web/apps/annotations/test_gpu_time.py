"""GPU time for sampling and pre-labeling: measured, charged, shown (memory/58).

A sampling batch is billed by its handler time, split across the clips it
finished; pre-labeling is billed at the machine it ran on, per frame checked.
Both charges carry their project, so a project can show what it used."""

import json
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.accounts import quota
from apps.accounts.models import UsageEvent
from apps.annotations import sampling_remote as sr
from apps.annotations.models import (AnnotationProject, FrameSamplingTask, PreAnnotationTask,
                                     SamplingBatch)
from apps.annotations.test_sampling_remote import GPU, FakeAWS
from apps.videos.models import Video

User = get_user_model()


def clip(task, download, scan, upload, **extra):
    return {"task_id": task.pk, "motion": {"profile": [0], "picked": [], "frames": 10},
            "frames": [], "total_frames": 1800, "candidates": 30,
            "seconds": {"download": download, "scan_and_detect": scan, "upload": upload}, **extra}


@override_settings(**GPU)
class SamplingChargeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("g", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])
        self.videos = []
        for i in range(3):
            v = Video.objects.create(user=self.user, title=f"c{i}", storage_key=f"u/c{i}.mp4",
                                     file_size_bytes=10_000_000, status=Video.Status.READY)
            self.project.videos.add(v)
            self.videos.append(v)
        self.aws = FakeAWS()
        patch = mock.patch("boto3.client", side_effect=self.aws.client)
        patch.start()
        self.addCleanup(patch.stop)

    def batch_of(self, n):
        params = sr.params_for(["bee"])
        tasks = [FrameSamplingTask.objects.create(user=self.user, project=self.project, video=v,
                                                  params=params) for v in self.videos[:n]]
        with self.settings(SAMPLING_BATCH_CLIPS=n):
            sr.dispatch()
        return tasks, SamplingBatch.objects.get()

    def finish(self, batch, clips, execution_seconds):
        self.aws.objects[("out", batch.result_key)] = json.dumps(
            {"clips": clips, "execution_seconds": execution_seconds}).encode()

    def test_batch_time_is_split_and_only_finished_clips_are_charged(self):
        (a, b, c), batch = self.batch_of(3)
        # Three clips side by side: their wall times (20 + 10 + 10) overlap; the
        # handler held the GPU for 24 s. b was cancelled meanwhile, c failed.
        FrameSamplingTask.objects.filter(pk=b.pk).update(status=FrameSamplingTask.Status.CANCELLED)
        self.finish(batch, [clip(a, 2, 16, 2), clip(b, 1, 8, 1),
                            {"task_id": c.pk, "error": "RuntimeError: decode", "frames": [],
                             "seconds": {"download": 10}}], 24.0)
        self.assertEqual(sr.collect(), 1)

        batch.refresh_from_db()
        self.assertEqual((batch.gpu_seconds, batch.clips_completed), (24.0, 1))
        a.refresh_from_db()
        self.assertAlmostEqual(a.gpu_seconds, 12.0)          # 24 × 20/40
        self.assertEqual(a.timing, {"download": 2, "scan_and_detect": 16, "upload": 2,
                                    "frames_decoded": 1800, "candidates": 30})
        for t in (b, c):
            t.refresh_from_db()
            self.assertEqual(t.gpu_seconds, 0)
        ev = UsageEvent.objects.get()
        self.assertAlmostEqual(ev.gpu_seconds, 12.0)
        self.assertEqual((ev.kind, ev.instance_type, ev.units, ev.project_id),
                         ("sampling", quota.SAM3_INSTANCE, 1, self.project.pk))

        self.assertEqual(sr.collect(), 0)                    # collected once, charged once
        self.assertEqual(UsageEvent.objects.count(), 1)

    def test_charged_time_is_the_batch_not_the_sum_of_overlapping_clips(self):
        tasks, batch = self.batch_of(3)
        self.finish(batch, [clip(t, 1, 18, 1) for t in tasks], 21.0)
        sr.collect()
        ev = UsageEvent.objects.get()
        self.assertAlmostEqual(ev.gpu_seconds, 21.0)          # not 3 × 18
        self.assertEqual(ev.units, 3)

    def test_a_result_without_handler_time_bills_the_scan_time(self):
        tasks, batch = self.batch_of(2)
        self.aws.objects[("out", batch.result_key)] = json.dumps(
            {"clips": [clip(t, 1, 5, 1) for t in tasks]}).encode()
        sr.collect()
        self.assertAlmostEqual(UsageEvent.objects.get().gpu_seconds, 10.0)

    def test_a_failed_batch_is_recorded_and_free(self):
        tasks, batch = self.batch_of(2)
        self.aws.objects[("out", batch.failure_uri.split("s3://out/", 1)[1])] = b"ModelError"
        sr.collect()
        batch.refresh_from_db()
        self.assertEqual(batch.status, "failed")
        self.assertIn("ModelError", batch.error)
        self.assertFalse(UsageEvent.objects.exists())


@override_settings(**GPU)
class PreLabelChargeTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("p", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["insect"])
        self.video = Video.objects.create(user=self.user, title="v", storage_key="u/v.mp4",
                                          file_size_bytes=10_000_000, status=Video.Status.READY)
        self.aws = FakeAWS()

    def finalize(self, labeler, result, **task_kw):
        from apps.annotations.views import finalize_preannotation_task
        task = PreAnnotationTask.objects.create(
            user=self.user, project=self.project, video=self.video, labeler=labeler,
            status=PreAnnotationTask.Status.PROCESSING, output_uri="s3://out/r.out", **task_kw)
        self.aws.objects[("out", "r.out")] = json.dumps(result).encode()
        self.assertTrue(finalize_preannotation_task(task, self.aws))
        task.refresh_from_db()
        return task, UsageEvent.objects.get()

    def test_yolo_on_the_sam3_endpoint_is_charged_as_a_g5_per_frame_checked(self):
        self.assertEqual(quota.auto_label_work("yolo", 5).instance, quota.SAM3_INSTANCE)
        task, ev = self.finalize("yolo", {"frames": [{"frame_number": 3, "boxes": []}],
                                          "total_frames_checked": 40, "execution_seconds": 90},
                                 params={"max_frames": 300})
        self.assertEqual((task.gpu_seconds, task.instance_type), (90.0, quota.SAM3_INSTANCE))
        self.assertEqual((ev.instance_type, ev.units, ev.project_id, ev.credits),
                         (quota.SAM3_INSTANCE, 40, self.project.pk, quota.credits_for(90, quota.SAM3_INSTANCE)))

    def test_the_machine_recorded_at_send_is_what_is_charged(self):
        _, ev = self.finalize("yolo", {"frames": [], "execution_seconds": 60},
                              instance_type=quota.VIDEO_INSTANCE)
        self.assertEqual(ev.instance_type, quota.VIDEO_INSTANCE)

    @override_settings(SAGEMAKER_SAM3_ENDPOINT_NAME="")
    def test_yolo_without_a_sam3_endpoint_runs_and_is_estimated_on_the_g4dn(self):
        self.assertEqual(quota.auto_label_work("yolo", 5).instance, quota.VIDEO_INSTANCE)

    def test_editor_suggestions_are_charged_too(self):
        task, ev = self.finalize("sam3", {"frames": [{"frame_number": 1, "boxes": []}],
                                          "execution_seconds": 12},
                                 params={"suggest_only": True})
        self.assertEqual((task.status, ev.gpu_seconds, ev.units), ("completed", 12.0, 1))


class ShownTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("s", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])
        other = User.objects.create_user("o", password="x")
        self.elsewhere = AnnotationProject.objects.create(user=other, name="Q", classes=["bee"])
        quota.charge(self.user, "sampling", 3 * 3600, quota.SAM3_INSTANCE, "t", "1",
                     project=self.project)
        quota.charge(self.user, "auto_label", 1800, quota.SAM3_INSTANCE, "t", "2",
                     project=self.project.pk)
        quota.charge(other, "sampling", 7200, quota.SAM3_INSTANCE, "t", "3", project=self.elsewhere)
        quota.charge(self.user, "analysis", 3600, quota.VIDEO_INSTANCE, "t", "4")

    def test_project_total_is_only_that_projects_work(self):
        u = quota.project_usage(self.project)
        self.assertAlmostEqual(u["hours"], 3.5)
        self.assertEqual(u["credits"], Decimal("7.0000"))
        self.assertEqual([(k["kind"], k["hours"]) for k in u["kinds"]],
                         [("sampling", 3.0), ("auto_label", 0.5)])

    def test_project_page_shows_it_to_managers(self):
        self.client.force_login(self.user)
        html = self.client.get(reverse("annotations:detail", args=[self.project.pk])).content.decode()
        self.assertIn("GPU used", html)
        self.assertIn("3.5 GPU-h", html)
        self.assertIn("pre-labeling 0.5 h", html)

    def test_usage_page_splits_the_month_by_kind(self):
        self.client.force_login(self.user)
        html = self.client.get(reverse("accounts:usage")).content.decode()
        self.assertIn("This month by kind", html)
        self.assertIn("Sampling <strong class=\"text-gray-900\">3.0 GPU-h</strong>", html)
        self.assertIn("Pre-labeling", html)
        self.assertIn("Video analysis", html)
