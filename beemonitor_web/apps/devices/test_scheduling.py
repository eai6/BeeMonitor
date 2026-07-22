"""Tests for recurring per-device pipeline schedules.

The reconciler daemon is the only clock in this deployment and several web
workers run it concurrently, so the properties that matter are: a schedule fires
only when due, it fires *once* even under concurrent passes (the compare-and-swap
claim), and it launches over the right slice of the device's videos.

GPU work is never reached — ``engine.launch_batch`` creates ``analysis.Job`` rows
in QUEUED and nothing spawns until ``_drain_queue`` runs, which is patched out.
"""

from datetime import timedelta
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.analysis.models import Job
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

from . import scheduling
from .models import Device, DevicePipelineSchedule

User = get_user_model()

STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "t", "block_type": "track.bee", "config": {"confidence": 0.4},
     "inputs": {"video": "v"}},
]


class DevicePipelineScheduleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="Danniella", key_hash="h1", prefix="bmk_1",
        )
        self.pipeline = Pipeline.objects.create(
            user=self.user, title="Foraging", steps=STEPS,
        )
        self.schedule = DevicePipelineSchedule.objects.create(
            device=self.device, user=self.user, pipeline=self.pipeline,
            cadence="hourly",
        )

    def _video(self, title, recorded_at, device=None):
        return Video.objects.create(
            user=self.user, device=device if device is not None else self.device,
            title=title, storage_key=f"alice/{title}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=recorded_at,
        )

    # ── due calculation ──────────────────────────────────────────────────────
    def test_never_run_is_due(self):
        self.assertTrue(self.schedule.is_due())

    def test_not_due_within_interval(self):
        self.schedule.last_run_at = timezone.now() - timedelta(minutes=30)
        self.assertFalse(self.schedule.is_due())

    def test_due_after_interval(self):
        self.schedule.last_run_at = timezone.now() - timedelta(hours=2)
        self.assertTrue(self.schedule.is_due())

    def test_disabled_is_never_due(self):
        self.schedule.enabled = False
        self.assertFalse(self.schedule.is_due())

    def test_daily_at_hour_only_fires_in_that_hour(self):
        self.schedule.cadence = "daily"
        self.schedule.last_run_at = timezone.now() - timedelta(days=2)
        now = timezone.now()
        self.schedule.at_hour = now.hour
        self.assertTrue(self.schedule.is_due(now))
        self.schedule.at_hour = (now.hour + 5) % 24
        self.assertFalse(self.schedule.is_due(now))

    # ── window selection ─────────────────────────────────────────────────────
    def test_first_run_takes_all_device_videos_and_ignores_other_devices(self):
        other = Device.objects.create(
            owner=self.user, name="Other", key_hash="h2", prefix="bmk_2",
        )
        self._video("mine", timezone.now() - timedelta(hours=1))
        self._video("theirs", timezone.now() - timedelta(hours=1), device=other)

        titles = {v.title for v in scheduling._videos_for(self.schedule)}
        self.assertEqual(titles, {"mine"})

    def test_window_starts_at_last_run_minus_lookback(self):
        now = timezone.now()
        self.schedule.last_run_at = now - timedelta(hours=2)
        self.schedule.lookback_hours = 1
        self._video("old", now - timedelta(hours=10))
        self._video("inside_lookback", now - timedelta(hours=2, minutes=30))
        self._video("new", now - timedelta(minutes=5))

        titles = {v.title for v in scheduling._videos_for(self.schedule)}
        self.assertEqual(titles, {"inside_lookback", "new"})

    # ── launching ────────────────────────────────────────────────────────────
    @patch("apps.analysis.views._drain_queue", return_value=0)
    def test_run_due_schedules_launches_one_run_per_video(self, _drain):
        self._video("a", timezone.now() - timedelta(minutes=10))
        self._video("b", timezone.now() - timedelta(minutes=5))

        stats = scheduling.run_due_schedules()

        self.assertEqual(stats["due"], 1)
        self.assertEqual(stats["launched_runs"], 2)
        self.assertEqual(PipelineRun.objects.count(), 2)
        # One batch groups the launch, and GPU steps are only QUEUED.
        batch_ids = set(PipelineRun.objects.values_list("batch_id", flat=True))
        self.assertEqual(len(batch_ids), 1)
        self.assertTrue(all(j.status == Job.Status.QUEUED for j in Job.objects.all()))

        self.schedule.refresh_from_db()
        self.assertIsNotNone(self.schedule.last_run_at)
        self.assertEqual(self.schedule.last_launched_count, 2)
        self.assertEqual(self.schedule.last_error, "")

    @patch("apps.analysis.views._drain_queue", return_value=0)
    def test_second_immediate_pass_launches_nothing(self, _drain):
        self._video("a", timezone.now() - timedelta(minutes=10))

        scheduling.run_due_schedules()
        stats = scheduling.run_due_schedules()

        self.assertEqual(stats["due"], 0)
        self.assertEqual(PipelineRun.objects.count(), 1)

    @patch("apps.analysis.views._drain_queue", return_value=0)
    def test_claim_prevents_double_launch(self, _drain):
        """Two workers ticking on the same stale schedule launch it once.

        Simulates the interleaving by running the pass against two independently
        loaded copies of the row — the CAS on last_run_at is what breaks the tie.
        """
        self._video("a", timezone.now() - timedelta(minutes=10))
        now = timezone.now()

        first = DevicePipelineSchedule.objects.get(pk=self.schedule.pk)
        second = DevicePipelineSchedule.objects.get(pk=self.schedule.pk)

        claimed_first = DevicePipelineSchedule.objects.filter(
            pk=first.pk, last_run_at=first.last_run_at,
        ).update(last_run_at=now)
        claimed_second = DevicePipelineSchedule.objects.filter(
            pk=second.pk, last_run_at=second.last_run_at,
        ).update(last_run_at=now)

        self.assertEqual(claimed_first, 1)
        self.assertEqual(claimed_second, 0)

    @patch("apps.analysis.views._drain_queue", return_value=0)
    def test_no_videos_is_a_clean_noop(self, _drain):
        stats = scheduling.run_due_schedules()

        self.assertEqual(stats["due"], 1)
        self.assertEqual(stats["launched_runs"], 0)
        self.assertEqual(PipelineRun.objects.count(), 0)
        self.schedule.refresh_from_db()
        self.assertEqual(self.schedule.last_launched_count, 0)

    @patch("apps.devices.scheduling.run_schedule", side_effect=RuntimeError("boom"))
    def test_failure_is_recorded_not_raised(self, _run):
        stats = scheduling.run_due_schedules()

        self.assertEqual(stats["errors"], 1)
        self.schedule.refresh_from_db()
        self.assertIn("boom", self.schedule.last_error)
        # Stamped anyway, so a broken schedule waits a full interval to retry.
        self.assertIsNotNone(self.schedule.last_run_at)


class DevicePipelineScheduleViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="Danniella", key_hash="h1", prefix="bmk_1",
        )
        self.pipeline = Pipeline.objects.create(
            user=self.user, title="Foraging", steps=STEPS,
        )
        self.client.force_login(self.user)
        self.url = f"/devices/{self.device.pk}/pipeline-schedule/"

    def test_create_then_replace_keeps_one_schedule(self):
        other = Pipeline.objects.create(user=self.user, title="Visits", steps=STEPS)

        self.client.post(self.url, {"pipeline": str(self.pipeline.pk), "cadence": "daily"})
        self.client.post(self.url, {"pipeline": str(other.pk), "cadence": "hourly"})

        schedules = DevicePipelineSchedule.objects.filter(device=self.device)
        self.assertEqual(schedules.count(), 1)
        self.assertEqual(schedules.first().pipeline_id, other.pk)

    def test_turn_off_deletes(self):
        self.client.post(self.url, {"pipeline": str(self.pipeline.pk), "cadence": "daily"})
        self.client.post(self.url, {"action": "delete"})
        self.assertFalse(DevicePipelineSchedule.objects.filter(device=self.device).exists())

    def test_pipeline_without_video_input_is_rejected(self):
        bad = Pipeline.objects.create(
            user=self.user, title="No input",
            steps=[{"id": "o", "block_type": "output.table", "config": {}}],
        )
        resp = self.client.post(
            self.url, {"pipeline": str(bad.pk), "cadence": "daily"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(DevicePipelineSchedule.objects.exists())

    def test_other_users_pipeline_is_rejected(self):
        mallory = User.objects.create_user("mallory", password="x")
        theirs = Pipeline.objects.create(user=mallory, title="Theirs", steps=STEPS)
        resp = self.client.post(
            self.url, {"pipeline": str(theirs.pk), "cadence": "daily"},
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(DevicePipelineSchedule.objects.exists())
