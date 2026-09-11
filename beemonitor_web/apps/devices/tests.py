"""Tests for the device 'Activity over time' series.

Covers the range controls (presets + custom From–To window) and the merge of the
device's on-card histogram with uploaded clips. Every recorded clip counts as
activity — the on-device bee-confirmation filter was removed, so there is no
longer a confirmed/unconfirmed split.
"""

import json
from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.videos.models import Video
from .models import Device, DeviceHeartbeat
from .views import _build_activity_series

User = get_user_model()


def _total(series_result):
    return sum(point["v"] for point in series_result["activity_series"])


class ActivitySeriesTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user("alice", password="x")
        cls.device = Device.objects.create(
            owner=cls.user, name="BeeMonitor2", key_hash="h2", prefix="bmk_2",
        )
        recorded = timezone.now() - timedelta(days=2)

        def mk(title):
            return Video.objects.create(
                user=cls.user, device=cls.device, title=title,
                storage_key=f"alice/{title}.mp4", file_size_bytes=1,
                status=Video.Status.READY, recorded_at=recorded, metadata={},
            )

        cls.v1, cls.v2, cls.v3 = mk("clip-a"), mk("clip-b"), mk("clip-c")

    def test_counts_every_recorded_clip(self):
        self.assertEqual(_total(_build_activity_series(self.device, "30d")), 3)

    def test_custom_range_window_bounds_results(self):
        # Add an older clip outside a narrow custom window.
        old = timezone.now() - timedelta(days=20)
        Video.objects.create(
            user=self.user, device=self.device, title="old", storage_key="alice/old.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=old, metadata={},
        )
        # setUpTestData's 3 clips are ~2 days ago. A window covering only "today
        # ±3 days" should include those 3 and exclude the 20-day-old one.
        today = timezone.now().date()
        start = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        res = _build_activity_series(self.device, "7d", start=start, end=end)
        self.assertEqual(res["activity_custom"], True)
        self.assertEqual(res["activity_start"], start)
        self.assertEqual(_total(res), 3)  # the 3 recent clips, not the 20-day-old one

    def test_invalid_custom_range_falls_back_to_preset(self):
        # start > end -> ignored, the 30d preset counts all 3.
        res = _build_activity_series(self.device, "30d",
                                     start="2026-12-31", end="2026-01-01")
        self.assertEqual(res["activity_custom"], False)
        self.assertEqual(_total(res), 3)

    def test_detail_page_renders_date_range_controls(self):
        # Render smoke test: the From–To controls appear and the page 200s.
        self.client.force_login(self.user)
        resp = self.client.get(reverse("devices:detail", args=[self.device.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'id="act-start"')
        self.assertContains(resp, 'id="act-end"')
        self.assertContains(resp, 'id="act-apply"')

    def test_oncard_histogram_merged_hourly(self):
        # 7d is hour-granularity, where the device's on-card histogram is merged
        # in (max per bucket) so clips show before they upload.
        hour_key = (timezone.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H")
        DeviceHeartbeat.objects.create(
            device=self.device, metrics={"activity_by_hour": {hour_key: 50}},
        )
        self.assertGreaterEqual(_total(_build_activity_series(self.device, "7d")), 50)


class MotionCalibrationDisplayTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="BeeMonitor3", key_hash="h3", prefix="bmk_3",
        )
        self.client.force_login(self.user)

    def test_detail_shows_learned_calibration_window(self):
        DeviceHeartbeat.objects.create(device=self.device, metrics={
            "motion_calibration": {
                "min_area": 24.0, "max_area": 480.0, "raw_p5": 40.0, "raw_p95": 300.0,
                "n_samples": 42, "n_clips": 6, "age_days": 1.2,
            },
        })
        html = self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()
        self.assertIn("Auto-calibration", html)
        self.assertIn("24.0", html)
        self.assertIn("480.0", html)
        self.assertIn("42", html)

    def test_detail_flags_few_samples(self):
        DeviceHeartbeat.objects.create(device=self.device, metrics={
            "motion_calibration": {"min_area": 10.0, "max_area": 90.0, "n_samples": 8, "age_days": 0.5},
        })
        html = self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()
        self.assertIn("few samples", html)

    def test_detail_no_calibration_reported(self):
        DeviceHeartbeat.objects.create(device=self.device, metrics={})
        html = self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()
        self.assertIn("no learned window reported", html)


class FleetUpdatePendingStateTests(TestCase):
    """The 'Updating →' state must survive a refresh (the reported bug) and clear
    once the device reports the target version (or the request goes stale)."""

    def setUp(self):
        self.owner = User.objects.create_user("alice", password="x")
        self.device, self.raw_key = Device.create_with_key(self.owner, "BeeMonitor9")

    def test_fleet_update_persists_target(self):
        from unittest.mock import patch
        self.client.force_login(self.owner)
        fake = {"version": "2026.06.27-abc1234", "url": "https://x", "sha256": "s",
                "sig": "g", "reqs_hash": "r"}
        with patch("apps.devices.views._resolve_edge_descriptor", return_value=fake):
            resp = self.client.post(
                reverse("devices:fleet_update"),
                data=json.dumps({"device_ids": [self.device.pk]}),
                content_type="application/json",
            )
        self.assertEqual(resp.status_code, 200)
        self.device.refresh_from_db()
        self.assertEqual(self.device.update_target, "2026.06.27-abc1234")
        self.assertIsNotNone(self.device.update_requested_at)

    def test_list_shows_updating_badge(self):
        self.device.update_target = "2026.06.27-abc1234"
        self.device.update_requested_at = timezone.now()
        self.device.save()
        DeviceHeartbeat.objects.create(device=self.device, metrics={"code_commit": "old123"})
        self.client.force_login(self.owner)
        html = self.client.get(reverse("devices:list")).content.decode()
        self.assertIn("Updating", html)
        self.assertIn("2026.06.27-abc1234", html)

    def _beat(self, metrics):
        return self.client.post(
            reverse("devices-heartbeat"), data={"metrics": json.dumps(metrics)},
            HTTP_AUTHORIZATION=f"Bearer {self.raw_key}",
        )

    def test_heartbeat_clears_target_when_version_lands(self):
        self.device.update_target = "v-new"
        self.device.update_requested_at = timezone.now()
        self.device.save()
        self._beat({"code_commit": "v-new"})
        self.device.refresh_from_db()
        self.assertEqual(self.device.update_target, "")  # cleared — update landed

    def test_heartbeat_keeps_target_when_version_differs(self):
        self.device.update_target = "v-new"
        self.device.update_requested_at = timezone.now()
        self.device.save()
        self._beat({"code_commit": "v-old"})
        self.device.refresh_from_db()
        self.assertEqual(self.device.update_target, "v-new")  # still updating

    def test_heartbeat_clears_stale_target(self):
        self.device.update_target = "v-new"
        self.device.update_requested_at = timezone.now() - timedelta(hours=3)
        self.device.save()
        self._beat({"code_commit": "v-old"})  # still old, but request is stale
        self.device.refresh_from_db()
        self.assertEqual(self.device.update_target, "")  # gave up showing "updating"
