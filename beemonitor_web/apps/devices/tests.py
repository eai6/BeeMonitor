"""Tests for the device 'Activity over time' series.

Focus: the "Confirmed" toggle must mean the same thing as the Processing page's
"Confirmed bee" filter — strictly bee_confirmed=True. Untagged clips (no
bee_confirmed key) belong to "all" only, not "confirmed". Uses the 30d range
(daily granularity) so the on-card heartbeat histogram is not involved and the
uploaded-video filter is tested in isolation.
"""

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


class ActivitySeriesConfirmedFilterTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.user = User.objects.create_user("alice", password="x")
        cls.device = Device.objects.create(
            owner=cls.user, name="BeeMonitor2", key_hash="h2", prefix="bmk_2",
        )
        recorded = timezone.now() - timedelta(days=2)

        def mk(title, metadata):
            return Video.objects.create(
                user=cls.user, device=cls.device, title=title,
                storage_key=f"alice/{title}.mp4", file_size_bytes=1,
                status=Video.Status.READY, recorded_at=recorded, metadata=metadata,
            )

        cls.v_true = mk("confirmed", {"bee_confirmed": True})
        cls.v_untagged = mk("untagged", {})              # no bee_confirmed key
        cls.v_false = mk("rejected", {"bee_confirmed": False})

    def test_confirmed_counts_only_true(self):
        # The bug: untagged clips were counted as confirmed. Now strict.
        self.assertEqual(_total(_build_activity_series(self.device, "30d", "confirmed")), 1)

    def test_unconfirmed_counts_only_false(self):
        self.assertEqual(_total(_build_activity_series(self.device, "30d", "unconfirmed")), 1)

    def test_all_counts_everything_including_untagged(self):
        self.assertEqual(_total(_build_activity_series(self.device, "30d", "all")), 3)

    def test_custom_range_window_bounds_results(self):
        # Add an older clip outside a narrow custom window.
        old = timezone.now() - timedelta(days=20)
        Video.objects.create(
            user=self.user, device=self.device, title="old", storage_key="alice/old.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=old,
            metadata={"bee_confirmed": True},
        )
        # setUpTestData's 3 clips are ~2 days ago. A window covering only "today
        # ±3 days" should include those 3 (all filter) and exclude the 20-day-old.
        today = timezone.now().date()
        start = (today - timedelta(days=3)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        res = _build_activity_series(self.device, "7d", "all", start=start, end=end)
        self.assertEqual(res["activity_custom"], True)
        self.assertEqual(res["activity_start"], start)
        self.assertEqual(_total(res), 3)  # the 3 recent clips, not the 20-day-old one

    def test_custom_range_confirmed_is_strict(self):
        today = timezone.now().date()
        start = (today - timedelta(days=5)).strftime("%Y-%m-%d")
        end = today.strftime("%Y-%m-%d")
        res = _build_activity_series(self.device, "7d", "confirmed", start=start, end=end)
        self.assertEqual(_total(res), 1)  # only the bee_confirmed=True clip

    def test_invalid_custom_range_falls_back_to_preset(self):
        # start > end -> ignored, preset 'all' over 30d counts all 3.
        res = _build_activity_series(self.device, "30d", "all",
                                     start="2026-12-31", end="2026-01-01")
        self.assertEqual(res["activity_custom"], False)
        self.assertEqual(_total(res), 3)

    def test_detail_page_renders_date_range_controls(self):
        # Render smoke test: the new From–To controls appear and the page 200s.
        self.client.force_login(self.user)
        resp = self.client.get(reverse("devices:detail", args=[self.device.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'id="act-start"')
        self.assertContains(resp, 'id="act-end"')
        self.assertContains(resp, 'id="act-apply"')

    def test_confirmed_ignores_oncard_histogram_hourly(self):
        # 7d is hour-granularity, where the on-card histogram is merged in. That
        # histogram (activity_by_hour) counts confirmed+untagged, so "confirmed"
        # must SKIP it and report only the 1 uploaded bee_confirmed=True video.
        hour_key = (timezone.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H")
        DeviceHeartbeat.objects.create(
            device=self.device, metrics={"activity_by_hour": {hour_key: 50}},
        )
        self.assertEqual(_total(_build_activity_series(self.device, "7d", "confirmed")), 1)
        # "all" still includes the on-card histogram (max per bucket).
        self.assertGreaterEqual(_total(_build_activity_series(self.device, "7d", "all")), 50)

    def test_confirmed_uses_strict_oncard_histogram(self):
        # New firmware reports confirmed_by_hour (strict). "confirmed" must use it
        # (7) for the on-card preview, NOT the loose activity_by_hour (50).
        hour_key = (timezone.now() - timedelta(hours=2)).strftime("%Y-%m-%dT%H")
        DeviceHeartbeat.objects.create(device=self.device, metrics={
            "activity_by_hour": {hour_key: 50},     # confirmed + untagged (loose)
            "confirmed_by_hour": {hour_key: 7},     # strict confirmed
            "unconfirmed_by_hour": {hour_key: 9},
        })
        confirmed_total = _total(_build_activity_series(self.device, "7d", "confirmed"))
        self.assertGreaterEqual(confirmed_total, 7)   # picks up the strict 7
        self.assertLess(confirmed_total, 50)          # ignores the loose 50


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
