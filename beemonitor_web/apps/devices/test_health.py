"""Device health history: per-minute samples, raw-beat pruning, the chart
series, and blank (not stale) tiles while a device is offline."""

import json
from datetime import timedelta
from zoneinfo import ZoneInfo

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from .health import RAW_HEARTBEAT_DAYS, health_series, prune_heartbeats
from .models import Device, DeviceHealthSample, DeviceHeartbeat

User = get_user_model()


class HealthHistoryTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("hana", password="x")
        self.device, self.raw_key = Device.create_with_key(self.user, "hive")
        self.client.force_login(self.user)

    def _beat(self, metrics):
        return self.client.post(
            reverse("devices-heartbeat"), data={"metrics": json.dumps(metrics)},
            HTTP_AUTHORIZATION=f"Bearer {self.raw_key}")

    def _old_beat(self, age, metrics, image=""):
        hb = DeviceHeartbeat.objects.create(device=self.device, metrics=metrics,
                                            image_storage_key=image)
        DeviceHeartbeat.objects.filter(pk=hb.pk).update(created_at=timezone.now() - age)
        return hb

    def test_beats_in_one_minute_share_a_row_and_the_latest_wins(self):
        self._beat({"storage_pct": 40.0, "cpu_temp_c": 50})
        self._beat({"storage_pct": 41.0, "cpu_temp_c": 51, "uptime_seconds": 99})
        rows = DeviceHealthSample.objects.filter(device=self.device)
        self.assertEqual(rows.count(), 1)
        self.assertEqual((rows[0].storage_pct, rows[0].uptime_seconds), (41.0, 99))

    def test_prune_folds_old_beats_into_history_then_deletes_them(self):
        old = self._old_beat(timedelta(days=RAW_HEARTBEAT_DAYS + 1), {"storage_pct": 12.5})
        kept_image = self._old_beat(timedelta(days=RAW_HEARTBEAT_DAYS + 1), {}, image="k.jpg")
        recent = self._old_beat(timedelta(days=1), {"storage_pct": 13.0})

        self.assertEqual(prune_heartbeats(), 1)
        ids = set(DeviceHeartbeat.objects.values_list("pk", flat=True))
        self.assertNotIn(old.pk, ids)
        self.assertIn(kept_image.pk, ids)
        self.assertIn(recent.pk, ids)
        self.assertTrue(DeviceHealthSample.objects.filter(device=self.device, storage_pct=12.5).exists())

    def test_backfill_command_is_idempotent(self):
        self._old_beat(timedelta(hours=3), {"cpu_temp_c": 44})
        call_command("backfill_health_samples", stdout=open("/dev/null", "w"))
        call_command("backfill_health_samples", stdout=open("/dev/null", "w"))
        self.assertEqual(DeviceHealthSample.objects.filter(device=self.device).count(), 1)

    def test_series_leaves_gaps_when_off_and_marks_reboots(self):
        now = timezone.now().replace(second=0, microsecond=0)
        for mins_ago, uptime in ((200, 5000), (190, 5600), (20, 60)):
            DeviceHealthSample.objects.create(device=self.device, minute=now - timedelta(minutes=mins_ago),
                                              storage_pct=40.0, uptime_seconds=uptime)
        pts = health_series(self.device, "24h", ZoneInfo("UTC"), now=now)["points"]
        self.assertEqual(len(pts), 288)
        self.assertEqual(sum(1 for p in pts if p["storage_pct"] is not None), 3)
        self.assertEqual(sum(1 for p in pts if p["reboot"]), 1)
        self.assertIsNone(pts[-1]["storage_pct"])  # nothing in the last 5 minutes

    def test_health_endpoint(self):
        r = self.client.get(reverse("devices:health", args=[self.device.pk]) + "?range=7d")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["range"], "7d")

    def test_offline_device_shows_blank_tiles_not_stale_numbers(self):
        self._beat({"storage_pct": 77.7, "cpu_temp_c": 63, "uptime_human": "9d 1h"})
        Device.objects.filter(pk=self.device.pk).update(
            last_seen_at=timezone.now() - timedelta(hours=6))

        html = self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()
        self.assertNotIn("77.7%", html)
        self.assertNotIn("63°C", html)
        self.assertNotIn("9d 1h", html)
        self.assertIn("service state is unknown", html)

        d = self.client.get(reverse("devices:status", args=[self.device.pk])).json()
        self.assertFalse(d["online"])
        self.assertIsNone(d["storage_pct"])
        self.assertIsNone(d["cpu_temp_c"])
        self.assertIsNone(d["uptime_human"])

    def test_online_device_still_shows_its_numbers(self):
        self._beat({"storage_pct": 77.7, "cpu_temp_c": 63})
        html = self.client.get(reverse("devices:detail", args=[self.device.pk])).content.decode()
        self.assertIn("77.7%", html)
        self.assertIn("Device health", html)
        # Inside the collapsed Advanced settings, like the other sections.
        self.assertLess(html.index("<details"), html.index("Device health"))
        self.assertLess(html.index("Device health"), html.index("</details>"))
