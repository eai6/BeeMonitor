"""Tests for the device cross-video foraging-trips download (concat streamer)."""

from datetime import date
from unittest import mock

from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse

from apps.analysis.models import DailyForagingSummary
from apps.devices.models import Device


class DeviceTripsDownloadTest(TestCase):
    def setUp(self):
        self.user = User.objects.create(username="dt")
        self.client.force_login(self.user)
        self.device, _ = Device.create_with_key(self.user, "dev")
        # Three days of pre-computed cross-video trips CSVs; one with no path.
        for d, path in ((date(2026, 7, 6), "u/daily_trips/s_dev1_2026-07-06.csv"),
                        (date(2026, 7, 8), "u/daily_trips/s_dev1_2026-07-08.csv"),
                        (date(2026, 7, 12), "u/daily_trips/s_dev1_2026-07-12.csv"),
                        (date(2026, 7, 20), "")):  # not yet computed → skipped
            DailyForagingSummary.objects.create(
                user=self.user, site_name="s", device=self.device, date=d,
                trips_csv_path=path)

    def _fake_s3(self):
        class FakeS3:
            def download_to_stream(self, container, path, buf):
                # Each day's CSV: header + one row tagged with its path.
                buf.write(("nest,exit_time,entry_time,duration_sec\r\n"
                           f"9,{path},x,5755.6\r\n").encode("utf-8"))
        return FakeS3()

    def test_concatenates_daily_csvs_header_once(self):
        with mock.patch("apps.analysis.views.get_s3_client", return_value=self._fake_s3()):
            resp = self.client.get(reverse("analysis:download_device_trips"),
                                   {"device": self.device.pk})
            body = b"".join(resp.streaming_content).decode()
        lines = [ln for ln in body.splitlines() if ln]
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(lines[0], "nest,exit_time,entry_time,duration_sec")   # header once
        self.assertEqual(len(lines), 4)                                        # header + 3 days
        self.assertTrue(all("5755.6" in ln for ln in lines[1:]))               # cross-video duration

    def test_date_range_limits_days(self):
        with mock.patch("apps.analysis.views.get_s3_client", return_value=self._fake_s3()):
            resp = self.client.get(reverse("analysis:download_device_trips"),
                                   {"device": self.device.pk, "from": "2026-07-07", "to": "2026-07-10"})
            body = b"".join(resp.streaming_content).decode()
        lines = [ln for ln in body.splitlines() if ln]
        # Only 2026-07-08 falls in [07-07, 07-10].
        self.assertEqual(len(lines), 2)  # header + 1 day
        self.assertIn("2026-07-08", body)

    def test_other_users_device_is_403(self):
        other = User.objects.create(username="other")
        self.client.force_login(other)
        resp = self.client.get(reverse("analysis:download_device_trips"),
                               {"device": self.device.pk})
        self.assertEqual(resp.status_code, 404)
