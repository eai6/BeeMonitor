"""Full-resolution stills (memory/40): uploaded exactly like videos (the same
initiate -> PUT -> complete calls, with kind "still"), a per-device interval
pushed to the recorder, "take one now", and the gallery / viewer pages."""

from datetime import datetime, timezone as dt_tz
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from .models import Device, DeviceShare, DeviceStill

User = get_user_model()
AUTH = "HTTP_AUTHORIZATION"


class StillUploadTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("st", password="x")
        self.device, self.raw_key = Device.create_with_key(self.user, "hive")
        self.prefix = f"users/{self.user.pk}/devices/{self.device.pk}/"

    def post(self, path, data):
        return self.client.post(path, data, content_type="application/json",
                                **{AUTH: f"Bearer {self.raw_key}"})

    def initiate(self, kind, filename="2026-09-25_10_00_00.jpg"):
        with patch("apps.api.uploads.get_s3_client") as s3:
            s3.return_value.generate_presigned_url.return_value = "https://put"
            return self.post("/api/v1/uploads/initiate", {
                "filename": filename, "size_bytes": 15_000_000, "kind": kind,
                "recorded_at": "2026-09-25T10:00:00Z"})

    def complete(self, key, **extra):
        with patch("apps.api.uploads.get_s3_client") as s3:
            s3.return_value.blob_exists.return_value = True
            return self.post("/api/v1/uploads/complete", dict({
                "storage_key": key, "file_size_bytes": 15_000_000, "kind": "still",
                "recorded_at": "2026-09-25T10:00:00Z", "width": 9152, "height": 6944,
                "sensor_mode": "64mp", "lens_position": 6.2}, **extra))

    def test_a_still_gets_a_key_under_stills(self):
        r = self.initiate("still")
        self.assertEqual(r.status_code, 200)
        key = r.json()["storage_key"]
        self.assertTrue(key.startswith(self.prefix + "stills/2026/09/25/"))
        self.assertTrue(key.endswith(".jpg"))
        self.assertEqual(r.json()["headers"]["Content-Type"], "image/jpeg")

    def test_the_preview_is_marked_as_one(self):
        self.assertTrue(self.initiate("still_thumb").json()["storage_key"].endswith(".thumb.jpg"))

    def test_a_video_still_refuses_a_jpg(self):
        self.assertEqual(self.initiate("", filename="x.jpg").status_code, 400)

    def test_a_still_must_be_a_jpg(self):
        self.assertEqual(self.initiate("still", filename="x.png").status_code, 400)

    def test_complete_makes_a_still_not_a_video(self):
        from apps.videos.models import Video
        key = self.prefix + "stills/2026/09/25/a.jpg"
        r = self.complete(key, thumb_key=self.prefix + "stills/2026/09/25/a.thumb.jpg")
        self.assertEqual(r.status_code, 201)
        s = DeviceStill.objects.get()
        self.assertEqual((s.device, s.width, s.height, s.sensor_mode, s.lens_position),
                         (self.device, 9152, 6944, "64mp", 6.2))
        self.assertEqual(s.taken_at, datetime(2026, 9, 25, 10, tzinfo=dt_tz.utc))
        self.assertTrue(s.thumb_key.endswith("a.thumb.jpg"))
        self.assertFalse(Video.objects.exists())

    def test_a_retried_complete_does_not_duplicate(self):
        key = self.prefix + "stills/2026/09/25/a.jpg"
        self.assertEqual(self.complete(key).status_code, 201)
        self.assertEqual(self.complete(key).status_code, 200)
        self.assertEqual(DeviceStill.objects.count(), 1)

    def test_another_devices_key_is_refused(self):
        r = self.complete("users/999/devices/999/stills/2026/09/25/a.jpg")
        self.assertEqual(r.status_code, 403)
        self.assertFalse(DeviceStill.objects.exists())

    def test_a_foreign_preview_key_is_dropped(self):
        self.complete(self.prefix + "stills/a.jpg", thumb_key="users/1/devices/999/x.thumb.jpg")
        self.assertEqual(DeviceStill.objects.get().thumb_key, "")


class StillSettingTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("own", password="x")
        self.device, self.raw_key = Device.create_with_key(self.owner, "hive")
        self.viewer = User.objects.create_user("vw", password="x")
        DeviceShare.objects.create(device=self.device, user=self.viewer, role="viewer")

    def test_the_interval_reaches_the_device(self):
        self.client.force_login(self.owner)
        self.client.post(reverse("devices:stills_setting", args=[self.device.pk]), {"interval": 30})
        self.device.refresh_from_db()
        self.assertEqual(self.device.stills_interval_min, 30)
        r = self.client.get(reverse("devices-command"), **{AUTH: f"Bearer {self.raw_key}"})
        self.assertEqual(r.json()["stills_interval_min"], 30)

    def test_under_15_minutes_is_refused(self):
        self.client.force_login(self.owner)
        self.client.post(reverse("devices:stills_setting", args=[self.device.pk]), {"interval": 5})
        self.device.refresh_from_db()
        self.assertEqual(self.device.stills_interval_min, 0)

    def test_a_viewer_cannot_change_it_or_take_one(self):
        self.client.force_login(self.viewer)
        for name, data in (("devices:stills_setting", {"interval": 30}), ("devices:take_still", {})):
            self.client.post(reverse(name, args=[self.device.pk]), data)
        self.device.refresh_from_db()
        self.assertEqual((self.device.stills_interval_min, self.device.pending_command), (0, ""))

    def test_take_one_now_is_a_device_command(self):
        self.client.force_login(self.owner)
        self.client.post(reverse("devices:take_still", args=[self.device.pk]))
        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "take_still")


class StillPagesTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("own", password="x")
        self.device, _ = Device.create_with_key(self.owner, "hive")
        self.stills = [DeviceStill.objects.create(
            device=self.device, storage_key=f"k/{i}.jpg", thumb_key=f"k/{i}.thumb.jpg",
            taken_at=datetime(2026, 9, 25, 8 + i, tzinfo=dt_tz.utc), width=9152, height=6944,
            file_size_bytes=15_000_000) for i in range(3)]

    def get(self, name, *args, user=None):
        self.client.force_login(user or self.owner)
        with patch("apps.devices.stills_views._presign", side_effect=lambda k: "https://s3/" + k):
            return self.client.get(reverse(name, args=[self.device.pk, *args]))

    def test_the_gallery_shows_the_day(self):
        r = self.get("devices:stills")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.context["cards"]), 3)
        self.assertIn("https://s3/k/0.thumb.jpg", r.content.decode())

    def test_the_viewer_links_prev_and_next(self):
        r = self.get("devices:still_detail", self.stills[1].pk)
        self.assertEqual((r.context["prev"], r.context["next"]), (self.stills[0], self.stills[2]))
        self.assertIn("https://s3/k/1.jpg", r.content.decode())

    def test_the_device_page_shows_the_setting_and_recent_stills(self):
        html = self.get("devices:detail").content.decode()
        self.assertIn("Full-resolution stills", html)
        self.assertIn(reverse("devices:stills_setting", args=[self.device.pk]), html)
        self.assertIn(reverse("devices:still_detail", args=[self.device.pk, self.stills[2].pk]), html)

    def test_a_stranger_cannot_see_them(self):
        stranger = User.objects.create_user("x", password="x")
        self.assertIn(self.get("devices:stills", user=stranger).status_code, (403, 404))
        self.assertIn(self.get("devices:still_detail", self.stills[0].pk, user=stranger).status_code,
                      (403, 404))
