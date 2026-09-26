"""Full-resolution photos live in the videos table (memory/40).

They upload exactly like videos (initiate -> PUT -> complete, kind "still") and
become ``Video(kind="photo")``. A motion burst's photos belong to the clip
recorded after them and show on its page; periodic photos appear in the
Processing hub's Photos. Everything that treats a row as a video file —
analysis, pipelines, schedules, annotation, counts — sees clips only, through
the default manager.
"""

from datetime import datetime, timedelta, timezone as dt_tz
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.videos.models import PendingDeviceDeletion, Video

from .models import Device, DeviceShare

User = get_user_model()
AUTH = "HTTP_AUTHORIZATION"
T0 = datetime(2026, 9, 25, 9, 12, 4, tzinfo=dt_tz.utc)


class PhotoTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("own", password="x")
        self.device, self.raw_key = Device.create_with_key(self.owner, "hive")
        self.prefix = f"users/{self.owner.pk}/devices/{self.device.pk}/"

    def api(self, path, data):
        return self.client.post(path, data, content_type="application/json",
                                **{AUTH: f"Bearer {self.raw_key}"})

    def complete_photo(self, key, **extra):
        with patch("apps.api.uploads.get_s3_client") as s3:
            s3.return_value.blob_exists.return_value = True
            return self.api("/api/v1/uploads/complete", dict({
                "storage_key": self.prefix + key, "file_size_bytes": 15_000_000,
                "kind": "still", "recorded_at": T0.isoformat(), "width": 9152,
                "height": 6944, "sensor_mode": "64mp", "lens_position": 6.2}, **extra))

    def complete_clip(self, filename, at=T0):
        with patch("apps.api.uploads.get_s3_client") as s3, \
             patch("apps.videos.thumbnails.queue_thumbnail"):
            s3.return_value.blob_exists.return_value = True
            return self.api("/api/v1/uploads/complete", {
                "storage_key": self.prefix + filename, "file_size_bytes": 1_000_000,
                "recorded_at": at.isoformat(), "filename": filename})

    def photo(self, i=0, parent=None, source="schedule", at=T0):
        return Video.everything.create(
            user=self.owner, device=self.device, kind="photo", parent=parent,
            title=f"Photo {i}", storage_key=f"p/{i}.jpg", file_size_bytes=15_000_000,
            status="ready", recorded_at=at + timedelta(seconds=i), width=9152, height=6944,
            metadata={"thumb_key": f"p/{i}.thumb.jpg", "source": source,
                      "burst_index": i if parent else None})


class UploadTests(PhotoTestCase):
    def test_initiate_puts_a_still_under_stills(self):
        with patch("apps.api.uploads.get_s3_client") as s3:
            s3.return_value.generate_presigned_url.return_value = "https://put"
            r = self.api("/api/v1/uploads/initiate", {
                "filename": "a.jpg", "size_bytes": 15_000_000, "kind": "still",
                "recorded_at": T0.isoformat()})
        self.assertTrue(r.json()["storage_key"].startswith(self.prefix + "stills/2026/09/25/"))

    def test_a_still_becomes_a_photo_in_the_videos_table(self):
        r = self.complete_photo("stills/a.jpg", thumb_key=self.prefix + "stills/a.thumb.jpg")
        self.assertEqual(r.status_code, 201)
        p = Video.everything.get()
        self.assertEqual((p.kind, p.width, p.height, p.recorded_at), ("photo", 9152, 6944, T0))
        self.assertEqual(p.metadata["thumb_key"], self.prefix + "stills/a.thumb.jpg")
        self.assertEqual(r.json()["still_id"], p.pk)       # the device's sidecar id
        self.assertFalse(Video.objects.exists())            # not a clip

    def test_a_retried_complete_does_not_duplicate(self):
        self.complete_photo("stills/a.jpg")
        self.assertEqual(self.complete_photo("stills/a.jpg").status_code, 200)
        self.assertEqual(Video.everything.count(), 1)

    def test_a_burst_photo_joins_the_clip_already_uploaded(self):
        self.complete_clip("2026-09-25_09_12_07.mp4")
        clip = Video.objects.get()
        self.complete_photo("stills/b.jpg", source="burst", burst_id="B", burst_index=0,
                            clip="2026-09-25_09_12_07.mp4")
        self.assertEqual(Video.everything.get(kind="photo").parent, clip)

    def test_a_clip_adopts_its_burst_photos_that_came_first(self):
        for i in range(5):
            self.complete_photo(f"stills/b{i}.jpg", source="burst", burst_id="B",
                                burst_index=i, clip="2026-09-25_09_12_07.mp4")
        self.complete_clip("2026-09-25_09_12_07.mp4")
        clip = Video.objects.get()
        self.assertEqual(clip.burst_photos().count(), 5)

    def test_another_devices_key_is_refused(self):
        with patch("apps.api.uploads.get_s3_client") as s3:
            s3.return_value.blob_exists.return_value = True
            r = self.api("/api/v1/uploads/complete", {
                "storage_key": "users/9/devices/9/stills/a.jpg", "file_size_bytes": 5,
                "kind": "still"})
        self.assertEqual(r.status_code, 403)


class ClipsOnlyByDefaultTests(PhotoTestCase):
    """Photos must not leak into anything that treats a row as a video file."""

    def test_the_default_manager_is_clips_only(self):
        self.photo()
        self.assertFalse(Video.objects.exists())
        self.assertFalse(Video.accessible(self.owner).exists())
        self.assertEqual(Video.accessible(self.owner, photos=True).count(), 1)

    def test_a_pipeline_run_on_a_photo_does_nothing(self):
        p = self.photo()
        self.client.force_login(self.owner)
        from apps.pipelines.models import PipelineRun
        self.client.post(reverse("pipelines:run_on_videos"), {"video_ids": [p.pk]})
        self.assertFalse(PipelineRun.objects.exists())

    def test_the_device_page_counts_clips_only(self):
        self.photo()
        self.client.force_login(self.owner)
        r = self.client.get(reverse("devices:detail", args=[self.device.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context["on_device"], {"videos": 0, "stills": 1, "pending": 0})


class PagesTests(PhotoTestCase):
    def setUp(self):
        super().setUp()
        self.clip = Video.objects.create(
            user=self.owner, device=self.device, title="clip", storage_key="v/1.mp4",
            file_size_bytes=1, status="ready", recorded_at=T0 + timedelta(seconds=4))
        self.burst = [self.photo(i, parent=self.clip, source="burst") for i in (3, 0, 4, 1, 2)]
        self.periodic = self.photo(9)
        self.client.force_login(self.owner)

    def get(self, url, **params):
        with patch("config.storage.get_s3_client") as s3:
            s3.return_value.generate_presigned_url.side_effect = lambda b, k, **kw: "https://s3/" + k
            return self.client.get(url, params)

    def test_the_clip_page_shows_its_burst_in_order(self):
        r = self.get(reverse("videos:detail", args=[self.clip.pk]))
        self.assertEqual([p.metadata["burst_index"] for p in r.context["burst"]], [0, 1, 2, 3, 4])
        self.assertIn("Photos taken at the trigger", r.content.decode())

    def test_a_photo_page_is_a_viewer_with_its_clip(self):
        r = self.get(reverse("videos:detail", args=[self.burst[1].pk]))
        html = r.content.decode()
        self.assertEqual(r.context["clip"], self.clip)
        self.assertIn("https://s3/p/0.thumb.jpg", html)           # Fit = the preview
        self.assertIn("Taken right before this clip", html)
        self.assertNotIn("Analysis Jobs", html)

    def test_the_hub_lists_periodic_photos_not_burst_ones(self):
        r = self.get(reverse("analysis:processing"), kind="photo")
        self.assertEqual([v.pk for v in r.context["videos"]], [self.periodic.pk])
        r = self.get(reverse("analysis:processing"))
        self.assertEqual([v.pk for v in r.context["videos"]], [self.clip.pk])

    def test_a_photo_thumbnail_is_the_devices_preview(self):
        r = self.get(reverse("videos:thumbnail", args=[self.periodic.pk]))
        self.assertEqual(r["Location"], "https://s3/p/9.thumb.jpg")

    def test_deleting_a_clip_takes_its_burst_photos_and_tombstones_them(self):
        with patch("config.storage.get_s3_client"):
            self.client.post(reverse("videos:delete", args=[self.clip.pk]))
        self.assertEqual(list(Video.everything.values_list("pk", flat=True)), [self.periodic.pk])
        self.assertEqual(PendingDeviceDeletion.objects.filter(is_photo=True).count(), 5)
        self.assertEqual(PendingDeviceDeletion.objects.filter(is_photo=False).count(), 1)

    def test_a_stranger_cannot_open_a_photo(self):
        self.client.force_login(User.objects.create_user("x", password="x"))
        self.assertEqual(self.get(reverse("videos:detail", args=[self.periodic.pk])).status_code, 404)


class SettingsAndCleanupTests(PhotoTestCase):
    def test_both_settings_reach_the_device(self):
        self.client.force_login(self.owner)
        url = reverse("devices:stills_setting", args=[self.device.pk])
        self.client.post(url, {"burst": "on"})
        self.client.post(url, {"interval": 30})
        r = self.client.get(reverse("devices-command"), **{AUTH: f"Bearer {self.raw_key}"})
        self.assertEqual((r.json()["motion_burst_stills"], r.json()["stills_interval_min"]), (5, 30))

    def test_under_15_minutes_is_refused(self):
        self.client.force_login(self.owner)
        self.client.post(reverse("devices:stills_setting", args=[self.device.pk]), {"interval": 5})
        self.device.refresh_from_db()
        self.assertEqual(self.device.stills_interval_min, 0)

    def test_a_viewer_cannot_change_settings_or_free_space(self):
        viewer = User.objects.create_user("vw", password="x")
        DeviceShare.objects.create(device=self.device, user=viewer, role="viewer")
        self.photo()
        self.client.force_login(viewer)
        self.client.post(reverse("devices:stills_setting", args=[self.device.pk]), {"burst": "on"})
        self.client.post(reverse("devices:free_space", args=[self.device.pk]))
        self.device.refresh_from_db()
        self.assertEqual(self.device.motion_burst_stills, 0)
        self.assertFalse(Video.everything.filter(device_delete_requested=True).exists())

    def test_free_space_then_the_device_frees_photos_through_still_ids(self):
        clip = Video.objects.create(user=self.owner, device=self.device, title="c",
                                    storage_key="v/c.mp4", file_size_bytes=1, status="ready")
        p = self.photo()
        legacy = self.photo(1)
        legacy.metadata["legacy_still_id"] = 7
        legacy.save()
        self.client.force_login(self.owner)
        self.client.post(reverse("devices:free_space", args=[self.device.pk]))

        auth = {AUTH: f"Bearer {self.raw_key}"}
        body = self.client.get("/api/v1/devices/cleanup", **auth).json()
        self.assertEqual(body["video_ids"], [clip.pk])
        self.assertEqual(sorted(body["still_ids"]), sorted([p.pk, 7]))
        r = self.api("/api/v1/devices/cleanup", {"deleted_stills": [p.pk, 7]})
        self.assertEqual(r.json()["confirmed"], 2)
        self.assertEqual(self.client.get("/api/v1/devices/cleanup", **auth).json()["still_ids"], [])

    def test_take_one_now_is_a_device_command(self):
        self.client.force_login(self.owner)
        self.client.post(reverse("devices:take_still", args=[self.device.pk]))
        self.device.refresh_from_db()
        self.assertEqual(self.device.pending_command, "take_still")
