"""0038 copies each DeviceStill into the videos table as a photo, keeping its
old id (devices in the field free the file by it) and attaching a burst photo
to the clip it preceded."""

from datetime import datetime, timedelta, timezone as dt_tz

from django.conf import settings
from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

BEFORE = [("devices", "0037_motion_burst_and_device_delete"), ("videos", "0010_photos_in_videos")]
AFTER = [("devices", "0038_stills_into_videos")]


class StillsIntoVideosMigrationTests(TransactionTestCase):
    def test_stills_become_photos(self):
        ex = MigrationExecutor(connection)
        ex.migrate(BEFORE)
        apps = ex.loader.project_state(BEFORE).apps
        User = apps.get_model(*settings.AUTH_USER_MODEL.split("."))
        Device = apps.get_model("devices", "Device")
        Still = apps.get_model("devices", "DeviceStill")
        Video = apps.get_model("videos", "Video")

        owner = User.objects.create(username="m")
        device = Device.objects.create(owner=owner, name="hive")
        t0 = datetime(2026, 9, 25, 9, 12, 4, tzinfo=dt_tz.utc)
        clip = Video.objects.create(user=owner, device=device, title="c", storage_key="v/c.mp4",
                                    file_size_bytes=1, recorded_at=t0 + timedelta(seconds=5))
        burst = Still.objects.create(device=device, taken_at=t0, storage_key="s/b.jpg",
                                     thumb_key="s/b.thumb.jpg", width=9152, height=6944,
                                     source="burst", burst_id="B", burst_index=0)
        periodic = Still.objects.create(device=device, taken_at=t0 + timedelta(hours=1),
                                        storage_key="s/p.jpg", device_delete_requested=True)

        ex = MigrationExecutor(connection)
        ex.migrate(AFTER)
        Video = ex.loader.project_state(AFTER).apps.get_model("videos", "Video")
        photos = {p.storage_key: p for p in Video.objects.filter(kind="photo")}
        self.assertEqual(set(photos), {"s/b.jpg", "s/p.jpg"})
        b, p = photos["s/b.jpg"], photos["s/p.jpg"]
        self.assertEqual(b.parent_id, clip.pk)
        self.assertEqual((b.width, b.metadata["thumb_key"], b.metadata["burst_index"]),
                         (9152, "s/b.thumb.jpg", 0))
        self.assertEqual(b.metadata["legacy_still_id"], burst.pk)
        self.assertIsNone(p.parent_id)
        self.assertTrue(p.device_delete_requested)
        self.assertEqual(p.metadata["legacy_still_id"], periodic.pk)
        self.assertNotIn("devicestill", [t.lower() for t in connection.introspection.table_names()
                                         if t == "devices_devicestill"])
