"""Full-resolution stills become photos in the videos table (memory/40).

Each DeviceStill is copied to a Video(kind="photo"); its old id is kept as
metadata["legacy_still_id"] because devices in the field hold
``still_id=<that id>`` sidecars and free the file by it. A burst photo is
attached to the clip it preceded (the first clip from the device starting
within 30 s). Then DeviceStill goes.
"""

from datetime import timedelta

from django.db import migrations


def stills_to_photos(apps, schema_editor):
    DeviceStill = apps.get_model("devices", "DeviceStill")
    Video = apps.get_model("videos", "Video")
    for s in DeviceStill.objects.select_related("device").iterator():
        if Video.objects.filter(kind="photo", storage_key=s.storage_key).exists():
            continue
        parent = None
        if s.burst_id:
            parent = (Video.objects.filter(
                device_id=s.device_id, kind="video",
                recorded_at__gte=s.taken_at - timedelta(seconds=5),
                recorded_at__lte=s.taken_at + timedelta(seconds=30))
                .order_by("recorded_at").first())
        Video.objects.create(
            user_id=s.device.owner_id, device_id=s.device_id, kind="photo", parent=parent,
            title=f"Photo {s.taken_at:%Y-%m-%d %H:%M:%S}", storage_key=s.storage_key,
            file_size_bytes=s.file_size_bytes, width=s.width or None, height=s.height or None,
            status="ready", recorded_at=s.taken_at,
            device_delete_requested=s.device_delete_requested,
            device_deleted_at=s.device_deleted_at,
            metadata={"device_id": s.device_id, "thumb_key": s.thumb_key,
                      "sensor_mode": s.sensor_mode, "lens_position": s.lens_position,
                      "source": s.source, "burst_id": s.burst_id,
                      "burst_index": s.burst_index, "legacy_still_id": s.id},
            year=s.taken_at.year, month=s.taken_at.month, day=s.taken_at.day,
            hour=s.taken_at.hour,
        )


class Migration(migrations.Migration):

    dependencies = [
        ("devices", "0037_motion_burst_and_device_delete"),
        ("videos", "0010_photos_in_videos"),
    ]

    operations = [
        migrations.RunPython(stills_to_photos, migrations.RunPython.noop),
        migrations.DeleteModel(name="DeviceStill"),
    ]
