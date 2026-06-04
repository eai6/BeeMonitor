"""Backfill Video.device from the legacy metadata.device_id stamp.

Before the FK existed, Pi uploads only recorded the device in the JSON
``metadata`` blob (``apps/api/uploads.py``). Link those rows to the real Device
now that the FK is in place. Reverse is a no-op (we leave the FK populated).
"""

from django.db import migrations


def backfill_device(apps, schema_editor):
    Video = apps.get_model("videos", "Video")
    Device = apps.get_model("devices", "Device")

    valid_ids = set(Device.objects.values_list("id", flat=True))
    updated = 0
    for video in Video.objects.filter(device__isnull=True).iterator():
        dev_id = (video.metadata or {}).get("device_id")
        if dev_id in valid_ids:
            video.device_id = dev_id
            video.save(update_fields=["device"])
            updated += 1
    if updated:
        print(f"  backfilled device on {updated} video(s)")


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("videos", "0004_video_device"),
        ("devices", "0002_deviceheartbeat"),
    ]

    operations = [
        migrations.RunPython(backfill_device, noop),
    ]
