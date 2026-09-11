"""Remove the on-device YOLO bee-confirmation feature.

The recorder no longer runs YOLO over a clip's frames to confirm/unconfirm it —
motion-triggered clips are simply recorded and uploaded. This migration:

  1. drops ``Device.bee_confirm_mode`` (the heartbeat no longer pushes it);
  2. collapses ``activity_crops_mode`` "confirmed" -> "all" (that choice only
     existed to mean "crops for confirmed bees only"; with no confirmer, nothing
     is ever rejected, so it already behaved as "all");
  3. strips the ``bee_confirmed`` / ``bee`` keys from historical Video.metadata.

Step 3 matters: clips uploaded while the confirmer ran carry
``metadata.bee_confirmed=False`` and were hidden from the Processing page by the
(now removed) filters. Left in place, that stale flag would keep misreporting
those clips as "not a bee" forever.
"""

from django.db import migrations, models


def clear_bee_metadata(apps, schema_editor):
    """Drop bee_confirmed/bee from every Video that carries them."""
    Video = apps.get_model("videos", "Video")
    qs = Video.objects.filter(metadata__has_any_keys=["bee_confirmed", "bee"])
    batch = []
    for video in qs.iterator(chunk_size=500):
        md = video.metadata or {}
        md.pop("bee_confirmed", None)
        md.pop("bee", None)
        video.metadata = md
        batch.append(video)
        if len(batch) >= 500:
            Video.objects.bulk_update(batch, ["metadata"])
            batch = []
    if batch:
        Video.objects.bulk_update(batch, ["metadata"])


def noop_reverse(apps, schema_editor):
    """Irreversible by design — the verdicts are gone with the feature."""


def collapse_crop_mode(apps, schema_editor):
    apps.get_model("devices", "Device").objects.filter(
        activity_crops_mode="confirmed").update(activity_crops_mode="all")


def restore_crop_mode(apps, schema_editor):
    apps.get_model("devices", "Device").objects.filter(
        activity_crops_mode="all").update(activity_crops_mode="all")


class Migration(migrations.Migration):

    dependencies = [
        ("devices", "0029_device_rotate_180"),
        ("videos", "0007_pendingdevicedeletion"),
    ]

    operations = [
        migrations.RunPython(collapse_crop_mode, restore_crop_mode),
        migrations.AlterField(
            model_name="device",
            name="activity_crops_mode",
            field=models.CharField(
                choices=[("all", "All activity — send every crop"),
                         ("off", "Off — don't send crops")],
                default="all", max_length=10),
        ),
        migrations.RemoveField(model_name="device", name="bee_confirm_mode"),
        migrations.RunPython(clear_bee_metadata, noop_reverse),
    ]
