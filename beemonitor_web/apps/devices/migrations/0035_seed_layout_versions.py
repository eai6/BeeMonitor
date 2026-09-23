"""Each device's current layout becomes v1, in use from the beginning, so clips
recorded before layout history existed keep analysing with it."""

from datetime import datetime, timezone

from django.db import migrations

BEGINNING = datetime(2000, 1, 1, tzinfo=timezone.utc)


def seed(apps, schema_editor):
    Device = apps.get_model("devices", "Device")
    Version = apps.get_model("devices", "DeviceLayoutVersion")
    for d in Device.objects.all().iterator():
        if not (d.roi_override or d.roi_polygon or d.nest_layout):
            continue
        if Version.objects.filter(device=d).exists():
            continue
        Version.objects.create(device=d, number=1, roi_override=d.roi_override,
                               roi_polygon=d.roi_polygon, nest_layout=d.nest_layout or [],
                               applied_at=BEGINNING)


class Migration(migrations.Migration):
    dependencies = [
        ("devices", "0034_device_layout_version"),
    ]

    operations = [
        migrations.RunPython(seed, migrations.RunPython.noop),
    ]
