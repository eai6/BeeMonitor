from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("videos", "0005_backfill_video_device"),
    ]

    operations = [
        migrations.AddField(
            model_name="video",
            name="device_delete_requested",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="video",
            name="device_deleted_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
