from django.db import migrations


class Migration(migrations.Migration):
    """Settings the device page no longer exposes: clip tail + max length are
    fixed fleet-wide (10s / 600s, pushed as constants), and the motion
    sensitivity override is gone so every device uses its own calibration."""

    dependencies = [
        ('devices', '0031_remove_device_video_upload_mode'),
    ]

    operations = [
        migrations.RemoveField(model_name='device', name='record_post_roll'),
        migrations.RemoveField(model_name='device', name='record_max_segment'),
        migrations.RemoveField(model_name='device', name='motion_var_threshold'),
    ]
