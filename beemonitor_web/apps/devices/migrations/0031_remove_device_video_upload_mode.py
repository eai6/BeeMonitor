from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0030_remove_bee_confirmation'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='device',
            name='video_upload_mode',
        ),
    ]
