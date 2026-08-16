from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0027_devicepipelineschedule'),
    ]

    operations = [
        migrations.AddField(
            model_name='device',
            name='roi_polygon',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
