from django.db import migrations, models


def bool_to_mode(apps, schema_editor):
    """Carry the old on/off crop toggle into the new 3-way mode: on -> confirmed
    (today's gate-mode behavior), off -> off."""
    Device = apps.get_model('devices', 'Device')
    Device.objects.filter(send_activity_crops=True).update(activity_crops_mode='confirmed')
    Device.objects.filter(send_activity_crops=False).update(activity_crops_mode='off')


def mode_to_bool(apps, schema_editor):
    Device = apps.get_model('devices', 'Device')
    Device.objects.exclude(activity_crops_mode='off').update(send_activity_crops=True)
    Device.objects.filter(activity_crops_mode='off').update(send_activity_crops=False)


class Migration(migrations.Migration):

    dependencies = [
        ('devices', '0020_device_send_activity_crops'),
    ]

    operations = [
        migrations.AddField(
            model_name='device',
            name='activity_crops_mode',
            field=models.CharField(
                choices=[
                    ('all', 'All activity — send every crop'),
                    ('confirmed', 'Confirmed only — send confirmed-bee crops'),
                    ('off', "Off — don't send crops"),
                ],
                default='confirmed', max_length=10),
        ),
        migrations.RunPython(bool_to_mode, mode_to_bool),
        migrations.RemoveField(
            model_name='device',
            name='send_activity_crops',
        ),
    ]
