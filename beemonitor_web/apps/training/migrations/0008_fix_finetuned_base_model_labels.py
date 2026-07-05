"""Relabel base_model on custom models produced by fine-tune jobs.

Fine-tune training jobs stored a placeholder architecture ("yolov8n") on
their resulting CustomModel even though the weights inherit the fine-tune
source's architecture (e.g. the production bee_tracking.pt is a YOLO26n).
Show the lineage instead.
"""

from django.db import migrations


def fix_labels(apps, schema_editor):
    CustomModel = apps.get_model("training", "CustomModel")
    for cm in CustomModel.objects.exclude(training_job=None).select_related("training_job"):
        job = cm.training_job
        if job.init_weights_key and job.init_from_label:
            cm.base_model = f"fine-tuned: {job.init_from_label}"[:50]
            cm.save(update_fields=["base_model"])


class Migration(migrations.Migration):

    dependencies = [
        ("training", "0007_trainingjob_val_percent"),
    ]

    operations = [
        migrations.RunPython(fix_labels, migrations.RunPython.noop),
    ]
