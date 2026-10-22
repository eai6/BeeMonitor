from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("pipelines", "0010_batch_share"),
    ]

    operations = [
        # Existing runs were counted with the old rules (1): the reconcile loop
        # recounts them. New runs are counted with the current ones.
        migrations.AddField(
            model_name="pipelinerun",
            name="counting_rules",
            field=models.PositiveSmallIntegerField(default=1),
        ),
        migrations.AlterField(
            model_name="pipelinerun",
            name="counting_rules",
            field=models.PositiveSmallIntegerField(default=2),
        ),
    ]
