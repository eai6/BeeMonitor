"""Each run's results slimmed for pages that list many clips (memory/76).

Only the column: filling it reads every run's whole results, which is too slow
to sit in front of the server starting. ``manage.py fill_run_summaries`` fills
it in the background after start-up, and a page that meets a run without one
builds it then (runsummary.use_slim)."""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [("pipelines", "0022_run_gpu_job_id")]

    operations = [
        migrations.AddField(model_name="pipelinerun", name="summary",
                            field=models.JSONField(blank=True, default=dict)),
        migrations.AddField(model_name="pipelinerun", name="rerun_votes",
                            field=models.BooleanField(default=False)),
    ]
