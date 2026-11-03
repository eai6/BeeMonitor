"""Each run's GPU job as its own indexed column (memory/75), filled from the
job id its context records (aggregate.run_job_id)."""

from django.db import migrations, models


def fill(apps, schema_editor):
    PipelineRun = apps.get_model("pipelines", "PipelineRun")
    batch = []
    for run in PipelineRun.objects.only("pk", "context").iterator(chunk_size=200):
        job_id = next((out.get("job_id") for out in (run.context or {}).values()
                       if isinstance(out, dict) and out.get("job_id")), None)
        if job_id:
            run.gpu_job_id = job_id
            batch.append(run)
        if len(batch) >= 500:
            PipelineRun.objects.bulk_update(batch, ["gpu_job_id"])
            batch = []
    if batch:
        PipelineRun.objects.bulk_update(batch, ["gpu_job_id"])


class Migration(migrations.Migration):

    dependencies = [("pipelines", "0021_run_removed_from_batch")]

    operations = [
        migrations.AddField(
            model_name="pipelinerun", name="gpu_job_id",
            field=models.IntegerField(blank=True, db_index=True, null=True)),
        migrations.RunPython(fill, migrations.RunPython.noop),
    ]
