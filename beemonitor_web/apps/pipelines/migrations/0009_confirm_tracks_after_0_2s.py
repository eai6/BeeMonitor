"""Saved BeeTrack nodes on the old confirmation default (0) move to 0.2 s.

The editor stores every field, so a node never touched holds "0" and the new
default would not reach it. Any other value was a choice and is kept.
"""

from django.db import migrations


def forwards(apps, schema_editor):
    Pipeline = apps.get_model("pipelines", "Pipeline")
    for pipeline in Pipeline.objects.all().iterator():
        changed = False
        for step in pipeline.steps or []:
            if step.get("block_type") != "track.mot":
                continue
            config = step.get("config") or {}
            if (config.get("tracker") or "beetrack").lower() != "beetrack":
                continue
            value = config.get("beetrack_min_hits_seconds")
            try:
                is_zero = value is not None and value != "" and float(value) == 0.0
            except (TypeError, ValueError):
                is_zero = False
            if is_zero:
                config["beetrack_min_hits_seconds"] = "0.2"
                step["config"] = config
                changed = True
        if changed:
            pipeline.save(update_fields=["steps"])


class Migration(migrations.Migration):
    dependencies = [("pipelines", "0008_one_lost_track_setting")]
    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
