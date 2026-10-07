"""Saved BeeTrack MOT nodes on the old timing defaults move to the new ones.

The editor stores every field, defaults included, so a node never touched still
holds "0.5" / "0.3" and a new registry default would not reach it. A value other
than the old default was a choice and is left alone.
"""

from django.db import migrations

OLD_TO_NEW = {
    "beetrack_max_age_seconds": (0.5, "2.0"),
    "beetrack_max_resurrection_seconds": (0.3, "1.0"),
}


def _is(value, number):
    try:
        return float(value) == number
    except (TypeError, ValueError):
        return False


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
            for name, (old, new) in OLD_TO_NEW.items():
                if name in config and _is(config[name], old):
                    config[name] = new
                    changed = True
            step["config"] = config
        if changed:
            pipeline.save(update_fields=["steps"])


class Migration(migrations.Migration):
    dependencies = [("pipelines", "0005_pipelinerun_fresh")]
    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
