"""BeeTrack's two lost-track settings become one.

"Keep a lost track" and "Revive a dead track within" did the same job back to
back; the revive stage is retired and the one setting covers the whole time,
so a saved node keeps the total it had: keep + revive (2.0 + 1.0 -> 3.0).
"""

from django.db import migrations


def _num(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def forwards(apps, schema_editor):
    Pipeline = apps.get_model("pipelines", "Pipeline")
    for pipeline in Pipeline.objects.all().iterator():
        changed = False
        for step in pipeline.steps or []:
            if step.get("block_type") != "track.mot":
                continue
            config = step.get("config") or {}
            if "beetrack_max_resurrection_seconds" not in config:
                continue
            revive = _num(config.pop("beetrack_max_resurrection_seconds")) or 0.0
            keep = _num(config.get("beetrack_max_age_seconds"))
            if keep is None:
                keep = 2.0
            total = round(keep + revive, 3)
            config["beetrack_max_age_seconds"] = str(int(total) if total.is_integer() else total)
            step["config"] = config
            changed = True
        if changed:
            pipeline.save(update_fields=["steps"])


class Migration(migrations.Migration):
    dependencies = [("pipelines", "0007_durations_in_seconds")]
    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
