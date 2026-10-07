"""Saved pipelines' frame-count settings move to seconds.

Every duration a pipeline sets is now in seconds, turned into frames with each
clip's own fps when it runs, so one setting means the same time on any camera.
Saved values are converted at 25 fps — the units' recording rate — so a
pipeline behaves exactly as before on their clips. Runs already launched keep
their frozen steps; the analyzers still read ``gap_frames`` there.
"""

from django.db import migrations

FPS = 25.0

# old key -> (new key, block types it lives on)
RENAMES = {
    "gap_frames": ("gap_seconds", ("analyze.events", "analyze.interactions",
                                   "analyze.visitation")),
    "sample_interval": ("sample_seconds", ("detect",)),
    "byte_track_buffer": ("byte_track_buffer_seconds", ("track.mot",)),
    "ocsort_max_age": ("ocsort_max_age_seconds", ("track.mot",)),
    "ocsort_min_hits": ("ocsort_min_hits_seconds", ("track.mot",)),
    "ocsort_delta_t": ("ocsort_delta_t_seconds", ("track.mot",)),
}


def _seconds(value):
    try:
        return str(round(float(value) / FPS, 3)).rstrip("0").rstrip(".") or "0"
    except (TypeError, ValueError):
        return None


def forwards(apps, schema_editor):
    Pipeline = apps.get_model("pipelines", "Pipeline")
    for pipeline in Pipeline.objects.all().iterator():
        changed = False
        for step in pipeline.steps or []:
            block = str(step.get("block_type") or "")
            config = step.get("config") or {}
            for old, (new, blocks) in RENAMES.items():
                if old not in config or not any(block.startswith(b) for b in blocks):
                    continue
                value = config.pop(old)
                if new not in config and value not in (None, ""):
                    seconds = _seconds(value)
                    if seconds is not None:
                        config[new] = seconds
                changed = True
            step["config"] = config
        if changed:
            pipeline.save(update_fields=["steps"])


class Migration(migrations.Migration):
    dependencies = [("pipelines", "0006_beetrack_longer_track_memory")]
    operations = [migrations.RunPython(forwards, migrations.RunPython.noop)]
