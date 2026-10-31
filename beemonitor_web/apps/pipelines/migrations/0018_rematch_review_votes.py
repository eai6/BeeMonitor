"""Re-attach review votes to their events after recounts (memory/69).

Votes are keyed by their event's ``action|reference|track|frame``. Until now
they were only moved after a re-track, so a recount that shifted events —
counting from the whole trajectory, a new layout — left them pointing at
events that no longer exist, and the batch's review metrics disappeared.
This moves each one to the event of the same track, action and reference
(or, failing that, the one of that action and reference within 0.5 s).
"""

from django.db import migrations


def rematch(apps, schema_editor):
    from django.db.models import Q

    from apps.pipelines.models import PipelineRun
    from apps.pipelines.retrack import votes
    from apps.pipelines.retrack.run import gpu_result, tracking_outputs

    runs = (PipelineRun.objects.filter(Q(event_reviews__isnull=False) | Q(trip_exits__isnull=False))
            .distinct())
    for run in runs.iterator():
        retracked = any((run.context[s].get("result") or {}).get("retracked")
                        for s in tracking_outputs(run))
        fps = ((gpu_result(run) or {}).get("summary_stats") or {}).get("video_fps")
        try:
            # Votes cast before a re-track were on the GPU's tracks: match by time then.
            votes.move_votes(run, fps, same_tracks=not retracked)
        except Exception:  # noqa: BLE001 - one odd run must not block the deploy
            pass


class Migration(migrations.Migration):

    dependencies = [("pipelines", "0017_batch_name")]

    operations = [migrations.RunPython(rematch, migrations.RunPython.noop)]
