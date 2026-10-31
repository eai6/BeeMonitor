"""Change which group a track, species, genus or family counts as, from the
review (memory/65).

The grouping is the pipeline's: a change is written into the Identify Species
node of every run in the batch and of the pipeline itself (when it's yours), so
the next clip added or run is grouped the same way. The clip being reviewed is
recounted at once; the batch's other runs are handed to the recount sweep,
which picks them up within one reconcile pass. No GPU: groups are worked out
from the crop readings already saved.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

SCOPES = ("track", "name", "genus", "family")


def group_step(run):
    """``(index, step)`` of the run's Identify Species node that grouped its
    tracks, or ``(None, None)``."""
    context = run.context or {}
    for i, step in enumerate(run.steps or []):
        if (step.get("block_type") == "identify.species"
                and "track_groups" in (context.get(step.get("id")) or {})):
            return i, step
    return None, None


def can_regroup(user, run) -> bool:
    """Only the run's launcher: the grouping is part of their pipeline."""
    return bool(user and user.is_authenticated
                and (user.is_superuser or run.user_id == user.pk))


def _rule(run, step, scope, track, group):
    from . import aggregate

    info = ((run.context or {}).get(step["id"]) or {}).get("track_groups", {}).get(str(track))
    if scope == "track":
        if info is None:
            raise ValueError(f"Track {track} has no group in this run.")
        return {"kind": "track", "job": aggregate.run_job_id(run), "track": int(track),
                "group": group}
    if info is None or not info.get("top"):
        raise ValueError(f"Track {track} has no species to make a rule from.")
    if scope == "name":
        value = info["top"]
    else:
        from . import taxon_groups

        value = (taxon_groups.lineages([info["top"]]).get(info["top"]) or {}).get(scope)
        if not value:
            raise ValueError(f"No {scope} is known for {info['top']}.")
    return {"kind": scope, "value": value, "group": group}


def set_group(run, user, scope, track, group) -> dict:
    """Write the rule, recount this run now and queue the batch's others.
    ``group`` empty takes the rule away. Returns ``{rule, track, queued}``."""
    from . import recount
    from .models import Pipeline, PipelineRun

    if scope not in SCOPES:
        raise ValueError("Choose this track, the species, its genus or its family.")
    group = " ".join(str(group or "").split())[:60]
    _i, step = group_step(run)
    if step is None:
        raise ValueError("This run has no Identify Species node grouping its tracks.")
    rule = _rule(run, step, scope, track, group)
    sid = step["id"]

    from . import taxon_groups

    runs = [run]
    if run.batch_id:
        runs = list(PipelineRun.objects.filter(batch_id=run.batch_id, user_id=run.user_id))
    for r in runs:
        steps = [dict(s) for s in r.steps or []]
        hit = False
        for s in steps:
            if s.get("id") == sid and s.get("block_type") == "identify.species":
                s["config"] = taxon_groups.with_rule(s.get("config"), rule)
                hit = True
        if hit:
            PipelineRun.objects.filter(pk=r.pk).update(steps=steps)
    pipeline = Pipeline.objects.filter(pk=run.pipeline_id, user=user).first()
    if pipeline is not None:
        steps = [dict(s) for s in pipeline.steps or []]
        for s in steps:
            if s.get("id") == sid and s.get("block_type") == "identify.species":
                s["config"] = taxon_groups.with_rule(s.get("config"), rule)
                pipeline.steps = steps
                pipeline.save(update_fields=["steps", "updated_at"])
                break

    run.refresh_from_db()
    recount.recount_run(run)
    run.refresh_from_db()
    queued = 0
    if scope != "track":
        others = [r.pk for r in runs if r.pk != run.pk
                  and r.status == PipelineRun.Status.COMPLETED]
        queued = PipelineRun.objects.filter(pk__in=others).update(counting_rules=0)
    info = ((run.context or {}).get(sid) or {}).get("track_groups", {}).get(str(track)) or {}
    return {"rule": rule, "track": info, "queued": queued}
