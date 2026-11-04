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

import json
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


# ── Apply the pipeline's grouping to a batch that already ran ──────────────

# Recounted while the launcher waits; the rest go to the recount sweep.
APPLY_NOW = 8
ANALYZERS = ("analyze.events", "analyze.interactions", "analyze.detection_count",
             "analyze.foraging_trips", "analyze.visitation", "analyze.interaction")


def pipeline_group_step(pipeline):
    """The pipeline's Identify Species step, or None."""
    for step in (pipeline.steps or []) if pipeline else []:
        if step.get("block_type") == "identify.species":
            return step
    return None


def _feeds_analyzers(pipeline, sid):
    """Whether the pipeline wires its Identify Species node into an analyzer's
    tracks (in line), rather than off to the side."""
    return any(s.get("block_type") in ANALYZERS and (s.get("inputs") or {}).get("tracks") == sid
               for s in pipeline.steps or [])


def applied_steps(run_steps, pipeline):
    """``(steps, grouped)``: the run's steps with the pipeline's current
    settings for the steps the web counts with — every analyzer (Events,
    Interactions…, matched by step id) and Identify Species with its wiring.
    ``grouped`` is False when the pipeline groups tracks but the run has no
    Identify Species node: its tracking classified no crops, so there is
    nothing to group without the GPU. None when nothing applies at all.

    The run keeps its own species model: its crops were read with it.
    Analyzers that took the run's tracks straight from MOT take them from
    Identify Species when the pipeline wires it in line, and from MOT again
    when it doesn't."""
    from .retrack.run import GPU_MOT_KEY, REPLAYABLE

    steps = [dict(s) for s in run_steps or []]
    by_id = {s.get("id"): s for s in (pipeline.steps or []) if pipeline}
    changed = False
    for s in steps:
        src_step = by_id.get(s.get("id"))
        if src_step is None or src_step.get("block_type") != s.get("block_type"):
            continue
        if s.get("block_type") in ANALYZERS:
            s["config"] = dict(src_step.get("config") or {})
            changed = True
        elif s.get("block_type") == "track.mot":
            # BeeTrack's time windows are re-tracked on the web (memory/69);
            # the rest of the node needs the GPU, so it stays as tracked.
            mine, theirs = dict(s.get("config") or {}), src_step.get("config") or {}
            if (mine.get("tracker") or "beetrack") != "beetrack" or \
                    (theirs.get("tracker") or "beetrack") != "beetrack":
                continue
            cfg = {**mine, **{k: theirs[k] for k in REPLAYABLE if k in theirs}}
            if {k: cfg.get(k) for k in REPLAYABLE} != {k: mine.get(k) for k in REPLAYABLE}:
                cfg.setdefault(GPU_MOT_KEY, {k: v for k, v in mine.items() if k != GPU_MOT_KEY})
                s["config"] = cfg
                changed = True

    src = pipeline_group_step(pipeline)
    mine = next((s for s in steps if s.get("block_type") == "identify.species"), None)
    if src is None or mine is None:
        return (steps, src is None) if changed else None
    model = (mine.get("config") or {}).get("model")
    mine["config"] = {**(src.get("config") or {}), **({"model": model} if model else {})}
    upstream = (mine.get("inputs") or {}).get("tracks")
    in_line = _feeds_analyzers(pipeline, src.get("id"))
    for s in steps:
        if s.get("block_type") not in ANALYZERS or not s.get("inputs"):
            continue
        tracks = s["inputs"].get("tracks")
        if in_line and tracks == upstream:
            s["inputs"] = {**s["inputs"], "tracks": mine["id"]}
        elif not in_line and tracks == mine["id"] and upstream:
            s["inputs"] = {**s["inputs"], "tracks": upstream}
    return steps, True


def apply_pipeline(batch_id, user, now=APPLY_NOW, only=None) -> dict:
    """Give every finished run of the batch its pipeline's current counting
    settings (analyzers, Identify Species and its wiring), then recount them:
    the first ``now`` here, the rest by the sweep within a reconcile pass. No
    GPU: everything is counted from the tracks and crop readings the runs
    already saved. ``{applied, recounted, queued, skipped}``; ``skipped`` are
    runs whose groups couldn't be applied (tracked without Identify Species).
    ``only``: run ids to update (memory/72, e.g. the clips reviewed); the
    others keep their counts and stay behind the pipeline."""
    from . import recount
    from .models import PipelineRun
    from .retrack import run as retrack

    from . import runsummary

    qs = PipelineRun.objects.filter(batch_id=batch_id, user=user,
                                    status=PipelineRun.Status.COMPLETED)
    if only is not None:
        qs = qs.filter(pk__in=list(only))
    # Slim copies (memory/76): only the clips recounted right here need their
    # whole results, and those are read one at a time.
    runs = runsummary.slim_runs(qs.select_related("pipeline").order_by("started_at"))
    out = {"applied": 0, "recounted": 0, "queued": 0, "skipped": 0}
    for i, run in enumerate(runs):
        applied = applied_steps(run.steps, run.pipeline)
        if applied is None:
            out["skipped"] += 1
            continue
        steps, grouped = applied
        if not grouped:
            out["skipped"] += 1
        PipelineRun.objects.filter(pk=run.pk).update(steps=steps)
        out["applied"] += 1
        run.steps = steps
        if retrack.needs_retrack(run):
            # Re-tracking takes seconds a clip: the sweep does it, in the
            # background, a few clips per pass (memory/69).
            PipelineRun.objects.filter(pk=run.pk).update(counting_rules=0)
            out["queued"] += 1
            out["retrack"] = out.get("retrack", 0) + 1
            continue
        if out["recounted"] < now:
            try:
                recount.recount_run(PipelineRun.objects.get(pk=run.pk))
                PipelineRun.objects.filter(pk=run.pk).update(counting_rules=recount.CURRENT_RULES)
                out["recounted"] += 1
                continue
            except Exception:
                logger.exception("apply groups: recount of run %s failed", run.pk)
        PipelineRun.objects.filter(pk=run.pk).update(counting_rules=0)
        out["queued"] += 1
    return out


def split_by_change(runs, pipeline):
    """``(web, gpu)``: the runs a recount brings up to the pipeline, and the
    ones whose changes need the GPU (a new detector, a new threshold...)."""
    web, gpu = [], []
    for run in runs:
        (gpu if pipeline_changes(run, pipeline)["gpu"] else web).append(run)
    return web, gpu


def _window_of(steps):
    """(start, end or None) of the clip part a run analysed, or None."""
    for step in steps or []:
        if step.get("block_type") == "input.video":
            cfg = step.get("config") or {}
            start, end = cfg.get("start_seconds"), cfg.get("end_seconds")
            if start is not None or end is not None:
                return (float(start or 0), float(end) if end is not None else None)
    return None


def rerun_on_gpu(runs, pipeline, user) -> int:
    """Run these finished clips again with the pipeline as it is now, in place
    (memory/76): same run, same batch, same part of the clip; the GPU steps
    run again and everything below them. Their review votes follow to the new
    events when each finishes (engine._move_rerun_votes). Raises
    ``accounts.quota.QuotaExceeded`` before starting any when they don't fit.
    Returns how many started."""
    from apps.accounts import quota
    from apps.videos.models import Video

    from . import engine
    from .aggregate import run_video_id

    plans = []
    videos = Video.everything.in_bulk([v for r in runs if (v := run_video_id(r))])
    for run in runs:
        video = videos.get(run_video_id(run))
        if video is None:
            continue
        steps = engine.steps_with_video_steps(pipeline.steps, video.pk, _window_of(run.steps))
        if engine.validate_steps(steps):
            continue
        plans.append((run, steps, video))
    quota.check(user, [w for _r, steps, video in plans
                       for w in quota.pipeline_work(steps, [video])])
    for run, steps, _video in plans:
        type(run).objects.filter(pk=run.pk).update(rerun_votes=True)
        full = type(run).objects.get(pk=run.pk)       # not a slim copy: it is saved
        engine.start_run(full, steps=steps, checked=True)
    if plans:
        try:
            from apps.analysis.views import _drain_queue
            _drain_queue()
        except Exception:  # noqa: BLE001 - the reconciler drains it anyway
            logger.exception("rerun: could not kick the GPU queue")
    return len(plans)


# ── What changed in the pipeline since a batch was counted ─────────────────

def _label(block_type, field_name, value):
    """A setting's value as the editor shows it (a choice's label)."""
    from .registry import get_block

    for f in (get_block(block_type) or {}).get("config_fields") or []:
        if f["name"] == field_name:
            for c in f.get("choices") or []:
                if str(c["value"]) == str(value):
                    return c["label"]
            break
    if value in (None, ""):
        return "default"
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "…"


def _field_label(block_type, field_name):
    from .registry import get_block

    for f in (get_block(block_type) or {}).get("config_fields") or []:
        if f["name"] == field_name:
            return f.get("label") or field_name
    return field_name


def _effective(block_type, config):
    """A step's settings with the registry's defaults filled in, so a field
    saved blank and one never saved compare equal."""
    from .registry import get_block

    out = {}
    for f in (get_block(block_type) or {}).get("config_fields") or []:
        v = (config or {}).get(f["name"])
        out[f["name"]] = f.get("default") if v in (None, "") else v
    return out


# Settings the web counts with; everything else on a GPU step changes tracking.
_GPU_FIELDS = {"identify.species": {"model"}}


def pipeline_changes(run, pipeline) -> dict:
    """``{"web": [...], "gpu": [...]}``: what differs between the steps this
    run was counted with and the pipeline's now. ``web`` changes a recount
    applies (no GPU); ``gpu`` ones need tracking again. Each item is
    ``{step, field, before, after}``."""
    from .registry import get_block
    from .retrack.run import REPLAYABLE

    web, gpu = [], []
    mine = {s.get("id"): s for s in run.steps or []}
    for src in pipeline.steps or []:
        bt = src.get("block_type", "")
        block = get_block(bt) or {}
        step = mine.get(src.get("id"))
        name = block.get("display_name") or bt
        # The Trips node's rules are read live, across clips: nothing to update.
        if bt.startswith("input.") or bt == "analyze.trips":
            continue
        if step is None or step.get("block_type") != bt:
            (gpu if block.get("backend") == "gpu" else web).append(
                {"step": name, "field": "", "before": "", "after": "added"})
            continue
        before, after = _effective(bt, step.get("config")), _effective(bt, src.get("config"))
        for key in after:
            if key in ("video_id", "start_seconds", "end_seconds"):
                continue
            if json.dumps(before.get(key), sort_keys=True, default=str) == \
                    json.dumps(after.get(key), sort_keys=True, default=str):
                continue
            if key == "leave_out":
                from . import taxon_groups
                b = sorted(taxon_groups.settings(step.get("config"))["leave_out"])
                a = sorted(taxon_groups.settings(src.get("config"))["leave_out"])
                if a == b:
                    continue
                web.append({"step": name, "field": _field_label(bt, key),
                            "before": ", ".join(b) or "none", "after": ", ".join(a) or "none"})
                continue
            if key == "grouping":
                from . import taxon_groups
                if taxon_groups.grouping_of(step.get("config")) == taxon_groups.grouping_of(src.get("config")):
                    continue
                item = {"step": name, "field": "Grouping", "before": "", "after": "edited"}
            else:
                item = {"step": name, "field": _field_label(bt, key),
                        "before": _label(bt, key, before.get(key)),
                        "after": _label(bt, key, after.get(key))}
            gpu_side = (block.get("backend") == "gpu" or key in _GPU_FIELDS.get(bt, set())
                        # MOT: BeeTrack's time windows re-track on the web
                        # (memory/69); its other settings need the GPU.
                        or (bt == "track.mot" and not (
                            key in REPLAYABLE
                            and (after.get("tracker") or "beetrack") == "beetrack"
                            and (before.get("tracker") or "beetrack") == "beetrack")))
            if bt == "track.mot" and not gpu_side:
                item["after"] += " (tracks again on the web)"
            (gpu if gpu_side else web).append(item)
        if (step.get("inputs") or {}) != (src.get("inputs") or {}) and block.get("backend") != "gpu":
            web.append({"step": name, "field": "Wiring", "before": "", "after": "changed"})
        if bt == "track.mot" and not any(c["step"] == name for c in web + gpu):
            # Same settings, newer lost-track rules (memory/72).
            from .retrack.run import needs_retrack
            if needs_retrack(run):
                web.append({"step": name, "field": "Lost-track rules", "before": "",
                            "after": "updated (tracks again on the web)"})
    return {"web": web, "gpu": gpu}


def batch_changes(runs, pipeline) -> dict | None:
    """The batch page's "pipeline changed" card: the changes (from the first
    clip that differs) and how many clips are behind. None when every clip
    was counted with the pipeline as it is."""
    first, behind = None, 0
    for run in runs:
        if run.status != run.Status.COMPLETED:
            continue
        changes = pipeline_changes(run, pipeline)
        if changes["web"] or changes["gpu"]:
            behind += 1
            first = first or changes
    if not behind:
        return None
    return {**first, "behind": behind}
