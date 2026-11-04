"""A run's results, slimmed for pages that list many clips (memory/76).

``PipelineRun.context`` holds every step's whole output: each event and
interaction row, the GPU result's summary, every track's species and group.
A batch page that loaded it for 400 clips used ~440 MB and got the web
server's workers killed. ``PipelineRun.summary`` is the same shape with the
heavy parts taken out, kept beside it on every save, so the code that reads a
run reads either one:

* event rows keep only what identifies and scores an event;
* interaction rows, per-track species rows and ``names_seen`` go (their
  counts and per-reference totals stay);
* ``track_groups`` and the GPU result's ``identification.by_track`` become a
  marker that says they exist;
* any other value in the GPU result's ``summary_stats`` bigger than
  ``STAT_LIMIT`` bytes goes.

A run whose ``context`` is the slim copy (``use_slim``) refuses to be saved,
so a page can never write it back over the real results.
"""

from __future__ import annotations

import json

# Bump when what is kept changes: the reconcile loop refills older summaries.
VERSION = 1

STAT_LIMIT = 2048
# Kept whatever their size: a photo's own detections are what a photo batch
# lists, and the frame rate times everything.
KEEP_STATS = ("photo", "video_fps", "fps")
# What an event row keeps: enough for its key (review.event_key), its score
# and its place in time.
EVENT_FIELDS = ("frame", "track_id", "action", "reference", "at_edge", "end_frame",
                "duration_sec")
# Stands in for a per-track mapping that was left out: truthy, so "has it?"
# still answers yes, and empty of tracks.
PRESENT = {"_slim": True}


def _event(row):
    from .primitives import normalize_event

    row = normalize_event(row)
    return {k: row.get(k) for k in EVENT_FIELDS
            if row.get(k) not in (None, "") or k in ("track_id", "reference", "action", "frame")}


def _small(value):
    try:
        return len(json.dumps(value, default=str)) <= STAT_LIMIT
    except (TypeError, ValueError):
        return False


def _result(result):
    if not isinstance(result, dict):
        return result
    out = dict(result)
    stats = result.get("summary_stats")
    if isinstance(stats, dict):
        slim = {k: v for k, v in stats.items()
                if k != "identification" and (k in KEEP_STATS or _small(v))}
        ident = stats.get("identification")
        if isinstance(ident, dict):
            slim["identification"] = {k: v for k, v in ident.items()
                                      if k != "by_track" and _small(v)}
            if ident.get("by_track"):
                slim["identification"]["by_track"] = PRESENT
        out["summary_stats"] = slim
    return out


def _output(value):
    if not isinstance(value, dict):
        return value
    out = dict(value)
    kind = value.get("table_kind")
    rows = value.get("rows")
    if isinstance(rows, list):
        if kind == "events":
            out["rows"] = [_event(r) for r in rows]
        else:
            out["rows"] = []
            count_key = {"interactions": "interaction_count"}.get(kind)
            if count_key and out.get(count_key) is None:
                out[count_key] = len(rows)
    if value.get("track_groups"):
        out["track_groups"] = PRESENT
    out.pop("names_seen", None)
    if "result" in value:
        out["result"] = _result(value["result"])
    return out


def build(context) -> dict:
    """The slim copy of a run's ``context``."""
    return {"v": VERSION,
            "context": {sid: _output(v) for sid, v in (context or {}).items()}}


def is_current(summary) -> bool:
    return isinstance(summary, dict) and summary.get("v") == VERSION


def use_slim(runs):
    """Point each run's ``context`` at its slim copy (``runs`` loaded with
    ``.defer("context")``). A run without a current summary gets one now,
    from its real context, saved for next time. Returns ``runs``."""
    from .models import PipelineRun

    stale = {r.pk: r for r in runs if not is_current(r.summary)}
    if stale:
        # One at a time: the point is never to hold many whole contexts.
        for pk, context in (PipelineRun.objects.filter(pk__in=list(stale))
                            .values_list("pk", "context").iterator(chunk_size=20)):
            stale[pk].summary = build(context)
            PipelineRun.objects.filter(pk=pk).update(summary=stale[pk].summary)
    for r in runs:
        r.context = r.summary["context"]
        r._slim = True
    return runs


def share_steps(qs):
    """``qs`` (PipelineRuns) as a list whose runs share what they have in
    common. A batch's runs hold the same steps but for the clip they are bound
    to, each a separate copy, as is the pipeline a join brings with every row:
    tens of KB per clip, for a read-only page, and most of what a batch of
    thousands costs (memory/77)."""
    seen, pipelines, runs = {}, {}, []
    for run in qs.iterator(chunk_size=200):
        run.steps = [seen.setdefault(json.dumps(s, sort_keys=True, default=str), s)
                     for s in run.steps or []]
        if run.pipeline_id in pipelines:
            run.pipeline = pipelines[run.pipeline_id]
        else:
            pipelines[run.pipeline_id] = run.pipeline
        runs.append(run)
    return runs


def slim_runs(qs):
    """``qs`` (PipelineRuns) as a list of runs reading their slim copy."""
    return use_slim(list(qs.defer("context")))
