"""How close the web replay is to the GPU's own tracking (memory/69).

A clip is re-tracked with the settings the GPU used, and the result is
compared with the GPU's tracks: how many of its detected boxes the replay
tracked, whether each GPU track stayed one track, and how many events come
out the same. A clip tracked with the worker's frame log should agree
exactly; an older one closely.
"""

from __future__ import annotations

import os
import tempfile


def gpu_settings(run):
    """The BeeTracker settings the GPU tracked this run's clip with: the ones
    recorded when "Update this batch" changed its MOT node, else the node's
    own (it hasn't changed since the GPU ran)."""
    from .. import executors
    from .replay import tracker_params
    from .run import GPU_MOT_KEY, mot_step

    cfg = (mot_step(run) or {}).get("config") or {}
    if isinstance(cfg.get(GPU_MOT_KEY), dict):
        cfg = cfg[GPU_MOT_KEY]
    return tracker_params(executors.tracker_settings(cfg))


def _events(run, result):
    """The run's Events step over ``result``'s tracks, ungrouped."""
    from .. import executors
    from ..review import event_key

    steps = run.steps or []
    for i, step in enumerate(steps):
        if step.get("block_type") != "analyze.events":
            continue
        inputs = executors.resolve_inputs(step, steps, i, run.context or {})
        tracks = dict(inputs.get("tracks") or inputs.get("in") or {})
        tracks.pop("dropped_tracks", None)
        tracks["result"] = result
        inputs = {**inputs, "tracks": tracks}
        out = executors.LOCAL_EXECUTORS["analyze.events"](step, run, run.context or {}, inputs, i)
        return [(r.get("action"), str(r.get("reference")), int(r.get("frame") or 0), event_key(r))
                for r in out.get("rows") or []]
    return []


def compare(run) -> dict:
    """``{boxes, same_track, gpu_tracks, tracks, events, events_same, exact}``
    for one run's clip."""
    import csv

    from .replay import TRACKING_COLUMNS
    from .run import _boxes, _job, _rows, crop_mapping, gpu_result, replay_clip

    base = gpu_result(run)
    job = _job(run)
    rows, exact = replay_clip(run, job, base, gpu_settings(run))
    gpu_rows = _rows(base.get("tracking_csv_path")) or []
    mapping = crop_mapping(gpu_rows, rows)
    # Each detected box once: the trigger frame of a lookback is in the
    # table twice (fed to the tracker twice).
    detected = _boxes(gpu_rows)

    per_track = {}
    for (old, _frame), new in mapping.items():
        per_track.setdefault(old, {}).setdefault(new, 0)
        per_track[old][new] += 1
    kept = sum(max(c.values()) for c in per_track.values())

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRACKING_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        path = fh.name
    try:
        gpu_events = _events(run, base)
        new_events = _events(run, {**base, "tracking_csv_path": path})
    finally:
        os.unlink(path)
    fps = float(((base.get("summary_stats") or {}).get("video_fps")) or 25)
    tol = max(1, round(fps * 0.5))
    pool = list(new_events)
    same = 0
    for action, ref, frame, _key in gpu_events:
        hit = next((e for e in pool if e[0] == action and e[1] == ref
                    and abs(e[2] - frame) <= tol), None)
        if hit:
            pool.remove(hit)
            same += 1
    return {
        "boxes": round(len(mapping) / len(detected), 3) if detected else None,
        "same_track": round(kept / len(mapping), 3) if mapping else None,
        "gpu_tracks": len({r.get("track_id") for r in gpu_rows}),
        "tracks": len({r["track_id"] for r in rows}),
        "events": len(gpu_events),
        "events_same": same,
        "events_new": len(new_events),
        "exact": exact,
    }


def batch_agreement(runs, clips=2, budget=60) -> dict | None:
    """Compare up to ``clips`` clips of a batch; the totals, or None when no
    clip could be replayed."""
    import time

    from .run import NotReplayable, gpu_result

    deadline = time.monotonic() + budget
    totals = {"clips": 0, "boxes": 0.0, "same_track": 0.0, "events": 0, "events_same": 0,
              "gpu_tracks": 0, "tracks": 0, "exact": True}
    for run in runs:
        if totals["clips"] >= clips or time.monotonic() > deadline:
            break
        base = gpu_result(run)
        if run.status != run.Status.COMPLETED or not base or not base.get("detections_csv_path"):
            continue
        try:
            c = compare(run)
        except NotReplayable:
            continue
        totals["clips"] += 1
        totals["boxes"] += c["boxes"] or 0
        totals["same_track"] += c["same_track"] or 0
        for k in ("events", "events_same", "gpu_tracks", "tracks"):
            totals[k] += c[k]
        totals["exact"] = totals["exact"] and c["exact"]
    if not totals["clips"]:
        return None
    n = totals["clips"]
    return {**totals, "boxes": round(100 * totals["boxes"] / n),
            "same_track": round(100 * totals["same_track"] / n),
            "events_pct": (round(100 * totals["events_same"] / totals["events"])
                           if totals["events"] else None)}
