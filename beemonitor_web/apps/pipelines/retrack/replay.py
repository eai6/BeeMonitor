"""Replay BeeTracker over a clip's saved detections (memory/69).

The worker calls ``tracker.update(detections, frame)`` once per frame, in
order, with an empty list on frames where the detector didn't run (motion
mode) or found nothing. When motion starts, the last half second of frames is
detected and fed in again first, with their old frame numbers (the
*lookback*), and the trigger frame is then fed once more as the current frame.

A clip run with the worker's frame log (``update_log.csv``) replays those
calls exactly. An older clip only has ``detections.csv``, whose ``mode``
column marks lookback rows: the calls are rebuilt from it — every frame in
order with its current detections (empty where there are none), and each run
of lookback frames replayed just before its last frame — which is close but
not exact (the motion decisions and empty lookback frames weren't saved).
"""

from __future__ import annotations

import math

# The worker's BeeTracker settings when the pipeline sends nothing
# (core/config.py TrackingConfig; match_distance_multiplier is BeeTracking's
# own default, never configurable).
WORKER_DEFAULTS = {
    "max_age_seconds": 0.5,
    "min_hits_seconds": 0.0,
    "max_resurrection_seconds": 0.3,
    "match_distance_multiplier": 8.0,
    "resurrection_search_multiplier": 2.5,
    "duplicate_distance_multiplier": 1.2,
    "iou_threshold": 0.25,
    "lost_search": "frames",
    "stay_in_reference": False,
}
# BeeTracking.lookback_seconds.
LOOKBACK_SECONDS = 0.5

TRACKING_COLUMNS = ["frame", "track_id", "x1", "y1", "x2", "y2", "cx", "cy",
                    "detected", "confidence", "source", "class", "taxon", "taxon_confidence",
                    "taxon_votes", "bee_id", "bee_id_method", "bee_id_confidence", "mode"]


def tracker_params(sent: dict | None) -> dict:
    """BeeTracker settings as the worker resolves them: its defaults, then
    whatever the pipeline sent (``tracker_params`` of the job config, or
    ``executors.tracker_settings`` of a MOT node). Keys BeeTracker doesn't take
    (old frame-count settings) are ignored, as the worker ignores them."""
    params = dict(WORKER_DEFAULTS)
    for key, value in (sent or {}).items():
        if key not in params or value is None:
            continue
        if key == "lost_search":
            params[key] = "speed" if str(value).lower() == "speed" else "frames"
        elif key == "stay_in_reference":
            params[key] = str(value).lower() in ("1", "true", "on", "yes")
        else:
            params[key] = float(value)
    return params


def _f(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _row(det):
    """A detections.csv row -> the positional row BeeTracker.update takes."""
    return [float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"]),
            _f(det.get("confidence")) or 0.0, det.get("source") or "yolo",
            det.get("taxon") if isinstance(det.get("taxon"), str) and det.get("taxon") else "bee"]


def calls_from_log(detections, log):
    """``[(frame, [rows], kind)]`` from the worker's frame log: one entry per
    ``update()`` call, in order; detections carry the call's ``seq``."""
    by_seq = {}
    for det in detections:
        seq = _f(det.get("seq"))
        if seq is not None:
            by_seq.setdefault(int(seq), []).append(_row(det))
    out = []
    for entry in sorted(log, key=lambda e: int(float(e["seq"]))):
        seq = int(float(entry["seq"]))
        out.append((int(float(entry["frame"])), by_seq.get(seq, []),
                    entry.get("kind") or "current"))
    return out


def calls_from_detections(detections, fps, first_frame, last_frame):
    """``[(frame, [rows], kind)]`` rebuilt from ``detections.csv`` alone (a clip
    tracked before the frame log). Kinds: ``lookback`` or ``current``."""
    current, lookback = {}, {}
    for det in detections:
        frame = _f(det.get("frame"))
        if frame is None:
            continue
        target = lookback if (det.get("mode") or "") == "lookback" else current
        target.setdefault(int(frame), []).append(_row(det))

    # Each run of consecutive lookback frames ends at the frame motion was seen
    # (the buffer holds the trigger frame), and the buffer is the half second
    # before it, empty frames included.
    width = max(int(fps * LOOKBACK_SECONDS), 1)
    triggers = {}
    run = []
    for frame in sorted(lookback):
        if run and frame != run[-1] + 1:
            triggers[run[-1]] = run
            run = []
        run.append(frame)
    if run:
        triggers[run[-1]] = run

    out = []
    for frame in range(first_frame, last_frame + 1):
        if frame in triggers:
            for f in range(max(frame - width + 1, first_frame), frame + 1):
                out.append((f, lookback.get(f, []), "lookback"))
        out.append((frame, current.get(frame, []), "current"))
    return out


def run_calls(calls, fps, frame_height, params, references=None):
    """Feed the calls to a fresh BeeTracker; the worker's per-frame results.

    As in BeeTracking.process_frame, the backfill of a track confirmed on a
    current frame is reported (before that frame), and a lookback call's
    backfill is not read."""
    from .bee_tracker import BeeTracker

    params = {k: v for k, v in params.items() if k != "lost_rules"}
    tracker = BeeTracker(fps=fps, bee_size=None, frame_height=frame_height,
                         references=references, **params)
    results = []
    for frame, rows, kind in calls:
        tracks = tracker.update(rows, frame)
        if kind == "lookback":
            results.append({"frame_num": frame, "tracks": tracks, "mode": "lookback"})
            continue
        by_frame = {}
        for bf_frame, row in getattr(tracker, "backfill", None) or []:
            by_frame.setdefault(bf_frame, []).append(row)
        for bf_frame in sorted(by_frame):
            results.append({"frame_num": bf_frame, "tracks": by_frame[bf_frame], "mode": "tracking"})
        results.append({"frame_num": frame, "tracks": tracks,
                        "mode": "tracking" if rows else "motion_detection"})
    return results


def tracking_rows(results):
    """The worker's tracking table (BeeTracking._create_tracking_dataframe):
    one row per track per frame, sorted by frame and track id. Species columns
    are filled from the crop votes afterwards."""
    rows = []
    for result in results:
        for track in result["tracks"]:
            tid = track.get("track_id")
            if tid is None:
                continue
            rows.append({
                "frame": result["frame_num"], "track_id": tid,
                "x1": track["x1"], "y1": track["y1"], "x2": track["x2"], "y2": track["y2"],
                "cx": track["cx"], "cy": track["cy"],
                "detected": int(track.get("detected", 1)),
                "confidence": track.get("confidence", 0.0),
                "source": track.get("source", "unknown"),
                "class": track.get("taxon", "bee"),
                "taxon": track.get("taxon", "bee"),
                "taxon_confidence": None, "taxon_votes": 0,
                "bee_id": track.get("bee_id"), "bee_id_method": track.get("bee_id_method"),
                "bee_id_confidence": track.get("bee_id_confidence", 0.0),
                "mode": result["mode"],
            })
    rows.sort(key=lambda r: (r["frame"], r["track_id"]))
    return rows


def replay(detections, fps, frame_height, params, first_frame, last_frame, log=None,
           references=None):
    """Tracking rows for one clip (or one chunk of one). ``references``: the
    tubes' boxes in pixels, for "a track lost at a tube stays there"."""
    if log:
        calls = calls_from_log(detections, log)
    else:
        calls = calls_from_detections(detections, fps, first_frame, last_frame)
    return tracking_rows(run_calls(calls, fps, frame_height, params, references))
