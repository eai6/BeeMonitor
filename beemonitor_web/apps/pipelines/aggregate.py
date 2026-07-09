"""Cross-video aggregation for a batch of pipeline runs.

Videos record windows of activity: a bee can exit its nest in one clip and
return in a later one, so per-video trip pairing misses (or truncates) those
trips. This module places every Exit/Entry event from a batch's videos on an
absolute timeline (video.recorded_at + frame/fps) and pairs Exit→Entry per
nest across the whole batch — a stdlib port of
``cloud/wrapper/foraging.compute_daily_foraging_trips`` (the web image has no
pandas). It also builds combined events/tracking CSVs with video + absolute
time columns prepended.
"""

import csv
import io
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

DEFAULT_MIN_SEC = 10.0
DEFAULT_MAX_SEC = 7200.0
DEFAULT_FPS = 30.0


# ── Batch sources ─────────────────────────────────────────────────────────────


def run_video_id(run):
    """The video pk this run was launched on (from its frozen input.video step)."""
    for s in (run.steps or []):
        if s.get("block_type") == "input.video":
            vid = (s.get("config") or {}).get("video_id")
            if str(vid).isdigit():
                return int(vid)
    return None


def run_gpu_result(run):
    """The detect/track step's JobResult summary dict stored in run.context."""
    for out in (run.context or {}).values():
        result = (out or {}).get("result") or {}
        if result.get("events_csv_path") or result.get("tracking_csv_path"):
            return result
    return {}


def collect_sources(runs):
    """Pair each completed run with its video + CSV paths.

    Returns (sources, skipped) where sources are dicts with run/video/result/
    recorded_at/fps and skipped is a list of {video, reason} for runs that
    can't join the aggregate (no timestamp, no result yet, ...).
    """
    from apps.videos.models import Video

    video_ids = [vid for r in runs if (vid := run_video_id(r)) is not None]
    videos = {v.pk: v for v in Video.objects.filter(pk__in=video_ids)}

    sources, skipped = [], []
    for run in runs:
        vid = run_video_id(run)
        video = videos.get(vid)
        title = getattr(video, "title", None) or (f"Video {vid}" if vid else "unknown video")
        if video is None:
            skipped.append({"video": title, "reason": "video no longer exists"})
            continue
        if run.status != run.Status.COMPLETED:
            skipped.append({"video": title, "reason": f"run {run.status}"})
            continue
        result = run_gpu_result(run)
        if not result.get("events_csv_path"):
            skipped.append({"video": title, "reason": "no events CSV in run output"})
            continue
        if not video.recorded_at:
            skipped.append({
                "video": title,
                "reason": "no recorded-at timestamp — can't place it on the day's timeline",
            })
            continue
        stats = result.get("summary_stats") or {}
        fps = video.fps or stats.get("fps") or DEFAULT_FPS
        sources.append({
            "run": run,
            "video": video,
            "title": title,
            "result": result,
            "recorded_at": video.recorded_at,
            "fps": max(float(fps), 1.0),
        })
    sources.sort(key=lambda s: s["recorded_at"])
    return sources, skipped


# ── CSV plumbing ──────────────────────────────────────────────────────────────


def read_processed_csv(blob_path):
    """Rows (as dicts) of a CSV in the processed bucket; [] on any failure."""
    try:
        from config.storage import get_s3_client

        buf = io.BytesIO()
        get_s3_client().download_to_stream("processed", blob_path, buf)
        text = buf.getvalue().decode("utf-8", errors="replace")
        return list(csv.DictReader(io.StringIO(text)))
    except Exception as e:
        logger.warning("aggregate: could not read %s: %s", blob_path, e)
        return []


def _frame_number(row):
    for key in ("frame_number", "frame", "frame_num"):
        if row.get(key, "") not in ("", None):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


def combined_csv(sources, path_key):
    """Concatenate each source's CSV (events or tracking), prepending
    video_title / video_recorded_at / absolute_time columns. Returns
    (fieldnames, row_iterator); fieldnames is None when nothing was readable."""
    extra = ["video_title", "video_recorded_at", "absolute_time"]
    fieldnames = None
    all_rows = []
    for src in sources:
        blob = src["result"].get(path_key)
        if not blob:
            continue
        rows = read_processed_csv(blob)
        if not rows:
            continue
        if fieldnames is None:
            fieldnames = extra + [c for c in rows[0].keys()]
        for row in rows:
            frame = _frame_number(row)
            abs_time = (
                src["recorded_at"] + timedelta(seconds=frame / src["fps"])
                if frame is not None else None
            )
            out = {
                "video_title": src["title"],
                "video_recorded_at": src["recorded_at"].isoformat(),
                "absolute_time": abs_time.isoformat() if abs_time else "",
            }
            out.update(row)
            all_rows.append(out)
    return fieldnames, all_rows


# ── Cross-video foraging trips ────────────────────────────────────────────────


def collect_events(sources):
    """Flatten all sources' Exit/Entry events onto the absolute timeline.

    Each event: {action, nest, time (datetime), video, track_id}. Shared by
    trip pairing and the activity charts.
    """
    events = []
    for src in sources:
        for row in read_processed_csv(src["result"]["events_csv_path"]):
            action = (row.get("action") or "").strip()
            nest = row.get("nest", "")
            frame = _frame_number(row)
            if action not in ("Exit", "Entry") or nest in ("", None) or frame is None:
                continue
            events.append({
                "action": action,
                "nest": str(nest),
                "time": src["recorded_at"] + timedelta(seconds=frame / src["fps"]),
                "video": src["title"],
                "track_id": row.get("track_id", ""),
            })
    return events


def activity_charts(events, trips=None):
    """Server-computed charts for the batch page (CSP-safe, no JS charting):

    - hour_of_day: 24 buckets, exit vs entry split — the daily rhythm.
    - ts: an hourly (or daily for long spans) time series with pre-computed SVG
      polyline points for exit / entry / all-events / trips line plots.
    Returns {} when there are no timestamped events.
    """
    if not events:
        return {}
    trips = trips or []
    exit_events = sum(1 for e in events if e["action"] == "Exit")
    entry_events = len(events) - exit_events

    # Hour-of-day, exit vs entry (stacked bars in the template).
    hod_exit, hod_entry = [0] * 24, [0] * 24
    for e in events:
        (hod_exit if e["action"] == "Exit" else hod_entry)[e["time"].hour] += 1
    hod_max = max((hod_exit[h] + hod_entry[h]) for h in range(24)) or 1
    hour_of_day = [{
        "label": f"{h:02d}", "exit": hod_exit[h], "entry": hod_entry[h],
        "total": hod_exit[h] + hod_entry[h],
        "pct_exit": round(100 * hod_exit[h] / hod_max, 1),
        "pct_entry": round(100 * hod_entry[h] / hod_max, 1),
    } for h in range(24)]
    hod_peak = max(range(24), key=lambda h: hod_exit[h] + hod_entry[h])

    # Time series: hourly buckets across the span (daily if the span is huge).
    def _hkey(t):
        return t.replace(minute=0, second=0, microsecond=0)
    tmin, tmax = _hkey(min(e["time"] for e in events)), _hkey(max(e["time"] for e in events))
    step, fmt = timedelta(hours=1), "%m-%d %H:%M"
    if (tmax - tmin) > timedelta(days=10):
        def _hkey(t):
            return t.replace(hour=0, minute=0, second=0, microsecond=0)
        tmin, tmax = _hkey(tmin), _hkey(tmax)
        step, fmt = timedelta(days=1), "%m-%d"

    buckets = []
    cur = tmin
    while cur <= tmax:
        buckets.append(cur)
        cur += step
    idx = {b: i for i, b in enumerate(buckets)}
    n = len(buckets)
    ex, en, tr = [0] * n, [0] * n, [0] * n
    for e in events:
        i = idx.get(_hkey(e["time"]))
        if i is not None:
            (ex if e["action"] == "Exit" else en)[i] += 1
    for t in trips:
        i = idx.get(_hkey(t["exit_time"]))
        if i is not None:
            tr[i] += 1
    evt = [ex[i] + en[i] for i in range(n)]

    W, H, PAD = 560, 170, 30

    def _pts(vals, ymax):
        if not vals or ymax <= 0:
            return ""
        pw, ph = W - 2 * PAD, H - 2 * PAD
        out = []
        for i, v in enumerate(vals):
            x = PAD + (pw * i / (n - 1) if n > 1 else pw / 2)
            y = PAD + ph * (1 - v / ymax)
            out.append(f"{x:.1f},{y:.1f}")
        return " ".join(out)

    ev_ymax = max(max(ex, default=0), max(en, default=0), 1)
    trip_ymax = max(max(tr, default=0), 1)
    labels = [b.strftime(fmt) for b in buckets]
    xticks = []
    if n:
        for i in dict.fromkeys((0, n // 2, n - 1)):
            x = PAD + (W - 2 * PAD) * i / (n - 1) if n > 1 else W / 2
            xticks.append({"label": labels[i], "x": round(x, 1)})

    return {
        "total_events": len(events),
        "exit_events": exit_events, "entry_events": entry_events,
        "hour_of_day": hour_of_day, "hod_peak": hod_peak,
        "ts": {
            "w": W, "h": H, "pad": PAD, "n": n,
            "exit_pts": _pts(ex, ev_ymax), "entry_pts": _pts(en, ev_ymax),
            "events_pts": _pts(evt, ev_ymax), "trips_pts": _pts(tr, trip_ymax),
            "ev_ymax": ev_ymax, "trip_ymax": trip_ymax,
            "xticks": xticks, "total_trips": len(trips),
        },
    }


def aggregate_trips(sources, min_sec=DEFAULT_MIN_SEC, max_sec=DEFAULT_MAX_SEC):
    """Pair Exit→Entry per nest across all sources on the absolute timeline.

    Returns (trips, summary). Trip dicts carry nest, exit/entry absolute times,
    duration, source video titles and track ids, and is_cross_video.
    """
    events = collect_events(sources)

    trips = []
    by_nest = {}
    for ev in events:
        by_nest.setdefault(ev["nest"], []).append(ev)
    for nest, nest_events in sorted(by_nest.items()):
        nest_events.sort(key=lambda e: e["time"])
        last_exit = None
        for ev in nest_events:
            if ev["action"] == "Exit":
                last_exit = ev
            elif ev["action"] == "Entry" and last_exit is not None:
                duration = (ev["time"] - last_exit["time"]).total_seconds()
                if min_sec <= duration <= max_sec:
                    trips.append({
                        "nest": nest,
                        "exit_time": last_exit["time"],
                        "entry_time": ev["time"],
                        "duration_sec": round(duration, 2),
                        "exit_video": last_exit["video"],
                        "entry_video": ev["video"],
                        "exit_track_id": last_exit["track_id"],
                        "entry_track_id": ev["track_id"],
                        "is_cross_video": last_exit["video"] != ev["video"],
                    })
                last_exit = None

    trips.sort(key=lambda t: t["exit_time"])
    durations = [t["duration_sec"] for t in trips]
    per_nest = {}
    for t in trips:
        per_nest[t["nest"]] = per_nest.get(t["nest"], 0) + 1
    summary = {
        "total_trips": len(trips),
        "cross_video_trips": sum(1 for t in trips if t["is_cross_video"]),
        "total_events": len(events),
        "avg_duration_sec": round(sum(durations) / len(durations), 1) if durations else 0,
        "min_duration_sec": min(durations) if durations else 0,
        "max_duration_sec": max(durations) if durations else 0,
        "trips_per_nest": dict(sorted(per_nest.items())),
    }
    return trips, summary


def trips_csv_rows(trips):
    """(fieldnames, rows) for downloading the aggregated trips as CSV."""
    fieldnames = ["nest", "exit_time", "entry_time", "duration_sec",
                  "exit_video", "entry_video", "exit_track_id", "entry_track_id",
                  "is_cross_video"]
    rows = [{**t,
             "exit_time": t["exit_time"].isoformat(),
             "entry_time": t["entry_time"].isoformat()} for t in trips]
    return fieldnames, rows
