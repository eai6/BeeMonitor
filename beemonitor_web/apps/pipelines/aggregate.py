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

# Widest pairing bounds. Summaries store trips paired at these so any narrower
# user bounds are a pure read-time filter on duration (pairing consumes the
# Exit on every Entry regardless of bounds, so results are identical).
FULL_MIN_SEC = 0.0
FULL_MAX_SEC = 86400.0


def clamp_trip_bounds(min_raw, max_raw):
    """(min_sec, max_sec) from raw query values, defaulted + clamped to
    [0, 86400] with max >= min. Shared by the batch page and the device chart
    so both interpret bounds identically."""
    def _num(raw, default):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    min_sec = max(0.0, min(86400.0, _num(min_raw, DEFAULT_MIN_SEC)))
    max_sec = max(min_sec, min(86400.0, _num(max_raw, DEFAULT_MAX_SEC)))
    return min_sec, max_sec


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


def read_processed_csv(blob_path, use_cache=False):
    """Rows (as dicts) of a CSV in the processed bucket; [] on any failure.

    ``use_cache=True`` memoizes the parsed rows (LocMem, 6 h) — safe only for
    blobs that are immutable once written (a completed job's events CSV) and
    worth it only for SMALL ones; events CSVs are a handful of rows per video,
    tracking CSVs are not cached."""
    if use_cache:
        from django.core.cache import cache

        key = f"prccsv:{blob_path}"
        hit = cache.get(key)
        if hit is not None:
            return hit
    try:
        from config.storage import get_s3_client

        buf = io.BytesIO()
        get_s3_client().download_to_stream("processed", blob_path, buf)
        text = buf.getvalue().decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception as e:
        logger.warning("aggregate: could not read %s: %s", blob_path, e)
        return []  # failures are NOT cached — next call retries
    if use_cache:
        cache.set(key, rows, 6 * 60 * 60)
    return rows


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
        for row in read_processed_csv(src["result"]["events_csv_path"], use_cache=True):
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
                "video_pk": getattr(src.get("video"), "pk", None),
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


def aggregate_trips(sources, min_sec=DEFAULT_MIN_SEC, max_sec=DEFAULT_MAX_SEC, events=None):
    """Pair Exit→Entry per nest across all sources on the absolute timeline.

    Returns (trips, summary). Trip dicts carry nest, exit/entry absolute times,
    duration, source video titles and track ids, and is_cross_video. Pass
    ``events`` (from collect_events) to reuse them and avoid a second S3 read.
    """
    if events is None:
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
                        "exit_video_pk": last_exit.get("video_pk"),
                        "entry_video_pk": ev.get("video_pk"),
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


def run_job_id(run):
    """The GPU job pk this run submitted, from its frozen context."""
    for out in (run.context or {}).values():
        job_id = (out or {}).get("job_id")
        if job_id:
            return job_id
    return None


def run_error(run):
    """The message that explains a failed run.

    Prefers the step's own error (the GPU job's text, which names the real
    cause) over the run-level summary, which is usually just "a step failed".
    """
    if run.status != run.Status.FAILED:
        return ""
    candidates = []
    for out in (run.context or {}).values():
        error = (out or {}).get("error")
        if error:
            candidates.append(error)
    # "Upstream step failed." is a consequence, never the cause — keep it only
    # if nothing else explains the run.
    real = [c for c in candidates if "Upstream step failed" not in c]
    return (real or candidates or [run.error_message or ""])[0]


def batch_rows(runs):
    """One row per run: clip, outcome, what it produced, what it cost.

    The batch page could only say "failed"; a row now carries the reason and
    the numbers, so a batch reads without opening anything.
    """
    from apps.analysis.models import Job, JobResult
    from apps.videos.models import Video

    video_ids = [v for r in runs if (v := run_video_id(r)) is not None]
    videos = {v.pk: v for v in Video.objects.filter(pk__in=video_ids).select_related("device")}
    job_ids = [j for r in runs if (j := run_job_id(r)) is not None]
    jobs = {j.pk: j for j in Job.objects.filter(pk__in=job_ids)}
    results = {r.job_id: r for r in JobResult.objects.filter(job_id__in=job_ids)}

    rows = []
    for run in runs:
        video = videos.get(run_video_id(run))
        job = jobs.get(run_job_id(run))
        result = results.get(job.pk) if job else None
        rows.append({
            "run": run,
            "video": video,
            "job": job,
            "result": result,
            "status": run.status,
            "error": run_error(run),
            "when": video.recorded_at if video else None,
            # GPU time is spent whether or not the run produced anything — a
            # failure that burned an hour is worth seeing next to one that did
            # not. Reported as time rather than money: seconds are a fact about
            # the work; a price is a claim about a rate card that drifts.
            "gpu_seconds": float(job.execution_seconds) if job and job.execution_seconds else 0.0,
        })
    rows.sort(key=lambda r: (r["when"] is None, r["when"]), reverse=True)
    return rows


def batch_summary(rows):
    """Counts and money for the whole batch, including what failures cost."""
    completed = [r for r in rows if r["status"] == "completed"]
    failed = [r for r in rows if r["status"] == "failed"]
    return {
        "total": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "running": len(rows) - len(completed) - len(failed),
        "pct_ok": round(100 * len(completed) / len(rows)) if rows else 0,
        "gpu_seconds": round(sum(r["gpu_seconds"] for r in rows), 1),
        "gpu_seconds_failed": round(sum(r["gpu_seconds"] for r in failed), 1),
        "failed_video_ids": [r["video"].pk for r in failed if r["video"]],
        "all_video_ids": [r["video"].pk for r in rows if r["video"]],
    }


# ── Analyzer-shaped results ──────────────────────────────────────────────────
# The batch page only ever aggregated foraging trips: it read events CSVs and
# rendered trips, entries/exits and nest chips whatever the pipeline computed. Run
# a Visitation pipeline and you got a page about a question you never asked.
#
# Each analyzer already tags its output with `table_kind` in run.context. Nothing
# read it at batch level. These functions merge each kind across the batch's runs,
# so the page can render what was actually computed.

def analyzer_outputs(run):
    """Every analyzer output in one run, as ``(kind, output)``.

    Most analyzers tag themselves with ``table_kind``. Foraging trips does not —
    it predates the table analyzers and returns ``artifact: "events"`` — so it is
    mapped here rather than left undetectable, which would make a trips pipeline
    look like it ran no analyzer at all.
    """
    out = []
    for value in (run.context or {}).values():
        if not isinstance(value, dict):
            continue
        kind = value.get("table_kind")
        if not kind and value.get("artifact") == "events":
            kind = "foraging_trips"
        if kind:
            out.append((kind, value))
    return out


def _merge_per_reference(bucket, rows, count_key):
    """Add one run's per-reference rows into the batch-wide tally.

    Keyed on the reference id, which is stable within a layout — that is what
    makes the same tube comparable across clips. Labels come along so the page
    never has to invent one.
    """
    for row in rows or []:
        ref_id = str(row.get("id", ""))
        if not ref_id:
            continue
        entry = bucket.setdefault(ref_id, {
            "id": ref_id, "label": row.get("label") or ref_id,
            count_key: 0, "visitors": 0, "partners": 0, "dwell_sec": 0.0,
            "duration_sec": 0.0, "clips": 0,
        })
        entry[count_key] += row.get(count_key, 0) or 0
        entry["visitors"] += row.get("visitors", 0) or 0
        entry["partners"] += row.get("partners", 0) or 0
        entry["dwell_sec"] += float(row.get("dwell_sec") or 0)
        entry["duration_sec"] += float(row.get("duration_sec") or 0)
        if row.get(count_key):
            entry["clips"] += 1
    return bucket


def aggregate_visitation(outputs):
    """Visits across the batch, and per reference.

    ``unique_visitors`` is summed rather than deduplicated: track ids are only
    unique within one clip, so the same bee in two clips is two visitors and
    there is no way to know otherwise. Stated on the page rather than hidden.
    """
    per_ref, totals = {}, {"unique_visitors": 0, "total_visits": 0, "total_dwell_sec": 0.0}
    for out in outputs:
        totals["unique_visitors"] += out.get("unique_visitors", 0) or 0
        totals["total_visits"] += out.get("total_visits", 0) or 0
        totals["total_dwell_sec"] += float(out.get("total_dwell_sec") or 0)
        _merge_per_reference(per_ref, out.get("per_reference"), "visits")

    rows = sorted(per_ref.values(), key=lambda r: (-r["visits"], r["id"]))
    for r in rows:
        r["dwell_sec"] = round(r["dwell_sec"], 1)
    totals["total_dwell_sec"] = round(totals["total_dwell_sec"], 1)
    totals["per_reference"] = rows
    totals["clips"] = len(outputs)
    return totals


def aggregate_interaction(outputs):
    per_ref, totals = {}, {"interaction_count": 0, "organism_organism": 0,
                           "organism_reference": 0, "total_duration_sec": 0.0}
    for out in outputs:
        for key in ("interaction_count", "organism_organism", "organism_reference"):
            totals[key] += out.get(key, 0) or 0
        totals["total_duration_sec"] += float(out.get("total_duration_sec") or 0)
        _merge_per_reference(per_ref, out.get("per_reference"), "interactions")

    rows = sorted(per_ref.values(), key=lambda r: (-r["interactions"], r["id"]))
    for r in rows:
        r["duration_sec"] = round(r["duration_sec"], 1)
    totals["total_duration_sec"] = round(totals["total_duration_sec"], 1)
    totals["per_reference"] = rows
    totals["clips"] = len(outputs)
    return totals


def aggregate_detection_count(outputs):
    totals = {"total": 0, "distinct": 0, "clips": len(outputs), "with_any": 0}
    for out in outputs:
        rows = out.get("rows") or []
        total = out.get("total") or sum(r.get("count", 0) or 0 for r in rows)
        distinct = out.get("distinct") or out.get("distinct_objects") or 0
        totals["total"] += total or 0
        totals["distinct"] += distinct or 0
        if total:
            totals["with_any"] += 1
    return totals


# Colony activity is deliberately absent: its computation stays, but it does not
# get a section of its own — a timeline belongs inside whichever analyzer ran.
AGGREGATORS = {
    "visitation": aggregate_visitation,
    "interaction": aggregate_interaction,
    "detection_count": aggregate_detection_count,
}

KIND_LABELS = {
    "foraging_trips": "Foraging trips",
    "visitation": "Visitation",
    "interaction": "Interactions",
    "detection_count": "Detection count",
}


def analyzer_results(runs):
    """``[{kind, label, summary}]`` for every analyzer this batch actually ran.

    Ordered by how many runs produced each, so the pipeline's main analyzer
    leads when a graph has more than one.
    """
    by_kind = {}
    for run in runs:
        for kind, output in analyzer_outputs(run):
            by_kind.setdefault(kind, []).append(output)

    results = []
    for kind, outputs in by_kind.items():
        aggregator = AGGREGATORS.get(kind)
        if not aggregator:
            # foraging_trips is rendered by the existing cross-video machinery;
            # colony_activity deliberately has no section. Both still register
            # so the page knows which analyzers ran.
            results.append({"kind": kind, "label": KIND_LABELS.get(kind, kind),
                            "summary": None, "clips": len(outputs)})
            continue
        results.append({
            "kind": kind,
            "label": KIND_LABELS.get(kind, kind.replace("_", " ").title()),
            "summary": aggregator(outputs),
            "clips": len(outputs),
        })
    results.sort(key=lambda r: -r["clips"])
    return results
