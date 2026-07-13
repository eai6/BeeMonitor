"""DailyForagingSummary lifecycle: the single source of truth for foraging trips.

One pairing implementation (``apps.pipelines.aggregate.aggregate_trips``) feeds
everything: the batch page pairs its own runs live; this module pairs a whole
(user, site, device, day) group and PERSISTS it, so charts read trips from the
DB — never from S3 in a request path.

Freshness is a dirty flag, not a scan: ``mark_stale_for_job`` runs inside the
job-completion path (cheap upsert), and the background reconciler calls
``sweep_stale`` every tick to recompute flagged (or never-computed) rows,
newest day first. Recomputes are idempotent full rebuilds, so concurrent
sweeps across gunicorn workers / App Runner instances are safe — last writer
wins, and the guarded stale-clear keeps a mid-recompute re-mark alive for the
next tick.

Rows with ``device IS NULL`` (legacy, pre-device-keyed summaries) are never
touched here: Postgres treats NULLs as distinct in the unique constraint, so
concurrent upserts would duplicate them — and the device chart never reads
them anyway.
"""

import io
import logging
import statistics
from datetime import datetime, timedelta, timezone as dt_timezone

from django.utils import timezone

from apps.analysis.models import DailyForagingSummary, Job, JobResult
from apps.pipelines import aggregate

logger = logging.getLogger(__name__)

# Compact stored-trip row: [exit_epoch_sec, duration_sec, nest, is_cross_video].
# Kept lean so a long custom range concatenates cheaply; full detail (entry
# time, videos, track ids) stays in the day's trips CSV.
TRIPS_PER_DAY_CAP = 10_000

TRIP_CSV_COLUMNS = [
    "nest", "exit_time", "entry_time", "duration_sec",
    "exit_video_id", "entry_video_id", "exit_track_id", "entry_track_id",
    "is_cross_video",
]


def mark_stale_for_job(job) -> None:
    """Flag the summary row for this job's (user, site, device, day) as stale.

    Called from the job-completion path (which runs inside browser poll
    handlers), so it must stay CHEAP: one upsert, no S3, no recompute. The
    reconciler sweep does the heavy lifting within ~a tick. Never raises."""
    try:
        video = getattr(job, "video", None)
        if video is None or not video.recorded_at or not video.site_name:
            return
        if video.device_id is None:
            return  # legacy NULL-device rows: unique constraint can't dedupe them
        DailyForagingSummary.objects.update_or_create(
            user_id=job.user_id,
            site_name=video.site_name,
            device_id=video.device_id,
            date=video.recorded_at.date(),
            defaults={"stale": True, "stale_marked_at": timezone.now()},
        )
    except Exception:  # noqa: BLE001 - must never break job completion
        logger.exception("mark_stale_for_job failed for job %s", getattr(job, "pk", "?"))


def day_sources(user_id, site_name, device_id, day):
    """aggregate.collect_events-compatible sources for one group's videos:
    the latest completed analysis per video recorded on ``day`` (UTC)."""
    jrs = (JobResult.objects
           .filter(job__status=Job.Status.COMPLETED,
                   job__user_id=user_id,
                   job__video__site_name=site_name,
                   job__video__device_id=device_id,
                   job__video__recorded_at__date=day)
           .exclude(events_csv_path="")
           .select_related("job", "job__video")
           .order_by("job__video_id", "-job__id"))
    sources, seen = [], set()
    for jr in jrs:
        v = jr.job.video
        if v.pk in seen or not v.recorded_at:
            continue
        seen.add(v.pk)  # latest completed analysis per video
        stats = jr.summary_stats or {}
        fps = v.fps or stats.get("video_fps") or stats.get("fps") or aggregate.DEFAULT_FPS
        sources.append({
            "title": v.title or f"Video {v.pk}",
            "video": v,
            "result": {"events_csv_path": jr.events_csv_path},
            "recorded_at": v.recorded_at,
            "fps": max(float(fps), 1.0),
        })
    sources.sort(key=lambda s: s["recorded_at"])
    return sources


def recompute_day(user_id, site_name, device_id, day) -> "DailyForagingSummary | None":
    """Full idempotent rebuild of one (user, site, device, day) summary.

    Pairs UNFILTERED (0–86400 s) and stores compact trips for read-time bounds
    filtering; row stats + the downloadable CSV keep the DEFAULT bounds so
    their meaning is unchanged. Reads S3 (one events CSV per video) — callers
    must be background paths (sweep / management command), never a request."""
    seen_marked_at = (DailyForagingSummary.objects
                      .filter(user_id=user_id, site_name=site_name,
                              device_id=device_id, date=day)
                      .values_list("stale_marked_at", flat=True).first())

    sources = day_sources(user_id, site_name, device_id, day)
    events = aggregate.collect_events(sources) if sources else []
    all_trips, _ = aggregate.aggregate_trips(
        sources, aggregate.FULL_MIN_SEC, aggregate.FULL_MAX_SEC, events=events)
    if len(all_trips) > TRIPS_PER_DAY_CAP:
        logger.warning("day trips capped for %s/%s dev%s %s: %d -> %d",
                       user_id, site_name, device_id, day,
                       len(all_trips), TRIPS_PER_DAY_CAP)
        all_trips = all_trips[:TRIPS_PER_DAY_CAP]

    compact = [[round(t["exit_time"].timestamp(), 1), t["duration_sec"],
                str(t["nest"]), bool(t["is_cross_video"])] for t in all_trips]

    # Row stats + CSV at the DEFAULT bounds (same numbers the model always held).
    default_trips = [t for t in all_trips
                     if aggregate.DEFAULT_MIN_SEC <= t["duration_sec"] <= aggregate.DEFAULT_MAX_SEC]
    durations = [t["duration_sec"] for t in default_trips]
    nest_counts = {}
    for t in default_trips:
        nest_counts[str(t["nest"])] = nest_counts.get(str(t["nest"]), 0) + 1

    trips_csv_path = ""
    if default_trips:
        trips_csv_path = _upload_trips_csv(user_id, site_name, device_id, day, default_trips)

    summary, _created = DailyForagingSummary.objects.update_or_create(
        user_id=user_id, site_name=site_name, device_id=device_id, date=day,
        defaults={
            "total_trips": len(default_trips),
            "cross_video_trips": sum(1 for t in default_trips if t["is_cross_video"]),
            "avg_duration_sec": round(statistics.mean(durations), 1) if durations else 0,
            "median_duration_sec": round(statistics.median(durations), 1) if durations else 0,
            "trips_per_nest": dict(sorted(nest_counts.items())),
            "trips_csv_path": trips_csv_path,
            "video_count": len(sources),
        },
    )
    # Guarded stale clear: only if no NEW mark landed while we were reading S3.
    # A re-marked row stays stale and the next sweep tick redoes it.
    (DailyForagingSummary.objects
     .filter(pk=summary.pk, stale_marked_at=seen_marked_at)
     .update(stale=False))
    # trips is set unconditionally (it reflects THIS rebuild either way).
    DailyForagingSummary.objects.filter(pk=summary.pk).update(trips=compact)
    return summary


def _upload_trips_csv(user_id, site_name, device_id, day, trips) -> str:
    """Write the day's default-bounds trips CSV to S3 (same columns as ever)."""
    import csv as _csv

    from config.storage import get_s3_client

    out = io.StringIO()
    writer = _csv.DictWriter(out, fieldnames=TRIP_CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for t in trips:
        writer.writerow({
            "nest": t["nest"],
            "exit_time": t["exit_time"].isoformat(),
            "entry_time": t["entry_time"].isoformat(),
            "duration_sec": t["duration_sec"],
            "exit_video_id": t.get("exit_video_pk") or "",
            "entry_video_id": t.get("entry_video_pk") or "",
            "exit_track_id": t.get("exit_track_id", ""),
            "entry_track_id": t.get("entry_track_id", ""),
            "is_cross_video": t["is_cross_video"],
        })
    blob_path = f"{user_id}/daily_trips/{site_name}_dev{device_id}_{day}.csv"
    try:
        get_s3_client().upload_stream(
            "processed", blob_path,
            io.BytesIO(out.getvalue().encode("utf-8")),
            content_type="text/csv",
        )
    except Exception as e:  # noqa: BLE001 - CSV is a convenience artifact
        logger.warning("trips CSV upload failed for %s: %s", blob_path, e)
        return ""
    return blob_path


def sweep_stale(limit: int = 4) -> int:
    """Recompute up to ``limit`` stale (or never-computed) summaries, newest
    day first so the default 7d chart window heals before deep history.
    Called from the background reconciler tick. Returns rows recomputed."""
    from django.db.models import Q

    rows = list(DailyForagingSummary.objects
                .filter(Q(stale=True) | Q(trips__isnull=True),
                        device__isnull=False)
                .order_by("-date")[:limit])
    done = 0
    for row in rows:
        try:
            recompute_day(row.user_id, row.site_name, row.device_id, row.date)
            done += 1
        except Exception:  # noqa: BLE001 - one bad day must not stall the sweep
            logger.exception("summary recompute failed for %s", row.pk)
            # Don't loop on a poisoned row forever: drop its flag (a future
            # completion re-marks it). Never-computed rows get trips=[] so they
            # leave the sweep filter too; rows with old data keep it.
            updates = {"stale": False}
            if row.trips is None:
                updates["trips"] = []
            DailyForagingSummary.objects.filter(pk=row.pk).update(**updates)
    return done


def device_trips(device, since, until, min_sec=None, max_sec=None):
    """Trip (exit_time, duration) tuples for a device in [since, until] (UTC),
    filtered by bounds — a PURE DB read over stored summaries.

    Summary rows are keyed by UTC calendar date while the chart window is an
    arbitrary instant range (display-tz), so rows are fetched with a ±1-day pad
    and trips filtered by their exact exit instant."""
    if min_sec is None:
        min_sec = aggregate.DEFAULT_MIN_SEC
    if max_sec is None:
        max_sec = aggregate.DEFAULT_MAX_SEC
    lo_ts, hi_ts = since.timestamp(), until.timestamp()
    rows = (DailyForagingSummary.objects
            .filter(device=device,
                    date__gte=since.date() - timedelta(days=1),
                    date__lte=until.date() + timedelta(days=1))
            .exclude(trips__isnull=True)
            .values_list("trips", flat=True))
    out = []
    for trips in rows:
        for row in trips or []:
            try:
                exit_epoch, duration = float(row[0]), float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if lo_ts <= exit_epoch <= hi_ts and min_sec <= duration <= max_sec:
                out.append(datetime.fromtimestamp(exit_epoch, tz=dt_timezone.utc))
    out.sort()
    return out
