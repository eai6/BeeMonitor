"""Compute cross-video foraging trips for each site+day.

Combines events from all videos at the same site+day, converts frame numbers
to absolute timestamps, and pairs Exit→Entry across video boundaries.

Usage:
    python manage.py compute_daily_trips
    python manage.py compute_daily_trips --site SiteA --date 2024-06-01
    python manage.py compute_daily_trips --recompute
"""

import csv
import io
import statistics
from collections import defaultdict
from datetime import datetime, timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.analysis.models import DailyForagingSummary, Job, JobResult


def _compute_daily_trips(video_events, min_sec=10, max_sec=7200):
    """Compute foraging trips across multiple videos using absolute timestamps.

    Args:
        video_events: list of dicts with recorded_at, fps, video_id, events (list of dicts)
    Returns:
        list of trip dicts
    """
    all_events = []
    for ve in video_events:
        recorded_at = ve["recorded_at"]
        fps = max(ve.get("fps", 30.0), 1.0)
        video_id = ve.get("video_id", "")

        for row in ve["events"]:
            sec_offset = float(row["frame_number"]) / fps
            abs_time = recorded_at + timedelta(seconds=sec_offset)
            all_events.append({
                "action": row["action"],
                "nest": row["nest"],
                "absolute_time": abs_time,
                "video_id": video_id,
                "track_id": row.get("track_id", ""),
            })

    if not all_events:
        return []

    # Group by nest, sort by time, pair Exit→Entry
    by_nest = defaultdict(list)
    for e in all_events:
        by_nest[e["nest"]].append(e)

    trips = []
    for nest_id, nest_events in by_nest.items():
        nest_sorted = sorted(nest_events, key=lambda x: x["absolute_time"])
        last_exit = None
        for ev in nest_sorted:
            if ev["action"] == "Exit":
                last_exit = ev
            elif ev["action"] == "Entry" and last_exit is not None:
                duration = (ev["absolute_time"] - last_exit["absolute_time"]).total_seconds()
                if min_sec <= duration <= max_sec:
                    trips.append({
                        "nest": nest_id,
                        "exit_time": last_exit["absolute_time"].isoformat(),
                        "entry_time": ev["absolute_time"].isoformat(),
                        "duration_sec": round(duration, 2),
                        "exit_video_id": last_exit["video_id"],
                        "entry_video_id": ev["video_id"],
                        "exit_track_id": last_exit["track_id"],
                        "entry_track_id": ev["track_id"],
                        "is_cross_video": last_exit["video_id"] != ev["video_id"],
                    })
                last_exit = None
    return trips


TRIP_COLUMNS = [
    "nest", "exit_time", "entry_time", "duration_sec",
    "exit_video_id", "entry_video_id", "exit_track_id", "entry_track_id",
    "is_cross_video",
]


class Command(BaseCommand):
    help = "Compute cross-video foraging trips for each site+day"

    def add_arguments(self, parser):
        parser.add_argument("--site", type=str, default="", help="Filter to specific site")
        parser.add_argument("--date", type=str, default="", help="Filter to specific date (YYYY-MM-DD)")
        parser.add_argument("--recompute", action="store_true", help="Recompute even if already exists")
        parser.add_argument("--fps", type=float, default=30.0, help="Default FPS")

    def handle(self, *args, **options):
        site_filter = options["site"]
        date_filter = options["date"]
        recompute = options["recompute"]
        default_fps = options["fps"]

        # Find all completed jobs grouped by site+day
        qs = JobResult.objects.filter(
            job__status=Job.Status.COMPLETED,
        ).exclude(events_csv_path="").select_related("job__video")

        if site_filter:
            qs = qs.filter(job__video__site_name=site_filter)
        if date_filter:
            try:
                d = datetime.strptime(date_filter, "%Y-%m-%d").date()
                qs = qs.filter(job__video__year=d.year, job__video__month=d.month, job__video__day=d.day)
            except ValueError:
                self.stderr.write(f"Invalid date: {date_filter}")
                return

        # Group by (user, site, device, date). One device == one camera, so
        # cross-video pairing stays valid within a device's own videos.
        groups = defaultdict(list)
        for result in qs.iterator():
            video = result.job.video
            if not video.recorded_at or not video.site_name:
                continue
            key = (result.job.user_id, video.site_name, video.device_id, video.recorded_at.date())
            groups[key].append(result)

        self.stdout.write(f"Found {len(groups)} site+device+day groups")

        if recompute:
            # Clear prior summaries (including legacy device=NULL rows) for the
            # site+days we're about to rebuild, so the new device-keyed rows can't
            # double-count alongside stale ones.
            site_day_keys = {(uid, site, date) for (uid, site, _dev, date) in groups}
            for uid, site, date in site_day_keys:
                DailyForagingSummary.objects.filter(
                    user_id=uid, site_name=site, date=date).delete()
        else:
            # Skip groups that already have a summary
            existing = set(
                DailyForagingSummary.objects.values_list(
                    "user_id", "site_name", "device_id", "date")
            )
            groups = {k: v for k, v in groups.items() if k not in existing}
            self.stdout.write(f"After skipping existing: {len(groups)} groups to process")

        from config.storage import get_s3_client
        s3 = get_s3_client()

        processed = 0
        errors = 0

        for (user_id, site_name, device_id, date), results in groups.items():
            try:
                video_events = []
                for result in results:
                    path = result.events_csv_path
                    if not path:
                        mid = result.job.modal_job_id
                        uid = str(result.job.user_id)
                        if mid:
                            path = f"{uid}/{mid}/events.csv"
                        else:
                            continue

                    try:
                        buf = io.BytesIO()
                        s3.download_to_stream("processed", path, buf)
                        content = buf.getvalue().decode("utf-8")
                        reader = csv.DictReader(io.StringIO(content))
                        events = list(reader)
                    except Exception:
                        continue

                    if not events:
                        continue

                    video = result.job.video
                    fps = default_fps
                    stats = result.summary_stats or {}
                    if stats.get("video_fps"):
                        fps = float(stats["video_fps"])

                    video_events.append({
                        "video_id": str(video.pk),
                        "recorded_at": video.recorded_at,
                        "fps": fps,
                        "events": events,
                    })

                if not video_events:
                    continue

                trips = _compute_daily_trips(video_events)

                # Compute summary
                cross_video = sum(1 for t in trips if t["is_cross_video"])
                durations = [t["duration_sec"] for t in trips]
                nest_counts = defaultdict(int)
                for t in trips:
                    nest_counts[str(t["nest"])] += 1

                avg_dur = round(statistics.mean(durations), 1) if durations else 0
                med_dur = round(statistics.median(durations), 1) if durations else 0

                # Upload trips CSV
                trips_csv_path = ""
                if trips:
                    output = io.StringIO()
                    writer = csv.DictWriter(output, fieldnames=TRIP_COLUMNS)
                    writer.writeheader()
                    writer.writerows(trips)
                    blob_path = f"{user_id}/daily_trips/{site_name}_dev{device_id}_{date}.csv"
                    s3.upload_stream(
                        "processed", blob_path,
                        io.BytesIO(output.getvalue().encode("utf-8")),
                        content_type="text/csv",
                    )
                    trips_csv_path = blob_path

                DailyForagingSummary.objects.update_or_create(
                    user_id=user_id,
                    site_name=site_name,
                    device_id=device_id,
                    date=date,
                    defaults={
                        "total_trips": len(trips),
                        "cross_video_trips": cross_video,
                        "avg_duration_sec": avg_dur,
                        "median_duration_sec": med_dur,
                        "trips_per_nest": dict(nest_counts),
                        "trips_csv_path": trips_csv_path,
                        "video_count": len(video_events),
                    },
                )

                processed += 1
                if processed % 10 == 0:
                    self.stdout.write(f"  Processed {processed} groups...")

            except Exception as e:
                errors += 1
                self.stderr.write(f"  Error for {site_name} {date}: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"Done. Processed: {processed}, Errors: {errors}"
        ))
