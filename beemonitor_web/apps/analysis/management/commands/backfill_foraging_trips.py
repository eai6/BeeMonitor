"""Backfill foraging trips for already-completed jobs.

Downloads each job's events CSV from S3, computes foraging trips,
uploads the trips CSV back to S3, and updates the JobResult record.
No GPU needed — pure CPU computation. Uses only stdlib (no pandas).

Usage:
    python manage.py backfill_foraging_trips
    python manage.py backfill_foraging_trips --dry-run
    python manage.py backfill_foraging_trips --limit 100
"""

import csv
import io
import statistics
from collections import defaultdict

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.analysis.models import Job, JobResult

TRIP_COLUMNS = [
    "nest", "exit_frame", "entry_frame", "exit_sec",
    "entry_sec", "duration_sec", "exit_track_id", "entry_track_id",
]


def _parse_events_csv(content):
    """Parse events CSV content into list of dicts. Returns (rows, headers)."""
    reader = csv.DictReader(io.StringIO(content))
    return list(reader), reader.fieldnames or []


def _compute_foraging_trips(events, fps=30.0, min_sec=10, max_sec=7200):
    """Pair Exit→Entry events per nest. Events is a list of dicts."""
    if not events:
        return []

    # Check required columns
    sample = events[0]
    if not all(k in sample for k in ("action", "nest", "frame_number")):
        return []

    fps = max(fps, 1.0)

    # Group by nest
    by_nest = defaultdict(list)
    for row in events:
        by_nest[row["nest"]].append(row)

    trips = []
    for nest_id, nest_events in by_nest.items():
        nest_sorted = sorted(nest_events, key=lambda r: float(r["frame_number"]))
        last_exit = None
        for row in nest_sorted:
            if row["action"] == "Exit":
                last_exit = row
            elif row["action"] == "Entry" and last_exit is not None:
                exit_frame = float(last_exit["frame_number"])
                entry_frame = float(row["frame_number"])
                exit_sec = exit_frame / fps
                entry_sec = entry_frame / fps
                duration = entry_sec - exit_sec
                if min_sec <= duration <= max_sec:
                    trips.append({
                        "nest": nest_id,
                        "exit_frame": int(exit_frame),
                        "entry_frame": int(entry_frame),
                        "exit_sec": round(exit_sec, 2),
                        "entry_sec": round(entry_sec, 2),
                        "duration_sec": round(duration, 2),
                        "exit_track_id": last_exit.get("track_id", ""),
                        "entry_track_id": row.get("track_id", ""),
                    })
                last_exit = None
    return trips


def _compute_trip_summary(trips):
    """Compute summary stats from a list of trip dicts."""
    if not trips:
        return {
            "total_trips": 0, "avg_duration_sec": 0, "median_duration_sec": 0,
            "min_duration_sec": 0, "max_duration_sec": 0, "trips_per_nest": {},
        }
    durations = [t["duration_sec"] for t in trips]
    nest_counts = defaultdict(int)
    for t in trips:
        nest_counts[str(t["nest"])] += 1
    return {
        "total_trips": len(trips),
        "avg_duration_sec": round(statistics.mean(durations), 1),
        "median_duration_sec": round(statistics.median(durations), 1),
        "min_duration_sec": round(min(durations), 1),
        "max_duration_sec": round(max(durations), 1),
        "trips_per_nest": dict(nest_counts),
    }


def _trips_to_csv(trips):
    """Convert list of trip dicts to CSV string."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=TRIP_COLUMNS)
    writer.writeheader()
    writer.writerows(trips)
    return output.getvalue()


class Command(BaseCommand):
    help = "Backfill foraging trips from existing events CSVs in S3"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
        parser.add_argument("--limit", type=int, default=0, help="Max jobs to process (0 = all)")
        parser.add_argument("--fps", type=float, default=30.0, help="Video FPS for frame-to-seconds conversion")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        limit = options["limit"]
        fps = options["fps"]

        qs = JobResult.objects.filter(
            job__status=Job.Status.COMPLETED,
            foraging_trip_count=0,
        ).exclude(events_csv_path="").select_related("job")

        total = qs.count()
        self.stdout.write(f"Found {total} completed jobs without foraging trips")

        if limit:
            qs = qs[:limit]
            self.stdout.write(f"Processing first {limit}")

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN — no changes will be made"))
            return

        import io
        from config.storage import get_s3_client
        s3 = get_s3_client()

        processed = 0
        skipped = 0
        errors = 0

        for result in qs.iterator():
            events_path = result.events_csv_path
            if not events_path:
                mid = result.job.modal_job_id
                uid = str(result.job.user_id)
                if mid:
                    events_path = f"{uid}/{mid}/events.csv"
                else:
                    skipped += 1
                    continue

            try:
                buf = io.BytesIO()
                s3.download_to_stream("processed", events_path, buf)
                content = buf.getvalue().decode("utf-8")
                events, _ = _parse_events_csv(content)

                if not events:
                    skipped += 1
                    continue

                job_fps = fps
                stats = result.summary_stats or {}
                if stats.get("video_fps"):
                    job_fps = float(stats["video_fps"])

                trips = _compute_foraging_trips(events, fps=job_fps)
                trip_summary = _compute_trip_summary(trips)

                if not trips:
                    result.foraging_trip_count = 0
                    result.avg_trip_duration_sec = 0
                    stats["foraging_trips"] = trip_summary
                    result.summary_stats = stats
                    result.save(update_fields=[
                        "foraging_trip_count", "avg_trip_duration_sec", "summary_stats",
                    ])
                    skipped += 1
                    continue

                uid = str(result.job.user_id)
                mid = result.job.modal_job_id or str(result.job.pk)
                trips_blob_path = f"{uid}/{mid}/foraging_trips.csv"

                trips_csv = _trips_to_csv(trips)
                s3.upload_stream(
                    "processed", trips_blob_path,
                    io.BytesIO(trips_csv.encode("utf-8")),
                    content_type="text/csv",
                )

                stats["foraging_trips"] = trip_summary
                result.foraging_trips_csv_path = trips_blob_path
                result.foraging_trip_count = trip_summary["total_trips"]
                result.avg_trip_duration_sec = trip_summary["avg_duration_sec"]
                result.summary_stats = stats
                result.save(update_fields=[
                    "foraging_trips_csv_path", "foraging_trip_count",
                    "avg_trip_duration_sec", "summary_stats",
                ])

                processed += 1
                if processed % 50 == 0:
                    self.stdout.write(f"  Processed {processed}...")

            except Exception as e:
                errors += 1
                self.stderr.write(f"  Error on job {result.job.pk}: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"Done. Processed: {processed}, Skipped: {skipped}, Errors: {errors}"
        ))
