"""Backfill foraging trips for already-completed jobs.

Downloads each job's events CSV from Azure, computes foraging trips,
uploads the trips CSV back to Azure, and updates the JobResult record.
No GPU needed — pure CPU computation.

Usage:
    python manage.py backfill_foraging_trips
    python manage.py backfill_foraging_trips --dry-run
    python manage.py backfill_foraging_trips --limit 100
"""

import csv
import io
import tempfile
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from apps.analysis.models import Job, JobResult


class Command(BaseCommand):
    help = "Backfill foraging trips from existing events CSVs in Azure"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
        parser.add_argument("--limit", type=int, default=0, help="Max jobs to process (0 = all)")
        parser.add_argument("--fps", type=float, default=30.0, help="Video FPS for frame→seconds conversion")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        limit = options["limit"]
        fps = options["fps"]

        # Find completed jobs without foraging trip data
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

        conn_str = getattr(settings, "AZURE_STORAGE_CONNECTION_STRING", "")
        if not conn_str:
            self.stderr.write(self.style.ERROR("AZURE_STORAGE_CONNECTION_STRING not set"))
            return

        from azure.storage.blob import BlobServiceClient
        service = BlobServiceClient.from_connection_string(conn_str)

        from cloud.wrapper.foraging import compute_foraging_trips, compute_trip_summary

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
                # Download events CSV from Azure
                blob = service.get_blob_client("processed", events_path)
                content = blob.download_blob().readall().decode("utf-8")
                events_df = pd.read_csv(io.StringIO(content))

                if events_df.empty:
                    skipped += 1
                    continue

                # Check fps from summary_stats if available
                job_fps = fps
                stats = result.summary_stats or {}
                if stats.get("video_fps"):
                    job_fps = float(stats["video_fps"])

                # Compute foraging trips
                trips_df = compute_foraging_trips(events_df, fps=job_fps)
                trip_summary = compute_trip_summary(trips_df)

                if trips_df.empty:
                    # Update record to reflect 0 trips (so we don't re-process)
                    result.foraging_trip_count = 0
                    result.avg_trip_duration_sec = 0
                    # Store trip summary in summary_stats
                    stats["foraging_trips"] = trip_summary
                    result.summary_stats = stats
                    result.save(update_fields=[
                        "foraging_trip_count", "avg_trip_duration_sec", "summary_stats",
                    ])
                    skipped += 1
                    continue

                # Build upload path
                uid = str(result.job.user_id)
                mid = result.job.modal_job_id or str(result.job.pk)
                trips_blob_path = f"{uid}/{mid}/foraging_trips.csv"

                # Upload trips CSV to Azure
                trips_csv_content = trips_df.to_csv(index=False)
                blob_client = service.get_blob_client("processed", trips_blob_path)
                blob_client.upload_blob(trips_csv_content.encode("utf-8"), overwrite=True)

                # Update JobResult
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
