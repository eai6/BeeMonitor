"""Backfill interaction CSVs by spawning Modal CPU jobs.

Runs nest detection + interaction analysis on existing tracking data.
Uses Modal (CPU-only) for YOLO + numpy dependencies.

Usage:
    python manage.py backfill_interactions
    python manage.py backfill_interactions --limit 100
    python manage.py backfill_interactions --dry-run
"""

from django.core.management.base import BaseCommand

from apps.analysis.models import Job, JobResult


class Command(BaseCommand):
    help = "Backfill interaction CSVs via Modal (CPU-only)"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--limit", type=int, default=0)
        parser.add_argument("--chunk-size", type=int, default=50,
                            help="Jobs per Modal function call")

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        limit = options["limit"]
        chunk_size = options["chunk_size"]

        # Find completed jobs without interaction data
        qs = JobResult.objects.filter(
            job__status=Job.Status.COMPLETED,
            interaction_count=0,
        ).exclude(
            tracking_csv_path="",
        ).select_related("job__video")

        total = qs.count()
        self.stdout.write(f"Found {total} completed jobs without interactions")

        if limit:
            qs = qs[:limit]
            self.stdout.write(f"Processing first {limit}")

        if dry_run:
            self.stdout.write(self.style.WARNING("DRY RUN"))
            return

        # Build job configs
        job_configs = []
        for result in qs.iterator():
            video = result.job.video
            if not video.storage_key or video.storage_key.startswith("s3://"):
                continue

            tracking_path = result.tracking_csv_path
            if not tracking_path:
                mid = result.job.modal_job_id
                uid = str(result.job.user_id)
                if mid:
                    tracking_path = f"{uid}/{mid}/tracking_results.csv"
                else:
                    continue

            fps = 30.0
            stats = result.summary_stats or {}
            if stats.get("video_fps"):
                fps = float(stats["video_fps"])

            # Pass stored nest bboxes if available (avoids re-detection)
            nest_bboxes = stats.get("nest_bboxes", {})

            job_configs.append({
                "job_id": result.job.modal_job_id or str(result.job.pk),
                "job_pk": result.job.pk,
                "user_id": str(result.job.user_id),
                "video_blob_path": video.storage_key,
                "tracking_csv_path": tracking_path,
                "fps": fps,
                "nest_bboxes": nest_bboxes,
            })

        if not job_configs:
            self.stdout.write("No eligible jobs to backfill.")
            return

        self.stdout.write(f"Spawning {len(job_configs)} jobs in chunks of {chunk_size}...")

        try:
            import modal
            fn = modal.Function.from_name("beemonitor-cloud", "backfill_interactions")

            # Spawn chunks
            calls = []
            for i in range(0, len(job_configs), chunk_size):
                chunk = job_configs[i:i + chunk_size]
                call = fn.spawn(jobs=chunk)
                calls.append((call, chunk))
                self.stdout.write(f"  Spawned chunk {i // chunk_size + 1} ({len(chunk)} jobs)")

            self.stdout.write(f"Spawned {len(calls)} chunks on Modal. Results will update via polling.")
            self.stdout.write("Run 'python manage.py poll_interaction_backfill' to collect results,")
            self.stdout.write("or they will be collected when jobs complete.")

            # Optionally wait and collect results immediately
            completed = 0
            errors = 0
            for call, chunk in calls:
                try:
                    results = call.get()  # blocking wait
                    # Build lookup: job_id -> job_pk
                    id_to_pk = {jc["job_id"]: jc["job_pk"] for jc in chunk}

                    for r in results:
                        job_pk = id_to_pk.get(r.get("job_id"))
                        if not job_pk:
                            continue

                        if r.get("status") == "completed":
                            JobResult.objects.filter(job_id=job_pk).update(
                                interactions_csv_path=r.get("interactions_csv_path", ""),
                                interaction_count=r.get("interaction_count", 0),
                            )
                            completed += 1
                        elif r.get("status") == "failed":
                            errors += 1
                            self.stderr.write(f"  Failed {r['job_id']}: {r.get('error', '')}")

                except Exception as e:
                    errors += len(chunk)
                    self.stderr.write(f"  Chunk failed: {e}")

            self.stdout.write(self.style.SUCCESS(
                f"Done. Completed: {completed}, Errors: {errors}"
            ))

        except ImportError:
            self.stderr.write(self.style.ERROR("Modal not installed. Cannot run backfill."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"Backfill failed: {e}"))
