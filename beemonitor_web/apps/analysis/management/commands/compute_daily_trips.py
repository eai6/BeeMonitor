"""Compute cross-video foraging trips for each site+device+day.

Thin CLI over ``apps.analysis.foraging.recompute_day`` — the same code the
background sweep uses, so the command, the reconciler, and the device chart
can never disagree. Pairing itself lives in
``apps.pipelines.aggregate.aggregate_trips`` (also used by the batch page).

Usage:
    python manage.py compute_daily_trips
    python manage.py compute_daily_trips --site SiteA --date 2024-06-01
    python manage.py compute_daily_trips --recompute

Without ``--recompute``, groups whose summary already has the compact ``trips``
JSON are skipped — so the entrypoint run backfills exactly the rows the new
chart path needs and nothing else.
"""

from collections import defaultdict
from datetime import datetime

from django.core.management.base import BaseCommand

from apps.analysis import foraging
from apps.analysis.models import DailyForagingSummary, Job, JobResult


class Command(BaseCommand):
    help = "Compute cross-video foraging trips for each site+device+day"

    def add_arguments(self, parser):
        parser.add_argument("--site", type=str, default="", help="Filter to specific site")
        parser.add_argument("--date", type=str, default="", help="Filter to specific date (YYYY-MM-DD)")
        parser.add_argument("--recompute", action="store_true",
                            help="Recompute even if a summary with trips already exists")

    def handle(self, *args, **options):
        site_filter = options["site"]
        date_filter = options["date"]
        recompute = options["recompute"]

        qs = (JobResult.objects
              .filter(job__status=Job.Status.COMPLETED)
              .exclude(events_csv_path="")
              .select_related("job__video"))
        if site_filter:
            qs = qs.filter(job__video__site_name=site_filter)
        if date_filter:
            try:
                d = datetime.strptime(date_filter, "%Y-%m-%d").date()
                qs = qs.filter(job__video__recorded_at__date=d)
            except ValueError:
                self.stderr.write(f"Invalid date: {date_filter}")
                return

        groups = defaultdict(int)
        for result in qs.iterator():
            video = result.job.video
            if not video.recorded_at or not video.site_name or video.device_id is None:
                continue  # NULL-device groups are legacy; the chart never reads them
            key = (result.job.user_id, video.site_name, video.device_id,
                   video.recorded_at.date())
            groups[key] += 1
        group_keys = sorted(groups, key=lambda k: k[3], reverse=True)  # newest day first
        self.stdout.write(f"Found {len(group_keys)} site+device+day groups")

        if not recompute:
            done = set(
                DailyForagingSummary.objects
                .filter(device__isnull=False, trips__isnull=False)
                .values_list("user_id", "site_name", "device_id", "date"))
            group_keys = [k for k in group_keys if k not in done]
            self.stdout.write(f"After skipping already-computed: {len(group_keys)} to process")

        processed = errors = 0
        for user_id, site_name, device_id, day in group_keys:
            try:
                foraging.recompute_day(user_id, site_name, device_id, day)
                processed += 1
                if processed % 10 == 0:
                    self.stdout.write(f"  Processed {processed} groups...")
            except Exception as e:  # noqa: BLE001 - keep going, report at the end
                errors += 1
                self.stderr.write(f"  Error for {site_name} dev{device_id} {day}: {e}")

        self.stdout.write(self.style.SUCCESS(
            f"Done. Processed: {processed}, Errors: {errors}"
        ))
