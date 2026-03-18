"""Mark stale processing jobs as failed.

Jobs stuck in 'processing' for over 1 hour with no modal_call_id are
from the old blocking code and will never complete.

Usage:
    python manage.py cleanup_stale_jobs
    python manage.py cleanup_stale_jobs --hours 2
"""

from datetime import timedelta

from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.analysis.models import Job


class Command(BaseCommand):
    help = "Mark stale processing/queued jobs as failed"

    def add_arguments(self, parser):
        parser.add_argument(
            "--hours", type=int, default=1,
            help="Mark jobs older than this many hours as failed (default: 1)",
        )

    def handle(self, *args, **options):
        cutoff = timezone.now() - timedelta(hours=options["hours"])

        stale = Job.objects.filter(
            status__in=[Job.Status.PROCESSING, Job.Status.QUEUED],
            started_at__lt=cutoff,
        )

        count = stale.count()
        if count == 0:
            self.stdout.write("No stale jobs found.")
            return

        stale.update(
            status=Job.Status.FAILED,
            error_message="Marked as failed: stale job from before async processing fix.",
        )

        self.stdout.write(self.style.SUCCESS(f"Marked {count} stale job(s) as failed."))
