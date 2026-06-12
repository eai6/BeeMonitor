"""Run (or re-run) BioCLIP classification over existing activities.

Phase 1 classification normally fires automatically when a frame is ingested.
This backfills activities that predate the endpoint, or re-runs after a model
change. Synchronous (no thread pool) so progress + errors are visible.

  python manage.py classify_activities                 # all pending activities
  python manage.py classify_activities --status analyzed --force  # re-run all
  python manage.py classify_activities --device 3 --limit 50
"""

from django.core.management.base import BaseCommand, CommandError

from apps.monitor import pipeline
from apps.monitor.models import Activity


class Command(BaseCommand):
    help = "Classify activity frames with BioCLIP (backfill / re-run)."

    def add_arguments(self, parser):
        parser.add_argument("--device", type=int, help="Limit to one device id.")
        parser.add_argument(
            "--status", default="pending",
            help="Activity status to process (default: pending). Use 'all' for any.")
        parser.add_argument("--limit", type=int, default=0, help="Max activities (0 = no cap).")
        parser.add_argument(
            "--force", action="store_true",
            help="Re-classify frames even if they already have detections.")

    def handle(self, *args, **opts):
        if not pipeline.enabled():
            raise CommandError(
                "SAGEMAKER_BIOCLIP_ENDPOINT_NAME is not set — no endpoint to call.")

        qs = Activity.objects.all().order_by("started_at")
        if opts["device"]:
            qs = qs.filter(device_id=opts["device"])
        if opts["status"] != "all":
            valid = dict(Activity.Status.choices)
            if opts["status"] not in valid:
                raise CommandError(f"--status must be one of {list(valid)} or 'all'.")
            qs = qs.filter(status=opts["status"])
        if opts["limit"]:
            qs = qs[: opts["limit"]]

        total = frames_done = 0
        for activity in qs:
            total += 1
            frames = activity.frames.all()
            if not opts["force"]:
                frames = frames.filter(detections__isnull=True).distinct()
            for frame in frames:
                pipeline.classify_frame(frame.id)
                frames_done += 1
            self.stdout.write(f"  activity {activity.id} ({activity.device_id}) -> {activity.frames.count()} frame(s)")
        self.stdout.write(self.style.SUCCESS(
            f"Done: {total} activities, {frames_done} frames classified."))
