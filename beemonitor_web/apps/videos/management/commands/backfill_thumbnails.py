"""Extract the review-grid still for clips uploaded before thumbnails existed."""

from django.core.management.base import BaseCommand

from apps.videos.models import Video
from apps.videos.thumbnails import extract_thumbnail


class Command(BaseCommand):
    help = "Sample one frame per clip for the review grid (skips clips that have one)."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=0,
                            help="Stop after this many clips (0 = all).")
        parser.add_argument("--device", type=str, default="",
                            help="Only clips from this device id.")
        parser.add_argument("--force", action="store_true",
                            help="Re-extract even where a thumbnail exists.")

    def handle(self, *args, **opts):
        qs = Video.objects.exclude(storage_key="").order_by("-recorded_at", "-uploaded_at")
        if not opts["force"]:
            qs = qs.filter(thumbnail_key="")
        if opts["device"]:
            qs = qs.filter(device_id=opts["device"])
        if opts["limit"]:
            qs = qs[:opts["limit"]]

        done = failed = 0
        for video in qs.iterator():
            if extract_thumbnail(video, force=opts["force"]):
                done += 1
            else:
                failed += 1
            if (done + failed) % 25 == 0:
                self.stdout.write(f"  {done} extracted, {failed} skipped/failed")

        self.stdout.write(self.style.SUCCESS(
            f"Done: {done} extracted, {failed} skipped or failed."))
