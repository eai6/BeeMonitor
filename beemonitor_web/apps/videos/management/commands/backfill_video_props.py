"""Measure fps / duration / resolution for clips ingested before we probed them.

``Video.fps`` and ``duration_seconds`` were declared from the first migration
and never written by any code path, so every frame→seconds conversion fell back
to an assumed 30 fps while the devices record at 25 — a 20% error in every
dwell time, visit length and trip duration. Ingest now measures these in the
thumbnail decode; this fills in the library that predates that.

Downloads each clip once, so run it with ``--limit`` against a large library.
"""

from django.core.management.base import BaseCommand
from django.db.models import Q

from apps.videos.models import Video


class Command(BaseCommand):
    help = "Probe stored clips for frame rate, duration and resolution."

    def add_arguments(self, parser):
        parser.add_argument("--limit", type=int, default=0,
                            help="Stop after this many clips (0 = all).")
        parser.add_argument("--device", type=str, default="",
                            help="Only clips from this device id.")
        parser.add_argument("--all", action="store_true",
                            help="Include clips that already have every property.")
        parser.add_argument("--dry-run", action="store_true",
                            help="Report what would be measured, write nothing.")

    def handle(self, *args, **opts):
        import cv2  # noqa: F401  — fail here, not per-clip, if it is missing

        qs = Video.objects.exclude(storage_key="").order_by("-recorded_at", "-uploaded_at")
        if not opts["all"]:
            qs = qs.filter(Q(fps__isnull=True) | Q(duration_seconds__isnull=True))
        if opts["device"]:
            qs = qs.filter(device_id=opts["device"])
        if opts["limit"]:
            qs = qs[:opts["limit"]]

        measured = unreadable = 0
        rates = {}
        for video in qs.iterator():
            props = self._probe(video)
            if not props.get("fps"):
                unreadable += 1
                self.stdout.write(f"  video {video.pk}: no frame rate reported")
                continue
            measured += 1
            rate = round(props["fps"])
            rates[rate] = rates.get(rate, 0) + 1
            if opts["dry_run"]:
                self.stdout.write(f"  video {video.pk}: " + ", ".join(
                    f"{k}={v}" for k, v in props.items()))
                continue
            from apps.videos.thumbnails import _store_props
            _store_props(video, props)

        if rates:
            spread = ", ".join(f"{n} clip(s) @ {r} fps" for r, n in sorted(rates.items()))
            self.stdout.write(f"Frame rates found: {spread}")
        self.stdout.write(self.style.SUCCESS(
            f"Done: {measured} measured, {unreadable} unreadable"
            + (" (dry run — nothing written)" if opts["dry_run"] else "")))

    def _probe(self, video):
        """Container properties for one clip, or {} if it cannot be read."""
        import os
        import tempfile

        import cv2

        from apps.videos.thumbnails import _grab_frame
        from config.storage import get_s3_client

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        try:
            get_s3_client().download_file("raw-videos", video.storage_key, tmp.name)
            _frame, props = _grab_frame(cv2, tmp.name)
            return props
        except Exception as exc:
            self.stderr.write(f"  video {video.pk}: {exc}")
            return {}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
