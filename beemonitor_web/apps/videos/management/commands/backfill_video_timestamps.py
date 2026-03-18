"""Backfill temporal fields on existing Video records.

Parses recorded_at, site_name, year, month, day, hour from the
video filename (azure_blob_path) or title.

Usage:
    python manage.py backfill_video_timestamps
    python manage.py backfill_video_timestamps --dry-run
"""

from django.core.management.base import BaseCommand

from apps.videos.models import Video


class Command(BaseCommand):
    help = "Backfill temporal fields (recorded_at, site_name, year, month, day, hour) on existing videos"

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be updated without saving",
        )

    def handle(self, *args, **options):
        dry_run = options["dry_run"]
        videos = Video.objects.all()
        updated = 0
        skipped = 0

        for video in videos:
            # Try parsing from the blob path filename, then from the title
            filename = video.azure_blob_path.split("/")[-1] if video.azure_blob_path else ""
            if not filename:
                filename = video.title

            site_name, recorded_at = Video.parse_timestamp_from_filename(filename)

            if not recorded_at and not site_name:
                self.stdout.write(f"  SKIP: {video.pk} — '{filename}' (no parseable timestamp)")
                skipped += 1
                continue

            changes = []
            if recorded_at and not video.recorded_at:
                video.recorded_at = recorded_at
                changes.append(f"recorded_at={recorded_at}")
            if site_name and not video.site_name:
                video.site_name = site_name
                changes.append(f"site_name={site_name}")

            # save() override fills year/month/day/hour from recorded_at
            if changes:
                if dry_run:
                    self.stdout.write(f"  WOULD UPDATE: {video.pk} — {', '.join(changes)}")
                else:
                    video.save()
                    self.stdout.write(f"  UPDATED: {video.pk} — {', '.join(changes)}")
                updated += 1
            else:
                self.stdout.write(f"  SKIP: {video.pk} — already has temporal data")
                skipped += 1

        action = "Would update" if dry_run else "Updated"
        self.stdout.write(self.style.SUCCESS(
            f"\n{action} {updated} video(s), skipped {skipped}."
        ))
