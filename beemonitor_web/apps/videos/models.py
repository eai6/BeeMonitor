import re
from datetime import datetime

from django.conf import settings
from django.db import models
from django.utils import timezone as django_timezone


class Video(models.Model):
    class Status(models.TextChoices):
        UPLOADING = "uploading", "Uploading"
        READY = "ready", "Ready"
        PROCESSING = "processing", "Processing"
        ARCHIVED = "archived", "Archived"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="videos",
    )
    source = models.ForeignKey(
        "sources.DataSource",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="videos",
    )
    title = models.CharField(max_length=300)
    azure_blob_path = models.CharField(max_length=500)
    file_size_bytes = models.BigIntegerField()
    duration_seconds = models.FloatField(null=True, blank=True)
    resolution = models.CharField(max_length=20, blank=True)
    fps = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADING,
    )
    metadata = models.JSONField(default=dict, blank=True)

    # Temporal organization fields
    recorded_at = models.DateTimeField(null=True, blank=True, help_text="When the video was recorded")
    site_name = models.CharField(max_length=200, blank=True, help_text="Field site name")
    year = models.IntegerField(null=True, blank=True)
    month = models.IntegerField(null=True, blank=True)
    day = models.IntegerField(null=True, blank=True)
    hour = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"

    @staticmethod
    def parse_timestamp_from_filename(filename):
        """
        Extract datetime and site name from BeeMonitor filename format.

        Expected format: ``site_YYYY-MM-DD_HH_MM_SS.mp4``
        Also handles: ``site_YYYY-MM-DD_HH-MM-SS.mp4`` and similar variations.

        Returns a tuple ``(site_name, recorded_at)`` where ``recorded_at`` is a
        timezone-aware datetime or None if parsing fails.
        """
        # Strip directory components, keep just the filename
        basename = filename.rsplit("/", 1)[-1]
        basename = basename.rsplit("\\", 1)[-1]

        # Try pattern: site_YYYY-MM-DD_HH_MM_SS.mp4
        pattern = r'^(.+?)_(\d{4})-(\d{2})-(\d{2})_(\d{2})[_\-](\d{2})[_\-](\d{2})\.\w+$'
        match = re.match(pattern, basename)
        if match:
            site = match.group(1)
            try:
                dt = datetime(
                    year=int(match.group(2)),
                    month=int(match.group(3)),
                    day=int(match.group(4)),
                    hour=int(match.group(5)),
                    minute=int(match.group(6)),
                    second=int(match.group(7)),
                    tzinfo=django_timezone.utc,
                )
                return site, dt
            except (ValueError, OverflowError):
                pass

        # Try pattern without seconds: site_YYYY-MM-DD_HH_MM.mp4
        pattern2 = r'^(.+?)_(\d{4})-(\d{2})-(\d{2})_(\d{2})[_\-](\d{2})\.\w+$'
        match2 = re.match(pattern2, basename)
        if match2:
            site = match2.group(1)
            try:
                dt = datetime(
                    year=int(match2.group(2)),
                    month=int(match2.group(3)),
                    day=int(match2.group(4)),
                    hour=int(match2.group(5)),
                    minute=int(match2.group(6)),
                    tzinfo=django_timezone.utc,
                )
                return site, dt
            except (ValueError, OverflowError):
                pass

        return "", None

    def save(self, *args, **kwargs):
        """Auto-fill year/month/day/hour from recorded_at if available."""
        if self.recorded_at:
            self.year = self.recorded_at.year
            self.month = self.recorded_at.month
            self.day = self.recorded_at.day
            self.hour = self.recorded_at.hour
        super().save(*args, **kwargs)
