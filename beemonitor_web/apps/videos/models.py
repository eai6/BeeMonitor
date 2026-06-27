import re
from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.db import models
from django.db.models import Q


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
    # The Pi/device that uploaded this video, if any. SET_NULL so deleting a
    # device preserves its videos (matches DeviceDeleteView's promise).
    device = models.ForeignKey(
        "devices.Device",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="videos",
    )
    title = models.CharField(max_length=300)
    storage_key = models.CharField(max_length=500)
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

    # Device storage cleanup (free SD-card space on the field unit). Two-key gate:
    # a clip is removed from the device only when it has uploaded (this Video row
    # exists) AND a human explicitly requests it via the dashboard — never auto.
    # `device_delete_requested` is set by that human action; `device_deleted_at`
    # is stamped once the device confirms it removed the local file. The server's
    # own copy (S3 + this row) is unaffected — see VideoDeleteView for that.
    device_delete_requested = models.BooleanField(default=False)
    device_deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"

    # ------------------------------------------------------------------
    # Sharing / access control
    # ------------------------------------------------------------------
    @staticmethod
    def accessible(user):
        """Videos the user owns OR can see via a device share (any role).

        The read-scope counterpart to ``Device.accessible(user)``: a viewer or
        manager on a device sees that device's videos (and the ecological data
        derived from them). Owner-only WRITE paths (delete, device-delete) must
        keep filtering on ``user=request.user`` and must NOT use this.
        """
        return Video.objects.filter(
            Q(user=user) | Q(device__shares__user=user)
        ).distinct()

    @staticmethod
    def manageable(user):
        """Videos the user may WRITE (run analysis / delete): owned, or on a
        device shared with them as **manager** (not viewer).

        Use this for action paths (run, delete, device-delete). Viewers get
        read-only access via ``accessible`` and are excluded here.
        """
        return Video.objects.filter(
            Q(user=user)
            | Q(device__shares__user=user, device__shares__role="manager")
        ).distinct()

    def managed_by(self, user) -> bool:
        """True if ``user`` owns this video or is a manager of its device."""
        if self.user_id == user.id:
            return True
        if self.device_id is None:
            return False
        return self.device.shares.filter(user=user, role="manager").exists()

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
                    tzinfo=dt_timezone.utc,
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
                    tzinfo=dt_timezone.utc,
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


class PendingDeviceDeletion(models.Model):
    """Tombstone so a clip's on-device (SD) copy still gets freed after the
    cloud Video row is deleted.

    The device cleanup endpoint normally lists live ``Video`` rows the user
    cleared for device deletion. But cloud-deleting a Video removes that row, so
    if its SD copy hadn't been freed yet it would be orphaned forever. On
    cloud-delete we drop a tombstone here keyed by the *original* video id —
    which still matches the Pi's ``<clip>.uploaded`` sidecar — so the device
    frees it on the next cleanup pass. The device's confirmation deletes the
    tombstone.
    """

    device = models.ForeignKey(
        "devices.Device", on_delete=models.CASCADE, related_name="pending_deletions",
    )
    video_id = models.IntegerField()  # the now-deleted Video.id; matches the sidecar
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("device", "video_id")

    def __str__(self) -> str:
        return f"pending device-delete dev={self.device_id} video={self.video_id}"
