import re
from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.db import models
from django.db.models import Q


class ClipManager(models.Manager):
    """Clips only — ``Video.objects``.

    Full-resolution photos (memory/40) live in this table too, so they share
    upload, access, the Processing hub and freeing the device. But nearly every
    existing path — analysis jobs, pipelines, device schedules, annotation,
    counts and charts — treats a row as a video file. Making the default
    manager clips-only keeps photos out of all of them by default; the few
    photo-aware places ask for ``Video.everything``.
    """

    def get_queryset(self):
        return super().get_queryset().filter(kind=Video.Kind.VIDEO)


class Site(models.Model):
    """A place clips were recorded, reusable across uploads (memory/44).

    Optional everywhere: a clip may have a site, a device, both or neither.
    The location, when given, is what BioCLIP's local species list is built
    from for clips that didn't come off a unit (a unit has its own lat/lon).
    """

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="sites")
    name = models.CharField(max_length=200)
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)
    notes = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        constraints = [models.UniqueConstraint(fields=["user", "name"], name="site_unique_per_user")]

    def __str__(self):
        return self.name

    @property
    def has_location(self) -> bool:
        return self.lat is not None and self.lon is not None


# A date-time anywhere in a file name, in the shapes cameras and people use:
# 2026-10-04_09_12_40, 20261004_091240, 2026-10-04T09-12-40, 2026_10_04 09.12.40,
# IMG_20261004_091240, VID-20261004-091240. Digits must not run on either side,
# so a counter like GX010231 never reads as a date.
_ANY_TIMESTAMP = re.compile(
    r"(?<!\d)(?P<y>20\d{2})[-_.]?(?P<mo>\d{2})[-_.]?(?P<d>\d{2})"
    r"[ T_\-.]?(?P<h>\d{2})[-_.:h]?(?P<mi>\d{2})(?:[-_.:m]?(?P<s>\d{2}))?(?!\d)"
)


class Video(models.Model):
    class Kind(models.TextChoices):
        VIDEO = "video", "Video"
        PHOTO = "photo", "Photo"

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
    # Where it was recorded, when the uploader chose a saved site. site_name
    # stays the display/filter value (it is also set without a Site).
    site = models.ForeignKey(Site, on_delete=models.SET_NULL, null=True, blank=True,
                             related_name="videos")
    title = models.CharField(max_length=300)
    storage_key = models.CharField(max_length=500)
    file_size_bytes = models.BigIntegerField()
    duration_seconds = models.FloatField(null=True, blank=True)
    resolution = models.CharField(max_length=20, blank=True)
    fps = models.FloatField(null=True, blank=True)
    # Frame size in pixels, measured from the container at ingest. Needed to
    # normalise anything the worker reports in pixels — detected reference
    # boxes, most importantly — into the 0..1 space the tracks live in.
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
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

    # One sampled frame, so the review grid is scannable without playing 200
    # clips. Key into the `processed` bucket; blank until extracted (older rows,
    # or a clip whose extraction failed — the grid falls back to a placeholder).
    thumbnail_key = models.CharField(max_length=500, blank=True)

    # A clip, or a full-resolution photo from the device's camera. A photo from
    # a motion burst belongs to the clip recorded right after it (``parent``);
    # a periodic photo stands alone. Photo details (sensor mode, lens, source,
    # burst index, the device's 1280 px preview key) sit in ``metadata``.
    kind = models.CharField(max_length=8, choices=Kind.choices, default=Kind.VIDEO)
    parent = models.ForeignKey("self", on_delete=models.CASCADE, null=True, blank=True,
                               related_name="+")

    objects = ClipManager()
    everything = models.Manager()

    class Meta:
        ordering = ["-uploaded_at"]
        # Deletes and FK lookups see photos too (a clip's delete takes its
        # burst photos with it).
        base_manager_name = "everything"
        indexes = [
            models.Index(fields=["device", "kind", "recorded_at"], name="video_device_kind_rec"),
            models.Index(fields=["kind", "recorded_at"], name="video_kind_rec"),
        ]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"

    # ------------------------------------------------------------------
    # Sharing / access control
    # ------------------------------------------------------------------
    @property
    def is_photo(self) -> bool:
        return self.kind == Video.Kind.PHOTO

    def burst_photos(self):
        """The photos a motion burst took right before this clip, in order."""
        return (Video.everything.filter(parent=self, kind=Video.Kind.PHOTO)
                .order_by("recorded_at", "id"))

    @staticmethod
    def accessible(user, photos=False):
        """Videos the user owns OR can see via a device share (any role).
        ``photos=True`` includes full-resolution photos (clips only by default).

        The read-scope counterpart to ``Device.accessible(user)``: a viewer or
        manager on a device sees that device's videos (and the ecological data
        derived from them). Owner-only WRITE paths (delete, device-delete) must
        keep filtering on ``user=request.user`` and must NOT use this.
        """
        base = Video.everything if photos else Video.objects
        return base.filter(
            Q(user=user) | Q(device__shares__user=user)
        ).distinct()

    @staticmethod
    def manageable(user, photos=False):
        """Videos the user may WRITE (run analysis / delete): owned, or on a
        device shared with them as **manager** (not viewer).

        Use this for action paths (run, delete, device-delete). Viewers get
        read-only access via ``accessible`` and are excluded here.
        """
        base = Video.everything if photos else Video.objects
        return base.filter(
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

        # Anywhere in the name, any of the common shapes (no site then).
        dt = Video.find_timestamp(basename)
        if dt:
            return "", dt
        return "", None

    @staticmethod
    def find_timestamp(text):
        """The first plausible date-time in ``text`` (UTC), or None."""
        for m in _ANY_TIMESTAMP.finditer(text or ""):
            try:
                dt = datetime(int(m["y"]), int(m["mo"]), int(m["d"]), int(m["h"]),
                              int(m["mi"]), int(m["s"] or 0), tzinfo=dt_timezone.utc)
            except (ValueError, OverflowError):
                continue
            return dt
        return None

    @staticmethod
    def resolve_recorded_at(explicit, filename=""):
        """``(recorded_at, source)`` from the best available evidence.

        Three sources of very different quality used to collapse into one field
        with no record of which was used:

        - ``"device"``  — an ISO timestamp the recorder sent. Trustworthy.
        - ``"file"``    — the video's own metadata (MP4/MOV creation time),
          read in the browser at upload (``resolve_upload_recorded_at``).
        - ``"filename"`` — parsed from the clip's name. Trustworthy.
        - ``"user"``    — a start time the uploader typed for the batch.
        - ``"upload_time"`` — wall-clock at ingest. **Not a recording time.**
          A device that buffered a backlog offline and flushed it on reconnect
          stamps every clip with the flush time, so they land on the wrong day
          in every time series while looking exactly like good data.

        Callers store the source alongside the value so aggregation can treat
        ``upload_time`` as unknown rather than as fact.
        """
        from django.utils import timezone as _tz

        if explicit:
            return explicit, "device"
        if filename:
            _site, parsed = Video.parse_timestamp_from_filename(filename)
            if parsed:
                return parsed, "filename"
        return _tz.now().astimezone(dt_timezone.utc), "upload_time"

    @staticmethod
    def resolve_upload_recorded_at(file_time=None, filename="", user_time=None):
        """``(recorded_at, source)`` for a browser upload (memory/44): the
        file's own metadata, then the file name, then the uploader's start
        time, then the upload time. A file time outside 2000..now+1 day is a
        camera with an unset clock and is ignored."""
        from datetime import timedelta
        from django.utils import timezone as _tz
        from django.utils.dateparse import parse_datetime

        def _aware(value):
            dt = parse_datetime(value) if isinstance(value, str) else value
            if dt is None:
                return None
            if _tz.is_naive(dt):
                dt = dt.replace(tzinfo=dt_timezone.utc)
            return dt.astimezone(dt_timezone.utc)

        now = _tz.now()
        ft = _aware(file_time) if file_time else None
        if ft and datetime(2000, 1, 1, tzinfo=dt_timezone.utc) <= ft <= now + timedelta(days=1):
            return ft, "file"
        if filename:
            _site, parsed = Video.parse_timestamp_from_filename(filename)
            if parsed:
                return parsed, "filename"
        ut = _aware(user_time) if user_time else None
        if ut:
            return ut, "user"
        return now.astimezone(dt_timezone.utc), "upload_time"

    @property
    def recorded_at_is_measured(self) -> bool:
        """False when the timestamp is really the upload time (see above)."""
        return (self.metadata or {}).get("recorded_at_source") != "upload_time"

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
    # A photo's sidecar is freed through the stills path (still_ids), a clip's
    # through video_ids — the device keeps them apart.
    is_photo = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("device", "video_id")

    def __str__(self) -> str:
        return f"pending device-delete dev={self.device_id} video={self.video_id}"
