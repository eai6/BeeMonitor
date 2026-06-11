"""Device model — physical Pis (or any client) that upload via the API.

Each Device owns a ``bmk_device_*`` credential. Only the SHA-256 hash of the
raw key is persisted; the raw value is shown to the admin once, at creation,
and never recoverable after. Same hashing pattern as
``apps.accounts.models.APIKey``.

Every object the device uploads lands in
``s3://raw-videos/users/<owner_id>/devices/<device_id>/<yyyy>/<mm>/<dd>/<uuid>.<ext>``
so the S3 key layout itself enforces tenant scoping in Phase 5 — even a
leaked presigned URL only grants access to one device's prefix.
"""

import hashlib
import secrets

from django.conf import settings
from django.db import models
from django.db.models import Q

# Access levels, lowest -> highest. "owner" is implicit (Device.owner).
_ROLE_RANK = {"viewer": 1, "manager": 2, "owner": 3}

# Allowed telemetry beat intervals (seconds -> label) for the dashboard control.
# The device picks the new rate up via the heartbeat/command response.
TELEMETRY_INTERVAL_CHOICES = [
    (5, "5 seconds"),
    (10, "10 seconds"),
    (30, "30 seconds"),
    (60, "1 minute"),
    (300, "5 minutes"),
    (1800, "30 minutes"),
    (3600, "1 hour"),
    (86400, "1 day"),
]
TELEMETRY_INTERVAL_VALUES = [v for v, _ in TELEMETRY_INTERVAL_CHOICES]


class Device(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="devices",
    )
    name = models.CharField(
        max_length=100,
        help_text="Nickname, e.g. 'field-site-1' or 'pi-natalies-hive-2'.",
    )
    location = models.CharField(max_length=200, blank=True, help_text="Optional label, e.g. 'north hedgerow'.")
    # Manually-set deployment coordinates (decimal degrees). Set at registration
    # or via edit; shown on the dashboard with a map link.
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)

    # Auth credential — SHA-256 hash of the raw bmk_device_* token.
    key_hash = models.CharField(max_length=128, unique=True)
    # First N chars of the raw key, shown in lists/admin for identification.
    prefix = models.CharField(max_length=20)

    is_active = models.BooleanField(default=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # How often the device sends a telemetry beat (seconds). Set from the
    # dashboard; the device adopts it via the heartbeat/command response.
    telemetry_interval_seconds = models.PositiveIntegerField(default=60)

    # The device's local timezone, reported via telemetry. Activity is bucketed
    # by the hive's local clock-hour (e.g. 8-9am where the device is), so a fleet
    # spread across timezones reads correctly no matter where it's viewed from.
    tz_name = models.CharField(max_length=64, blank=True, default="")
    tz_offset_min = models.IntegerField(null=True, blank=True)

    # Manual motion-tuning overrides (null = use the device's auto-calibration).
    # Applied by the recorder on top of calibration.json. Higher var_threshold or
    # min_blobs, or a tighter area window, = less sensitive.
    motion_var_threshold = models.PositiveIntegerField(null=True, blank=True)
    motion_min_area = models.FloatField(null=True, blank=True)
    motion_max_area = models.FloatField(null=True, blank=True)
    motion_min_blobs = models.PositiveIntegerField(null=True, blank=True)

    # One-shot: auto-capture a first picture when the device first comes online
    # so the camera card isn't blank. Set once it's been requested.
    first_image_requested = models.BooleanField(default=False)

    # Pending command for the device, returned in the next heartbeat response and
    # then cleared. "" | "capture_image" | "stream" | "wifi_stream".
    pending_command = models.CharField(max_length=32, blank=True, default="")
    command_params = models.JSONField(default=dict, blank=True)

    # 5c: live MJPEG stream the device serves on its LAN (WiFi) while active.
    # The URL is a private LAN address — reachable on the same network or via
    # Raspberry Pi Connect, not from the public dashboard directly.
    stream_url = models.CharField(max_length=200, blank=True, default="")
    stream_expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        verbose_name = "Device"
        verbose_name_plural = "Devices"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.prefix}…)"

    # ------------------------------------------------------------------
    # Sharing / access control
    # ------------------------------------------------------------------
    @staticmethod
    def accessible(user):
        """Devices the user owns OR has been shared (any role)."""
        return Device.objects.filter(
            Q(owner=user) | Q(shares__user=user)
        ).distinct()

    def role_for(self, user) -> "str | None":
        """'owner' | 'manager' | 'viewer' | None for ``user`` on this device."""
        if not user or not user.is_authenticated:
            return None
        if self.owner_id == user.id:
            return "owner"
        share = self.shares.filter(user=user).first()
        return share.role if share else None

    def can(self, user, level: str) -> bool:
        """True if ``user``'s role on this device is >= ``level``."""
        role = self.role_for(user)
        return role is not None and _ROLE_RANK[role] >= _ROLE_RANK[level]

    @property
    def telemetry_interval_label(self) -> str:
        """Human label for the current beat interval (e.g. '1 minute')."""
        return dict(TELEMETRY_INTERVAL_CHOICES).get(
            self.telemetry_interval_seconds, f"{self.telemetry_interval_seconds}s")

    def motion_tuning_dict(self) -> dict:
        """Non-null motion overrides, in the keys the recorder expects."""
        d = {}
        if self.motion_var_threshold is not None:
            d["var_threshold"] = self.motion_var_threshold
        if self.motion_min_area is not None:
            d["min_area"] = self.motion_min_area
        if self.motion_max_area is not None:
            d["max_area"] = self.motion_max_area
        if self.motion_min_blobs is not None:
            d["min_blobs"] = self.motion_min_blobs
        return d

    @property
    def map_url(self) -> str:
        """OpenStreetMap link for the set coordinates, or '' if none."""
        if self.lat is not None and self.lon is not None:
            return (f"https://www.openstreetmap.org/?mlat={self.lat}"
                    f"&mlon={self.lon}#map=15/{self.lat}/{self.lon}")
        return ""

    @classmethod
    def create_with_key(cls, owner, name: str, location: str = "",
                        lat=None, lon=None) -> tuple["Device", str]:
        """Create a Device + fresh credential.

        Returns ``(device_instance, raw_key)``. The raw key is shown to the
        admin exactly once — it is never persisted in plaintext, only the
        SHA-256 hash. Same convention as ``APIKey.create_key``.
        """
        random_part = secrets.token_urlsafe(32)
        raw_key = f"bmk_device_{random_part}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        prefix = raw_key[:16]

        device = cls.objects.create(
            owner=owner,
            name=name,
            location=location,
            lat=lat,
            lon=lon,
            key_hash=key_hash,
            prefix=prefix,
        )
        return device, raw_key


class DeviceHeartbeat(models.Model):
    """A periodic health beat from a field device (telemetry + one image).

    Field units send these hourly over cellular (cheap) so we can tell the unit
    is alive without moving video. The bulk video is WiFi-gated and arrives
    separately. ``metrics`` holds the free-form payload the Pi reports (storage,
    uptime, CPU temp, service health, cellular signal, schedule window, …);
    ``image_storage_key`` points at the JPEG stored in S3 raw-videos.

    "Offline" is never stored — it's derived at view time from the age of the
    most recent beat vs the device's reported interval.
    """

    device = models.ForeignKey(
        Device,
        on_delete=models.CASCADE,
        related_name="heartbeats",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    metrics = models.JSONField(default=dict, blank=True)
    # S3 key (in the raw-videos bucket) of this beat's image, if one was sent.
    image_storage_key = models.CharField(max_length=500, blank=True)
    # Denormalised for cheap list-view sorting/highlighting.
    storage_pct = models.FloatField(null=True, blank=True)
    # Denormalised activity proxy (snippets in the trailing activity window) so
    # the activity-over-time graph can aggregate via the ORM (Max per hour/day).
    snippets_last_period = models.IntegerField(null=True, blank=True)
    # Per-beat GPS (from the modem GNSS), stored only when
    # settings.DEVICE_STORE_GPS_PER_HEARTBEAT is on (location history/trail).
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["device", "-created_at"])]

    def __str__(self) -> str:
        return f"heartbeat {self.device.name} @ {self.created_at:%Y-%m-%d %H:%M}"


class DeviceShare(models.Model):
    """Grants another account access to a device.

    The owner keeps full control (and is the only one who can manage shares).
    A *viewer* sees data only (telemetry, weather, activity, videos, on-demand
    photo); a *manager* additionally does maintenance (edit, WiFi, update,
    revoke/reactivate). Use case: a teacher gets shared access to some devices
    while the owner retains management until they choose to change it.
    """

    class Role(models.TextChoices):
        VIEWER = "viewer", "Viewer — data only"
        MANAGER = "manager", "Manager — data + control"

    device = models.ForeignKey(
        Device, on_delete=models.CASCADE, related_name="shares",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="shared_devices",
    )
    role = models.CharField(
        max_length=16, choices=Role.choices, default=Role.VIEWER,
    )
    created_at = models.DateTimeField(auto_now_add=True)
    # Who granted the share (the owner at grant time).
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="+",
    )

    class Meta:
        unique_together = ("device", "user")
        ordering = ["user__username"]

    def __str__(self) -> str:
        return f"{self.device.name} -> {self.user} ({self.role})"
