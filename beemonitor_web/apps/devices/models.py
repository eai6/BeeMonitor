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
    location = models.CharField(max_length=200, blank=True)

    # Auth credential — SHA-256 hash of the raw bmk_device_* token.
    key_hash = models.CharField(max_length=128, unique=True)
    # First N chars of the raw key, shown in lists/admin for identification.
    prefix = models.CharField(max_length=20)

    is_active = models.BooleanField(default=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Last known GPS fix (from the modem's GNSS, reported in telemetry). Always
    # updated when a beat carries a fix, regardless of per-heartbeat storage.
    last_lat = models.FloatField(null=True, blank=True)
    last_lon = models.FloatField(null=True, blank=True)
    last_fix_at = models.DateTimeField(null=True, blank=True)

    # Pending command for the device, returned in the next heartbeat response and
    # then cleared. "" | "capture_image" | "stream" (picture / live view).
    pending_command = models.CharField(max_length=32, blank=True, default="")
    command_params = models.JSONField(default=dict, blank=True)

    class Meta:
        verbose_name = "Device"
        verbose_name_plural = "Devices"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.name} ({self.prefix}…)"

    @classmethod
    def create_with_key(cls, owner, name: str, location: str = "") -> tuple["Device", str]:
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
    # Per-beat GPS (from the modem GNSS), stored only when
    # settings.DEVICE_STORE_GPS_PER_HEARTBEAT is on (location history/trail).
    lat = models.FloatField(null=True, blank=True)
    lon = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["device", "-created_at"])]

    def __str__(self) -> str:
        return f"heartbeat {self.device.name} @ {self.created_at:%Y-%m-%d %H:%M}"
