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
from django.utils import timezone

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


# --- WittyPi power schedule (remote review/edit) -------------------------------
# The *desired* power schedule the dashboard sets and the device reconciles to on
# its next check-in (rides the heartbeat response, like motion_tuning). The device
# is the safety authority — it validates, clamps to a guaranteed wake floor, and
# keeps the WittyPi low-voltage cutoff in every mode (see
# memory/16_remote_scheduling_design.md §5). Server validation here is shape-only.
WAKE_MODES = ["daylight", "window", "interval", "always_on"]


def default_wake_schedule() -> dict:
    """Default desired schedule (callable so the JSONField default is per-row)."""
    return {"mode": "daylight"}


def _clean_hhmm(v) -> "str | None":
    """Validate an "HH:MM" clock string; return it zero-padded, or None."""
    if not isinstance(v, str):
        return None
    parts = v.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if 0 <= h <= 23 and 0 <= m <= 59:
        return f"{h:02d}:{m:02d}"
    return None


def clean_wake_schedule(spec) -> "tuple[dict | None, str]":
    """Validate a desired wake-schedule spec from the dashboard.

    Returns ``(cleaned, "")`` on success or ``(None, error)``. This is *shape*
    validation only — the device enforces the real safety guardrails (wake floor,
    clamp, confirm-or-revert); the server is never trusted for them.
    """
    if not isinstance(spec, dict):
        return None, "Schedule must be an object."
    mode = spec.get("mode")
    if mode not in WAKE_MODES:
        return None, "Unknown schedule mode."
    out = {"mode": mode}
    if mode == "window":
        on = _clean_hhmm(spec.get("on"))
        off = _clean_hhmm(spec.get("off"))
        if on is None or off is None:
            return None, "Window needs valid on and off times (HH:MM)."
        if on == off:
            return None, "Window on and off times must differ."
        out["on"], out["off"] = on, off
    elif mode == "interval":
        try:
            every = int(spec.get("wake_every_min"))
            on_min = int(spec.get("on_minutes"))
        except (TypeError, ValueError):
            return None, "Interval needs whole minutes."
        if not (5 <= every <= 1440):
            return None, "Wake every must be 5–1440 minutes."
        if not (1 <= on_min <= every):
            return None, "On-minutes must be ≥1 and ≤ the wake interval."
        out["wake_every_min"], out["on_minutes"] = every, on_min
    # daylight / always_on carry no params.
    return out, ""


def wake_schedule_label(spec) -> str:
    """Human one-liner for a schedule spec, for the dashboard."""
    spec = spec or {}
    mode = spec.get("mode", "daylight")
    if mode == "daylight":
        return "Daylight (sun-up to sun-down)"
    if mode == "always_on":
        return "Always on (24/7)"
    if mode == "window":
        return f"On {spec.get('on', '?')}–{spec.get('off', '?')} daily"
    if mode == "interval":
        return (f"Wake every {spec.get('wake_every_min', '?')} min "
                f"for {spec.get('on_minutes', '?')} min")
    return str(mode)


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
    # User-chosen timezone the dashboard displays this device's activity in
    # (IANA, e.g. "America/New_York"). Blank = use the device-reported tz_name
    # (then UTC). Charts/tile are converted to this from each clip's true instant.
    display_tz = models.CharField(max_length=64, blank=True, default="")

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

    # ROI editor (normalized 0..1 coords, resolution-independent):
    #   roi_override = [x1, y1, x2, y2]  — the hotel/motion region, or None (auto)
    #   nest_layout  = [{"id": int, "box": [x1, y1, x2, y2]}, ...] — manual holes
    roi_override = models.JSONField(null=True, blank=True)
    nest_layout = models.JSONField(default=list, blank=True)

    # Daily cap on mover-crop uploads over cellular (BioCLIP feed). Pushed in the
    # heartbeat; the device honours it over its env default. None = use the device
    # default; 0 = stop crop upload entirely (cellular-budget kill-switch).
    frame_daily_cap = models.PositiveIntegerField(null=True, blank=True)

    # Which activities the device samples + sends BioCLIP "review" crops for (1-few
    # per activity) over cellular. Pushed in the heartbeat; the recorder hot-reloads
    # it. "all" = every activity (max data for cloud tagging, incl. unconfirmed/
    # shadow motion); "confirmed" = only on-device-confirmed bees (default, lowest
    # cellular/compute); "off" = stop sampling entirely (no SD/CPU/cellular spend).
    ACTIVITY_CROP_MODES = [
        ("all", "All activity — send every crop"),
        ("confirmed", "Confirmed only — send confirmed-bee crops"),
        ("off", "Off — don't send crops"),
    ]
    activity_crops_mode = models.CharField(
        max_length=10, default="confirmed", choices=ACTIVITY_CROP_MODES)

    # On-device YOLO bee-confirmation mode, pushed in the heartbeat (the recorder
    # hot-reloads it over its env default). "" = device default; off = no
    # confirmation; tag = observe/label without suppressing; gate = filter
    # (unconfirmed clips aren't counted as activity + crops not sent).
    BEE_CONFIRM_MODES = [
        ("", "Default (unit's env setting)"),
        ("off", "Off — track all activity"),
        ("tag", "Observe — label only"),
        ("gate", "Filter — drop unconfirmed"),
    ]
    bee_confirm_mode = models.CharField(
        max_length=8, blank=True, default="", choices=BEE_CONFIRM_MODES)

    # Hardware id (Pi serial), set during zero-touch enrollment so re-enrolling
    # the same physical unit maps back to the same Device instead of duplicating.
    hw_id = models.CharField(max_length=64, blank=True, default="", db_index=True)

    # Pending command for the device, returned in the next heartbeat response and
    # then cleared. "" | "capture_image" | "stream" | "wifi_stream".
    pending_command = models.CharField(max_length=32, blank=True, default="")
    command_params = models.JSONField(default=dict, blank=True)

    # Artifact-update target tracking. `pending_command="update"` is cleared as soon
    # as the heartbeat hands it back, so it can't drive a UI "updating" state; this
    # field persists the version a unit is being updated TO until it reports it's on
    # that version (heartbeat clears it) or the request goes stale. Lets the device
    # list show "Updating → <version>" across refreshes instead of looking idle.
    update_target = models.CharField(max_length=64, blank=True, default="")
    update_requested_at = models.DateTimeField(null=True, blank=True)

    # Desired WittyPi power schedule (review/edit from the dashboard). The device
    # reconciles to it via the heartbeat response and enforces the safety floor on
    # its side. Default {"mode": "daylight"} = current sun-window behaviour.
    wake_schedule = models.JSONField(default=default_wake_schedule, blank=True)
    # Whether the device may actually WRITE the schedule to the WittyPi. Off by
    # default: the device reports the schedule (report-only) but doesn't program
    # the WittyPi until this is turned on — a deliberate gate so a schedule can't
    # strand a unit before it's been watched on hardware. Pushed in the heartbeat.
    wake_schedule_apply = models.BooleanField(default=False)

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

    def online_window_seconds(self) -> int:
        """How long since the last check-in before this unit is 'offline'.

        Scales to the device's OWN beat cadence rather than a flat window: on
        cellular a unit only contacts the server at its beat (the between-beat poll
        is WiFi-only), so a slow-beat unit would otherwise read offline just for
        being between beats. = max(grace floor, MISSED_BEATS × its beat interval),
        so it flips offline only after missing ~MISSED_BEATS of its own beats.
        """
        interval = self.telemetry_interval_seconds or 60
        return max(settings.DEVICE_ONLINE_GRACE_SECONDS,
                   interval * settings.DEVICE_ONLINE_MISSED_BEATS)

    def is_online(self) -> bool:
        """True if the device checked in within its expected beat cadence.

        Single source of truth for online/offline — shared by the device pages and
        both AI agents so the verdict can't drift between them. Independent of
        ``is_active`` (a revoked unit can still be 'online' until it stops beating).
        """
        if not self.last_seen_at:
            return False
        age = (timezone.now() - self.last_seen_at).total_seconds()
        return age <= self.online_window_seconds()

    def wake_schedule_dict(self) -> dict:
        """The desired schedule spec, falling back to the daylight default."""
        spec = self.wake_schedule if isinstance(self.wake_schedule, dict) else None
        return spec or default_wake_schedule()

    @property
    def wake_schedule_label(self) -> str:
        """Human one-liner for the desired schedule (e.g. 'Always on (24/7)')."""
        return wake_schedule_label(self.wake_schedule_dict())

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

    def remote_config_summary(self) -> dict:
        """Human-readable snapshot of the remote config pushed to this device.

        Mirrors the fields the heartbeat sends to the unit, with the meaning of the
        'default/off' sentinels spelled out (crop cap 0 = off, report-only schedule,
        etc.) so the AI assistants can explain a unit's state without guessing.
        Single source of truth — both the monitoring agent and the device support
        assistant read this. Read-only.
        """
        cap = self.frame_daily_cap
        if cap is None:
            crop = "device default (unit's env setting)"
        elif cap == 0:
            crop = "0 — crop upload OFF (cellular kill-switch; no BioCLIP crops sent)"
        else:
            crop = f"{cap} crops/day"

        bee_mode = self.bee_confirm_mode or ""
        bee_label = dict(self.BEE_CONFIRM_MODES).get(bee_mode, bee_mode)
        nests = self.nest_layout if isinstance(self.nest_layout, list) else []

        return {
            "device_id": self.pk,
            "name": self.name,
            "active": self.is_active,
            "telemetry_beat": {
                "seconds": self.telemetry_interval_seconds,
                "label": self.telemetry_interval_label,
            },
            "bee_confirmation_mode": {"value": bee_mode or "(default)", "meaning": bee_label},
            "review_crops_over_cellular": {
                "all": "all activity (every crop sent, incl. unconfirmed)",
                "confirmed": "confirmed bees only",
                "off": "off — not sampling/sending BioCLIP review crops",
            }.get(self.activity_crops_mode, self.activity_crops_mode),
            "cellular_crop_daily_cap": crop,
            "motion_tuning": self.motion_tuning_dict() or "auto-calibration (no manual overrides)",
            "hotel_roi": self.roi_override or "auto (no manual ROI set)",
            "nest_tubes": {"count": len(nests), "layout": nests},
            "wake_schedule": {
                "spec": self.wake_schedule_dict(),
                "label": self.wake_schedule_label,
                "applied_to_hardware": self.wake_schedule_apply,
                "note": ("report-only — the device reports this schedule but does NOT "
                         "program the WittyPi until apply is enabled"
                         if not self.wake_schedule_apply else
                         "the device programs the WittyPi to this schedule"),
            },
            "display_timezone": self.display_tz or self.tz_name or "(UTC / unset)",
        }

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

    def rotate_key(self) -> str:
        """Issue a fresh credential for this device; returns the new raw key.

        Used by zero-touch re-enrollment (a re-flashed unit can't recover its old
        key, so we mint a new one). Only the hash is stored.
        """
        random_part = secrets.token_urlsafe(32)
        raw_key = f"bmk_device_{random_part}"
        self.key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        self.prefix = raw_key[:16]
        self.save(update_fields=["key_hash", "prefix"])
        return raw_key


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


class EnrollmentToken(models.Model):
    """A per-user secret baked into a field unit's SD image for zero-touch setup.

    On first boot a device with no key POSTs this token + its hardware id to
    /api/v1/devices/enroll; the server creates a Device for this user and returns
    a fresh device key — no copy-paste. Only the SHA-256 hash is stored; the raw
    ``bmk_enroll_*`` value is shown once. Revoke to stop new enrollments.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="enrollment_tokens",
    )
    label = models.CharField(max_length=100, blank=True, default="")
    key_hash = models.CharField(max_length=128, unique=True)
    prefix = models.CharField(max_length=20)
    revoked = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"enroll {self.prefix}… ({self.user})"

    @classmethod
    def create_token(cls, user, label: str = "") -> "tuple[EnrollmentToken, str]":
        raw = f"bmk_enroll_{secrets.token_urlsafe(32)}"
        tok = cls.objects.create(
            user=user, label=label,
            key_hash=hashlib.sha256(raw.encode()).hexdigest(),
            prefix=raw[:18],
        )
        return tok, raw
