"""User-facing UI for managing devices.

The Django admin has the same capabilities, but admin access is restricted to
staff. End users get this app's pages.
"""

import json
import logging
import urllib.parse
import urllib.request
from datetime import timedelta, timezone as dt_timezone

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.cache import cache
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.db.models.functions import TruncDay, TruncHour
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView, UpdateView

from .forms import DeviceCreateForm, DeviceEditForm
from .models import Device, DeviceShare

logger = logging.getLogger(__name__)


def _device_or_403(user, pk, level="viewer") -> Device:
    """Fetch a device the user can access at >= ``level``, else 404/403.

    404 if the device doesn't exist; PermissionDenied (403) if it exists but the
    user's role is below the required level. ``level`` is viewer|manager|owner.
    """
    device = get_object_or_404(Device, pk=pk)
    if not device.can(user, level):
        raise PermissionDenied("You don't have access to this device.")
    return device

def _is_online(device) -> bool:
    """Derived (not stored): has the device beaten recently enough to be 'online'?

    Telemetry beats every ~60s, so a unit that hasn't checked in within
    ``settings.DEVICE_ONLINE_GRACE_SECONDS`` (default 180 = ~3 missed beats) is
    considered offline — regardless of ``is_active``. ``last_seen_at`` is bumped
    by DeviceKeyAuthentication on every beat.
    """
    if not device.last_seen_at:
        return False
    age = (timezone.now() - device.last_seen_at).total_seconds()
    return age <= settings.DEVICE_ONLINE_GRACE_SECONDS


# Friendly text for the last USB-transfer result the device reported.
_USB_ERRORS = {
    "no_usb": "No USB drive detected — plug one into the device and try again.",
    "mount_failed": "Found a USB but couldn't mount it (unsupported filesystem?).",
    "not_writable": "USB is read-only or full.",
    "bad_target": "USB target wasn't a usable folder.",
    "no_record_dir": "Recordings folder missing on the device.",
}


def _activity_this_hour(device) -> int:
    """Snippets recorded in the device's CURRENT local clock hour.

    recorded_at carries the hive's wall-clock time (from the clip filename), so
    we shift "now" by the device's reported UTC offset to find the top of the
    hive's current hour and count from there — correct wherever the device sits.
    """
    from apps.videos.models import Video
    offset = timedelta(minutes=device.tz_offset_min or 0)
    hive_now = timezone.now() + offset
    start = hive_now.replace(minute=0, second=0, microsecond=0)
    cloud = Video.objects.filter(device=device, recorded_at__gte=start).count()
    # Prefer the device's creation-based count (clips on the card this hour) so
    # it updates immediately, not after upload. max() so we never under-report.
    latest = device.heartbeats.first()
    dev_hour = ((latest.metrics or {}) if latest else {}).get("clips_this_hour")
    if isinstance(dev_hour, (int, float)):
        return max(cloud, int(dev_hour))
    return cloud


def _videos_payload(device, limit: int = 12) -> list:
    """Recent videos for live-refreshing the device page's video list."""
    from django.template.defaultfilters import date as _date
    out = []
    for v in device.videos.all()[:limit]:
        when = (_date(v.recorded_at, "M j, Y H:i") if v.recorded_at
                else _date(v.uploaded_at, "M j, Y"))
        out.append({
            "url": reverse("videos:detail", args=[v.pk]),
            "title": v.title,
            "when": when,
            "status": v.get_status_display(),
        })
    return out


def _usb_text(usb) -> "dict | None":
    """Turn the device's raw usb-status dict into {ok, text, running?, pct?}."""
    if not usb:
        return None
    state = usb.get("state")
    if state == "running":
        pct = int(usb.get("pct") or 0)
        done, total = usb.get("done") or 0, usb.get("total") or 0
        return {"ok": True, "running": True, "pct": pct,
                "text": f"Copying to USB… {pct}% ({done}/{total})"}
    if state == "ejected":
        return {"ok": True, "text": "USB ejected — safe to remove."}
    if usb.get("ok"):
        c, s, f = usb.get("copied", 0), usb.get("skipped", 0), usb.get("failed", 0)
        h = usb.get("human") or ""
        parts = [f"copied {c}" + (f" ({h})" if h and c else "")]
        if s:
            parts.append(f"{s} already on USB")
        if f:
            parts.append(f"{f} failed")
        return {"ok": True, "text": "USB: " + ", ".join(parts)}
    detail = usb.get("detail", "")
    return {"ok": False, "text": _USB_ERRORS.get(detail, detail or "USB transfer failed.")}


def _fetch_weather(lat: float, lon: float, start_date: str, end_date: str, hourly: bool) -> dict:
    """Hourly/daily temperature + precipitation from Open-Meteo (free, no key).

    Uses the *forecast* endpoint with start/end dates: unlike the archive API it
    covers the recent past through today, which is what a live device chart needs.
    Cached for an hour (historical weather doesn't change) so we don't refetch on
    every page load. Returns {} on any failure -> the chart just omits weather.
    """
    cache_key = f"wx:{lat:.3f}:{lon:.3f}:{start_date}:{end_date}:{'h' if hourly else 'd'}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",  # matches our UTC heartbeat buckets exactly
    }
    if hourly:
        params["hourly"] = "temperature_2m,precipitation"
    else:
        params["daily"] = "temperature_2m_max,precipitation_sum"
    url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode(params)

    data: dict = {}
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # pragma: no cover - network hiccup must not 500 the page
        logger.warning("weather fetch failed for (%.3f,%.3f): %s", lat, lon, e)
        data = {}
    cache.set(cache_key, data, 60 * 60)  # 1 hour
    return data


def _weather_lookup(buckets, gran, lat, lon) -> dict:
    """Map each activity bucket (UTC) -> {"temp", "precip"} from Open-Meteo.

    Hourly granularity keys on "YYYY-MM-DDTHH"; daily on "YYYY-MM-DD". Both the
    request and the bucket keys are UTC, so they align exactly.
    """
    if not buckets:
        return {}
    utc = [b.astimezone(dt_timezone.utc) for b in buckets]
    start = min(utc).strftime("%Y-%m-%d")
    end = max(utc).strftime("%Y-%m-%d")
    hourly = gran == "hour"
    data = _fetch_weather(lat, lon, start, end, hourly=hourly)

    out: dict = {}
    if hourly:
        h = data.get("hourly", {})
        times = h.get("time", [])
        temps = h.get("temperature_2m", []) or []
        precs = h.get("precipitation", []) or []
        for i, t in enumerate(times):
            out[t[:13]] = {  # "2026-06-04T18:00" -> "2026-06-04T18"
                "temp": temps[i] if i < len(temps) else None,
                "precip": (precs[i] if i < len(precs) else None) or 0,
            }
    else:
        d = data.get("daily", {})
        times = d.get("time", [])
        tmax = d.get("temperature_2m_max", []) or []
        psum = d.get("precipitation_sum", []) or []
        for i, t in enumerate(times):
            out[t] = {
                "temp": tmax[i] if i < len(tmax) else None,
                "precip": (psum[i] if i < len(psum) else None) or 0,
            }
    return out


def _presign_image(storage_key: str):
    """Short-lived presigned GET URL for a heartbeat image, or None."""
    if not storage_key:
        return None
    try:
        from config.storage import get_s3_client
        return get_s3_client().generate_presigned_url(
            "raw-videos", storage_key, expiry_hours=1,
        )
    except Exception as e:  # pragma: no cover - S3 hiccup shouldn't 500 the page
        logger.warning("presign failed for %s: %s", storage_key, e)
        return None


class DeviceListView(LoginRequiredMixin, ListView):
    template_name = "devices/list.html"
    context_object_name = "devices"

    def get_queryset(self):
        # Owned + shared devices.
        return Device.accessible(self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        # Enrich each device with derived status + latest-beat summary for the
        # list table. Device counts are small, so per-row lookups are fine.
        for device in ctx["devices"]:
            latest = device.heartbeats.first()
            device.online = _is_online(device)
            device.latest_hb = latest
            device.storage_pct = latest.storage_pct if latest else None
            device.video_count = device.videos.count()
            device.my_role = device.role_for(self.request.user)
        return ctx


# range key -> (days back, bucket granularity, label format)
_ACTIVITY_RANGES = {
    "24h": (1, "hour", "%H:%M"),
    "7d": (7, "hour", "%b %d %H:%M"),
    "30d": (30, "day", "%b %d"),
    "90d": (90, "day", "%b %d"),
}


def _build_activity_series(device, range_key: str) -> dict:
    """Snippets recorded per real clock-hour/day, keyed off each clip's
    recorded_at (not a rolling window). Shared by the page + the live poll."""
    from django.db.models import Count
    from apps.videos.models import Video

    if range_key not in _ACTIVITY_RANGES:
        range_key = "7d"
    days, gran, fmt = _ACTIVITY_RANGES[range_key]
    since = timezone.now() - timedelta(days=days)
    trunc = TruncHour if gran == "hour" else TruncDay

    rows = list(
        Video.objects
        .filter(device=device, recorded_at__isnull=False, recorded_at__gte=since)
        .annotate(bucket=trunc("recorded_at"))
        .values("bucket")
        .annotate(v=Count("id"))
        .order_by("bucket")
    )

    # Overlay weather (Open-Meteo, cached 1h) when the device has coordinates.
    weather_enabled = device.lat is not None and device.lon is not None
    wx = {}
    if weather_enabled and rows:
        wx = _weather_lookup([r["bucket"] for r in rows], gran, device.lat, device.lon)

    wkey = "%Y-%m-%dT%H" if gran == "hour" else "%Y-%m-%d"
    counts = {}     # bucket key -> clip count
    key_dt = {}     # bucket key -> aware datetime (wall-clock, for label/iso)
    for r in rows:
        bu = r["bucket"].astimezone(dt_timezone.utc)
        k = bu.strftime(wkey)
        counts[k] = r["v"]
        key_dt[k] = bu

    # Overlay the device's creation-based per-hour histogram (clips on the card),
    # so recent hours show activity the moment a clip is recorded — before it
    # uploads. Hour-granularity ranges only; older day-ranges are fully uploaded.
    if gran == "hour":
        latest = device.heartbeats.first()
        hist = ((latest.metrics or {}) if latest else {}).get("activity_by_hour")
        if isinstance(hist, dict):
            from datetime import datetime as _dt
            floor = since - timedelta(hours=24)  # generous (tz offset slack)
            for k, c in hist.items():
                try:
                    dt = _dt.strptime(k, "%Y-%m-%dT%H").replace(tzinfo=dt_timezone.utc)
                except (ValueError, TypeError):
                    continue
                if dt < floor:
                    continue
                counts[k] = max(counts.get(k, 0), int(c))  # device ≥ cloud while on card
                key_dt.setdefault(k, dt)

    series = []
    for k in sorted(counts, key=lambda kk: key_dt[kk]):
        bu = key_dt[k]
        w = wx.get(bu.strftime(wkey), {})
        series.append({
            # iso is the bucket's wall-clock (device-local); the browser shows
            # the server label `t` as-is (see chart JS), so it isn't re-shifted.
            "iso": bu.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "t": bu.strftime(fmt),
            "v": counts[k],
            "temp": w.get("temp"),
            "precip": w.get("precip"),
        })
    return {
        "activity_series": series,
        "activity_range": range_key,
        "activity_ranges": [{"key": k, "label": k} for k in _ACTIVITY_RANGES],
        "activity_gran": gran,
        "weather_enabled": weather_enabled,
    }


class DeviceDetailView(LoginRequiredMixin, DetailView):
    """Per-device dashboard: latest health beat, image timeline, its videos."""

    template_name = "devices/detail.html"
    context_object_name = "device"

    def get_queryset(self):
        return Device.accessible(self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = self.object

        # Role-based UI gating: viewers see data only; managers + owner see
        # the maintenance controls.
        role = device.role_for(self.request.user)
        ctx["role"] = role
        ctx["is_owner"] = role == "owner"
        ctx["can_manage"] = device.can(self.request.user, "manager")

        latest = device.heartbeats.first()
        ctx["online"] = _is_online(device)
        ctx["latest_hb"] = latest
        metrics = (latest.metrics if latest else {}) or {}
        ctx["metrics"] = metrics
        ctx["latest_image_url"] = _presign_image(latest.image_storage_key) if latest else None
        sp = metrics.get("storage_pct")
        if sp is None and latest is not None:
            sp = latest.storage_pct
        ctx["storage_pct"] = sp
        ctx["services"] = [
            {"label": "Recorder", "ok": bool(metrics.get("recorder_active"))},
            {"label": "Uploader", "ok": bool(metrics.get("uploader_active"))},
            {"label": "Cellular", "ok": bool(metrics.get("cellular_active"))},
        ]
        # Telemetry rate control (manager+) + which link the last beat rode.
        from .models import TELEMETRY_INTERVAL_CHOICES
        ctx["telemetry_interval_choices"] = TELEMETRY_INTERVAL_CHOICES
        ctx["active_transport"] = metrics.get("active_transport")
        ctx["usb_status"] = _usb_text(metrics.get("usb"))  # last USB result (or None)
        ctx["usb_present"] = metrics.get("usb_present")     # USB plugged in now?
        ctx["activity_hour"] = _activity_this_hour(device)  # clips in this clock hour
        ctx["tz_name"] = device.tz_name


        # Videos uploaded by this device (device-scoped slice of /videos/).
        ctx["videos"] = device.videos.all()[:12]
        ctx["video_count"] = device.videos.count()

        # Activity-over-time series for the chart (actual snippets per bucket).
        ctx.update(_build_activity_series(
            device, self.request.GET.get("range", "7d")))
        return ctx


class DeviceCreateView(LoginRequiredMixin, FormView):
    """Create a device + show the raw key once on the next page.

    The raw bmk_device_* value is passed via the session (one-shot, popped on
    read) rather than a query string, so it doesn't end up in browser history
    or server logs.
    """

    template_name = "devices/create.html"
    form_class = DeviceCreateForm

    def form_valid(self, form):
        device, raw_key = Device.create_with_key(
            owner=self.request.user,
            name=form.cleaned_data["name"],
            location=form.cleaned_data.get("location", ""),
            lat=form.cleaned_data.get("lat"),
            lon=form.cleaned_data.get("lon"),
        )
        # Stash for the one-shot "created" page.
        self.request.session[f"device_key:{device.pk}"] = raw_key
        # Also keep it available (peeked, not popped) for the guided setup
        # walkthrough so its command blocks can pre-fill the real key.
        self.request.session[f"setup_key:{device.pk}"] = raw_key
        logger.info("device created: user=%s name=%s id=%s",
                    self.request.user.pk, device.name, device.pk)
        return redirect("devices:created", pk=device.pk)


class DeviceCreatedView(LoginRequiredMixin, TemplateView):
    """One-time view that shows the raw key after creation.

    Reads + pops the raw key from session on first GET. Refresh / re-visit
    won't show it again — the user has to either save it now or revoke the
    device and create a new one.
    """

    template_name = "devices/created.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = get_object_or_404(
            Device, pk=kwargs["pk"], owner=self.request.user,
        )
        raw_key = self.request.session.pop(f"device_key:{device.pk}", None)
        ctx["device"] = device
        ctx["raw_key"] = raw_key  # None on refresh — template handles that.
        ctx["api_base"] = settings.BEEMONITOR_DEVICE_API_BASE
        return ctx


class DeviceEditView(LoginRequiredMixin, UpdateView):
    """Edit a device's name / location label / deployment coordinates."""

    template_name = "devices/edit.html"
    form_class = DeviceEditForm
    context_object_name = "device"

    def get_object(self, queryset=None):
        # Managers + owner can edit; only the owner sees sharing/delete (gated
        # in the template via is_owner).
        return _device_or_403(self.request.user, self.kwargs["pk"], "manager")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = self.object
        ctx["is_owner"] = device.owner_id == self.request.user.id
        if ctx["is_owner"]:
            ctx["shares"] = device.shares.select_related("user").all()
            ctx["share_roles"] = DeviceShare.Role.choices
        return ctx

    def form_valid(self, form):
        messages.success(self.request, f"Device '{form.instance.name}' updated.")
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("devices:detail", kwargs={"pk": self.object.pk})


class DeviceRevokeView(LoginRequiredMixin, View):
    """Mark a device inactive — it can no longer authenticate."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.is_active = False
        device.save(update_fields=["is_active"])
        messages.success(
            request,
            f"Device '{device.name}' revoked. Its key can no longer authenticate.",
        )
        return redirect("devices:list")


class DeviceReactivateView(LoginRequiredMixin, View):
    """Un-revoke a device (re-enable an old key)."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.is_active = True
        device.save(update_fields=["is_active"])
        messages.success(request, f"Device '{device.name}' reactivated.")
        return redirect("devices:list")


class DeviceDeleteView(LoginRequiredMixin, View):
    """Hard delete a device. Existing videos uploaded by it are preserved."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "owner")
        name = device.name
        device.delete()
        messages.success(request, f"Device '{name}' deleted.")
        return redirect("devices:list")


class DeviceRequestImageView(LoginRequiredMixin, View):
    """Queue a one-shot picture-on-demand; the device acts on its next beat."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "viewer")  # on-demand photo is data, not control
        # Optional debug overlay: motion ROI + nest-hole detections on the still.
        roi = request.POST.get("roi") in ("1", "true", "on")
        device.pending_command = "capture_image"
        device.command_params = {"roi": True} if roi else {}
        device.save(update_fields=["pending_command", "command_params"])
        return JsonResponse({"ok": True, "eta_seconds": settings.DEVICE_ONLINE_GRACE_SECONDS})


class DeviceWifiView(LoginRequiredMixin, View):
    """Queue a WiFi control command (on / off / connect / forget).

    Rides the same cellular command channel as picture-on-demand, so it reaches
    the device even while WiFi is off — the Pi acts on its next command poll.
    """

    VALID = {"wifi_on", "wifi_off", "wifi_connect", "wifi_forget"}
    LABELS = {
        "wifi_on": "Turn WiFi on",
        "wifi_off": "Turn WiFi off",
        "wifi_connect": "Connect WiFi",
        "wifi_forget": "Forget network",
    }

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        action = request.POST.get("action", "")
        if action not in self.VALID:
            messages.error(request, "Unknown WiFi action.")
            return redirect("devices:detail", pk=pk)

        params: dict = {}
        if action in ("wifi_connect", "wifi_forget"):
            ssid = (request.POST.get("ssid") or "").strip()
            if not ssid:
                messages.error(request, "A network name (SSID) is required.")
                return redirect("devices:detail", pk=pk)
            params["ssid"] = ssid
            if action == "wifi_connect":
                # Stored briefly in command_params, then cleared once the device
                # picks it up. Never written to the server logs.
                params["password"] = request.POST.get("password") or ""

        device.pending_command = action
        device.command_params = params
        device.save(update_fields=["pending_command", "command_params"])

        label = self.LABELS[action]
        if action == "wifi_connect":
            label = f"Connect to '{params['ssid']}'"
        elif action == "wifi_forget":
            label = f"Forget '{params['ssid']}'"
        messages.success(
            request,
            f"{label} queued — the device will act on its next check-in (within ~1 min).",
        )
        return redirect("devices:detail", pk=pk)


class DeviceUpdateView(LoginRequiredMixin, View):
    """Queue a remote software-update command.

    Rides the same cellular command channel as the other commands. On the device,
    a two-phase updater fetches in telemetry's firewall-allowed cgroup, then a
    separate unit restarts the services, health-checks, and rolls back to the
    previous commit if they don't come up. See hardware/update.sh. The deployed
    commit + last-update result come back in the heartbeat (metrics.code_commit /
    metrics.update) and are shown on the device page.
    """

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        ref = (request.POST.get("ref") or "origin/main").strip() or "origin/main"
        device.pending_command = "update"
        device.command_params = {"ref": ref}
        device.save(update_fields=["pending_command", "command_params"])
        messages.success(
            request,
            f"Software update to '{ref}' queued — the device will fetch, restart, and "
            "auto-roll-back if unhealthy. This can take a few minutes over cellular; "
            "watch the version below.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceUsbTransferView(LoginRequiredMixin, View):
    """Queue an on-demand copy of new recordings to a plugged-in USB drive.

    The device runs usb-transfer.sh on its next command poll, which copies new
    clips to whatever USB is connected and marks them with a .usb sidecar. A USB
    drive must be physically plugged into the Pi.
    """

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.pending_command = "usb_transfer"
        device.command_params = {}
        device.save(update_fields=["pending_command", "command_params"])
        messages.success(
            request,
            "USB copy queued — make sure a USB drive is plugged into the device. "
            "It copies new clips on its next check-in; watch the progress bar, "
            "then Eject before removing the stick.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceUsbEjectView(LoginRequiredMixin, View):
    """Queue a safe-eject (unmount) of the device's plugged-in USB drive."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.pending_command = "usb_eject"
        device.command_params = {}
        device.save(update_fields=["pending_command", "command_params"])
        messages.success(
            request,
            "Eject queued — the device will unmount the USB on its next check-in. "
            "Wait for 'USB ejected — safe to remove' before pulling it out.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceTelemetryRateView(LoginRequiredMixin, View):
    """Set how often the device sends a telemetry beat.

    Validated against the allowed presets; the device adopts the new rate via
    its next heartbeat/command response (within ~COMMAND_POLL_SECONDS).
    """

    def post(self, request, pk):
        from .models import TELEMETRY_INTERVAL_VALUES
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        try:
            secs = int(request.POST.get("interval", ""))
        except (TypeError, ValueError):
            secs = None
        if secs not in TELEMETRY_INTERVAL_VALUES:
            if is_ajax:
                return JsonResponse({"error": "Invalid telemetry rate."}, status=400)
            messages.error(request, "Invalid telemetry rate.")
            return redirect("devices:detail", pk=pk)
        device.telemetry_interval_seconds = secs
        device.save(update_fields=["telemetry_interval_seconds"])
        if is_ajax:
            return JsonResponse({
                "ok": True, "interval": secs,
                "label": device.telemetry_interval_label,
            })
        messages.success(
            request,
            f"Telemetry rate set to {device.telemetry_interval_label}. The device "
            "will adopt it on its next check-in.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceStatusView(LoginRequiredMixin, View):
    """Live telemetry snapshot, polled by the device page so it refreshes in
    place (online, last seen, storage, services, WiFi, transport, activity) —
    no manual page reload needed."""

    def get(self, request, pk):
        from django.utils.timesince import timesince
        device = _device_or_403(request.user, pk, "viewer")
        latest = device.heartbeats.first()
        metrics = (latest.metrics if latest else {}) or {}
        sp = metrics.get("storage_pct")
        if sp is None and latest is not None:
            sp = latest.storage_pct
        last_seen = (timesince(device.last_seen_at) + " ago"
                     if device.last_seen_at else "never")
        activity = _build_activity_series(device, request.GET.get("range", "7d"))
        return JsonResponse({
            "online": _is_online(device),
            "last_seen": last_seen,
            "storage_pct": sp,
            "storage_free_human": metrics.get("storage_free_human"),
            "recordings_human": metrics.get("recordings_human"),
            "videos_recorded": metrics.get("videos_recorded"),
            "usb_transferred": metrics.get("usb_transferred"),
            "snippets_last_period": metrics.get("snippets_last_period"),
            "activity_hour": _activity_this_hour(device),
            "telemetry_period_human": metrics.get("telemetry_period_human"),
            "uptime_human": metrics.get("uptime_human"),
            "cpu_temp_c": metrics.get("cpu_temp_c"),
            "video_count": device.videos.count(),
            "videos": _videos_payload(device),
            "pending_uploads": metrics.get("pending_uploads"),
            "schedule_window": metrics.get("schedule_window"),
            "code_commit": metrics.get("code_commit"),
            "services": {
                "recorder": bool(metrics.get("recorder_active")),
                "uploader": bool(metrics.get("uploader_active")),
                "cellular": bool(metrics.get("cellular_active")),
            },
            "wifi_enabled": metrics.get("wifi_enabled"),
            "wifi_ssid": metrics.get("wifi_ssid"),
            "active_transport": metrics.get("active_transport"),
            "usb_status": _usb_text(metrics.get("usb")),
            "usb_present": metrics.get("usb_present"),
            "telemetry_interval_label": device.telemetry_interval_label,
            "activity_series": activity["activity_series"],
            "activity_gran": activity["activity_gran"],
            "weather_enabled": activity["weather_enabled"],
        })


class DeviceMotionTuningView(LoginRequiredMixin, View):
    """Set manual motion-tuning overrides (blank = auto-calibration).

    The device applies them on top of its calibration within a few minutes
    (hot-reload). Higher var_threshold / min_blobs or a tighter area window =
    less sensitive.
    """

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        def num(name, cast, lo, hi):
            raw = (request.POST.get(name) or "").strip()
            if raw == "":
                return None, True  # blank -> clear the override (use auto)
            try:
                v = cast(raw)
            except (TypeError, ValueError):
                return None, False
            return (v, lo <= v <= hi)

        var, ok1 = num("var_threshold", int, 4, 255)
        mn, ok2 = num("min_area", float, 0, 1_000_000)
        mx, ok3 = num("max_area", float, 0, 1_000_000_000)
        blobs, ok4 = num("min_blobs", int, 1, 100)
        if not (ok1 and ok2 and ok3 and ok4):
            err = "Invalid motion-tuning value (check the ranges)."
            if is_ajax:
                return JsonResponse({"error": err}, status=400)
            messages.error(request, err)
            return redirect("devices:detail", pk=pk)

        device.motion_var_threshold = var
        device.motion_min_area = mn
        device.motion_max_area = mx
        device.motion_min_blobs = blobs
        device.save(update_fields=["motion_var_threshold", "motion_min_area",
                                   "motion_max_area", "motion_min_blobs"])
        if is_ajax:
            return JsonResponse({"ok": True, "tuning": device.motion_tuning_dict()})
        messages.success(
            request, "Motion tuning saved — the device applies it within a few "
            "minutes (its next calibration reload).")
        return redirect("devices:detail", pk=pk)


class DeviceLatestImageView(LoginRequiredMixin, View):
    """Latest on-demand image (presigned URL) — polled after a photo request."""

    def get(self, request, pk):
        device = _device_or_403(request.user, pk, "viewer")
        hb = device.heartbeats.exclude(image_storage_key="").first()
        if not hb:
            return JsonResponse({"url": None})
        return JsonResponse({
            "url": _presign_image(hb.image_storage_key),
            "ts": hb.created_at.isoformat(),
        })


class DeviceShareAddView(LoginRequiredMixin, View):
    """Owner shares the device with another account (viewer or manager)."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "owner")
        identifier = (request.POST.get("identifier") or "").strip()
        role = request.POST.get("role", DeviceShare.Role.VIEWER)
        if role not in dict(DeviceShare.Role.choices):
            role = DeviceShare.Role.VIEWER
        if not identifier:
            messages.error(request, "Enter the person's email or username.")
            return redirect("devices:edit", pk=pk)

        User = get_user_model()
        grantee = User.objects.filter(
            Q(email__iexact=identifier) | Q(username__iexact=identifier)
        ).first()
        if not grantee:
            messages.error(
                request,
                f"No account found for '{identifier}'. Ask them to register first, "
                "then share again.",
            )
            return redirect("devices:edit", pk=pk)
        if grantee.id == device.owner_id:
            messages.error(request, "You already own this device.")
            return redirect("devices:edit", pk=pk)

        share, created = DeviceShare.objects.update_or_create(
            device=device, user=grantee,
            defaults={"role": role, "created_by": request.user},
        )
        verb = "shared with" if created else "updated for"
        messages.success(
            request,
            f"Device {verb} {grantee.username} as {share.get_role_display()}.",
        )
        return redirect("devices:edit", pk=pk)


class DeviceShareRemoveView(LoginRequiredMixin, View):
    """Owner revokes a share."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "owner")
        share = get_object_or_404(DeviceShare, pk=request.POST.get("share_id"), device=device)
        username = share.user.username
        share.delete()
        messages.success(request, f"Removed {username}'s access to '{device.name}'.")
        return redirect("devices:edit", pk=pk)
