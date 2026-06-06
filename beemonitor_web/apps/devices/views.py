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
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.cache import cache
from django.db.models import Max
from django.db.models.functions import TruncDay, TruncHour
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView, UpdateView

from .forms import DeviceCreateForm, DeviceEditForm
from .models import Device

logger = logging.getLogger(__name__)

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
        return Device.objects.filter(owner=self.request.user)

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
        return ctx


class DeviceDetailView(LoginRequiredMixin, DetailView):
    """Per-device dashboard: latest health beat, image timeline, its videos."""

    template_name = "devices/detail.html"
    context_object_name = "device"

    def get_queryset(self):
        return Device.objects.filter(owner=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = self.object

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


        # Videos uploaded by this device (device-scoped slice of /videos/).
        ctx["videos"] = device.videos.all()[:12]
        ctx["video_count"] = device.videos.count()

        # Activity-over-time series for the chart (peak snippets/period per bucket).
        ctx.update(self._activity_series(device))
        return ctx

    # range key -> (days back, bucket granularity, label format)
    _RANGES = {
        "24h": (1, "hour", "%H:%M"),
        "7d": (7, "hour", "%b %d %H:%M"),
        "30d": (30, "day", "%b %d"),
        "90d": (90, "day", "%b %d"),
    }

    def _activity_series(self, device) -> dict:
        range_key = self.request.GET.get("range", "7d")
        if range_key not in self._RANGES:
            range_key = "7d"
        days, gran, fmt = self._RANGES[range_key]
        since = timezone.now() - timedelta(days=days)
        trunc = TruncHour if gran == "hour" else TruncDay

        rows = list(
            device.heartbeats
            .filter(created_at__gte=since, snippets_last_period__isnull=False)
            .annotate(bucket=trunc("created_at"))
            .values("bucket")
            .annotate(v=Max("snippets_last_period"))
            .order_by("bucket")
        )

        # Overlay weather (Open-Meteo) when the device has coordinates set.
        weather_enabled = device.lat is not None and device.lon is not None
        wx = {}
        if weather_enabled and rows:
            wx = _weather_lookup(
                [r["bucket"] for r in rows], gran, device.lat, device.lon)

        wkey = "%Y-%m-%dT%H" if gran == "hour" else "%Y-%m-%d"
        series = []
        for r in rows:
            b = r["bucket"]
            bu = b.astimezone(dt_timezone.utc)
            w = wx.get(bu.strftime(wkey), {})
            series.append({
                # iso is UTC; the browser formats the x-axis label in the
                # viewer's local timezone (the server runs in UTC).
                "iso": bu.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "t": b.strftime(fmt),  # fallback label if JS can't format
                "v": r["v"],
                "temp": w.get("temp"),
                "precip": w.get("precip"),
            })
        return {
            "activity_series": series,
            "activity_range": range_key,
            "activity_ranges": [
                {"key": k, "label": k} for k in self._RANGES
            ],
            "activity_gran": gran,  # "hour" | "day" — picks the label format
            "weather_enabled": weather_enabled,
        }


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
        return ctx


class DeviceEditView(LoginRequiredMixin, UpdateView):
    """Edit a device's name / location label / deployment coordinates."""

    template_name = "devices/edit.html"
    form_class = DeviceEditForm
    context_object_name = "device"

    def get_queryset(self):
        return Device.objects.filter(owner=self.request.user)

    def form_valid(self, form):
        messages.success(self.request, f"Device '{form.instance.name}' updated.")
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("devices:detail", kwargs={"pk": self.object.pk})


class DeviceRevokeView(LoginRequiredMixin, View):
    """Mark a device inactive — it can no longer authenticate."""

    def post(self, request, pk):
        device = get_object_or_404(Device, pk=pk, owner=request.user)
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
        device = get_object_or_404(Device, pk=pk, owner=request.user)
        device.is_active = True
        device.save(update_fields=["is_active"])
        messages.success(request, f"Device '{device.name}' reactivated.")
        return redirect("devices:list")


class DeviceDeleteView(LoginRequiredMixin, View):
    """Hard delete a device. Existing videos uploaded by it are preserved."""

    def post(self, request, pk):
        device = get_object_or_404(Device, pk=pk, owner=request.user)
        name = device.name
        device.delete()
        messages.success(request, f"Device '{name}' deleted.")
        return redirect("devices:list")


class DeviceRequestImageView(LoginRequiredMixin, View):
    """Queue a one-shot picture-on-demand; the device acts on its next beat."""

    def post(self, request, pk):
        device = get_object_or_404(Device, pk=pk, owner=request.user)
        device.pending_command = "capture_image"
        device.command_params = {}
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
        device = get_object_or_404(Device, pk=pk, owner=request.user)
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


class DeviceLatestImageView(LoginRequiredMixin, View):
    """Latest on-demand image (presigned URL) — polled after a photo request."""

    def get(self, request, pk):
        device = get_object_or_404(Device, pk=pk, owner=request.user)
        hb = device.heartbeats.exclude(image_storage_key="").first()
        if not hb:
            return JsonResponse({"url": None})
        return JsonResponse({
            "url": _presign_image(hb.image_storage_key),
            "ts": hb.created_at.isoformat(),
        })
