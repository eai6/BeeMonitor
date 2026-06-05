"""User-facing UI for managing devices.

The Django admin has the same capabilities, but admin access is restricted to
staff. End users get this app's pages.
"""

import logging
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Max
from django.db.models.functions import TruncDay, TruncHour
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView

from .forms import DeviceCreateForm
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
            device.thumb_url = _presign_image(latest.image_storage_key) if latest else None
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

        # GPS (last known fix) → coords + OpenStreetMap link.
        if device.last_lat is not None and device.last_lon is not None:
            ctx["gps_map_url"] = (
                f"https://www.openstreetmap.org/?mlat={device.last_lat}"
                f"&mlon={device.last_lon}#map=15/{device.last_lat}/{device.last_lon}"
            )

        # Image timeline — most recent stills (hourly), each presigned.
        timeline = []
        for hb in device.heartbeats.all()[:24]:
            url = _presign_image(hb.image_storage_key)
            if url:
                timeline.append({"created_at": hb.created_at, "url": url})
        ctx["timeline"] = timeline

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

        rows = (
            device.heartbeats
            .filter(created_at__gte=since, snippets_last_period__isnull=False)
            .annotate(bucket=trunc("created_at"))
            .values("bucket")
            .annotate(v=Max("snippets_last_period"))
            .order_by("bucket")
        )
        series = [{"t": r["bucket"].strftime(fmt), "v": r["v"]} for r in rows]
        return {
            "activity_series": series,
            "activity_range": range_key,
            "activity_ranges": [
                {"key": k, "label": k} for k in self._RANGES
            ],
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


class DeviceRequestStreamView(LoginRequiredMixin, View):
    """Queue a bounded live-view (rapid stills) for the device."""

    def post(self, request, pk):
        device = get_object_or_404(Device, pk=pk, owner=request.user)
        try:
            duration = min(max(int(request.POST.get("duration", 60)), 5), 300)
        except (TypeError, ValueError):
            duration = 60
        device.pending_command = "stream"
        device.command_params = {"duration": duration}
        device.save(update_fields=["pending_command", "command_params"])
        return JsonResponse({"ok": True, "duration": duration})


class DeviceLatestImageView(LoginRequiredMixin, View):
    """Latest on-demand image (presigned URL) — polled by the live view."""

    def get(self, request, pk):
        device = get_object_or_404(Device, pk=pk, owner=request.user)
        hb = device.heartbeats.exclude(image_storage_key="").first()
        if not hb:
            return JsonResponse({"url": None})
        return JsonResponse({
            "url": _presign_image(hb.image_storage_key),
            "ts": hb.created_at.isoformat(),
        })
