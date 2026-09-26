"""Full-resolution stills: the setting, "take one now", the gallery, the viewer.

The device takes a still between clips every ``stills_interval_min`` minutes
(64 MP cameras only) and uploads it like a video (memory/40). Reading them is
data (viewer role); changing the interval or taking one is control (manager).
"""

from __future__ import annotations

import logging
from datetime import datetime

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.http import Http404, JsonResponse
from django.shortcuts import redirect
from django.views import View
from django.views.generic import TemplateView

from .models import DeviceStill
from .views import _device_or_403

logger = logging.getLogger(__name__)


def _presign(key: str) -> str:
    if not key:
        return ""
    try:
        from config.storage import get_s3_client
        return get_s3_client().generate_presigned_url("raw-videos", key, expiry_hours=2)
    except Exception as e:  # the page still renders; the tile shows its time
        logger.warning("presign failed for %s: %s", key, e)
        return ""


def still_card(still: DeviceStill) -> dict:
    return {"still": still, "thumb_url": _presign(still.thumb_key or still.storage_key)}


class DeviceStillsSettingView(LoginRequiredMixin, View):
    """Set how often the device takes a full-resolution still (0 = off)."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        try:
            minutes = int(request.POST.get("interval") or 0)
        except (TypeError, ValueError):
            minutes = -1
        if minutes not in dict(device.STILLS_INTERVALS):
            messages.error(request, "Pick one of the offered intervals.")
            return redirect("devices:detail", pk=pk)
        device.stills_interval_min = minutes
        device.save(update_fields=["stills_interval_min"])
        label = dict(device.STILLS_INTERVALS)[minutes]
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": True, "interval": minutes, "label": label})
        messages.success(request, f"Full-resolution stills: {label.lower()}. The device "
                                  "adopts it on its next check-in.")
        return redirect("devices:detail", pk=pk)


class DeviceTakeStillView(LoginRequiredMixin, View):
    """Ask the device for one still now. It takes it when no clip is open and
    uploads it like a video, so it appears here once the device has WiFi."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.pending_command = "take_still"
        device.command_params = {}
        device.save(update_fields=["pending_command", "command_params"])
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": True})
        messages.success(request, "Asked the device for a still. It appears here once "
                                  "the device has uploaded it over WiFi.")
        return redirect("devices:stills", pk=pk)


class DeviceStillsView(LoginRequiredMixin, TemplateView):
    """Stills by day, newest day first."""

    template_name = "devices/stills.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = _device_or_403(self.request.user, kwargs["pk"], "viewer")
        stills = DeviceStill.objects.filter(device=device)
        days = list(stills.annotate(day=TruncDate("taken_at")).values("day")
                    .annotate(n=Count("id")).order_by("-day"))
        day = None
        raw = (self.request.GET.get("day") or "").strip()
        if raw:
            try:
                day = datetime.strptime(raw, "%Y-%m-%d").date()
            except ValueError:
                day = None
        if day is None and days:
            day = days[0]["day"]
        shown = (stills.filter(taken_at__date=day).order_by("taken_at")
                 if day else stills.none())
        ctx.update({
            "device": device,
            "days": days,
            "day": day,
            "cards": [still_card(s) for s in shown],
            "total": stills.count(),
            "can_manage": _can(self.request.user, device, "manager"),
        })
        return ctx


class DeviceStillDetailView(LoginRequiredMixin, TemplateView):
    """One still: fit or 100%, download the original, previous / next."""

    template_name = "devices/still_detail.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device = _device_or_403(self.request.user, kwargs["pk"], "viewer")
        try:
            still = DeviceStill.objects.get(device=device, pk=kwargs["still_pk"])
        except DeviceStill.DoesNotExist:
            raise Http404("No such still.")
        siblings = DeviceStill.objects.filter(device=device)
        ctx.update({
            "device": device,
            "still": still,
            "image_url": _presign(still.storage_key),
            "thumb_url": _presign(still.thumb_key or still.storage_key),
            "prev": siblings.filter(taken_at__lt=still.taken_at).order_by("-taken_at").first(),
            "next": siblings.filter(taken_at__gt=still.taken_at).order_by("taken_at").first(),
        })
        return ctx


def _can(user, device, level) -> bool:
    try:
        _device_or_403(user, device.pk, level)
        return True
    except Exception:
        return False


def recent_stills(device, n=8) -> list[dict]:
    """The device page's strip: the newest ``n`` stills."""
    return [still_card(s) for s in DeviceStill.objects.filter(device=device)[:n]]
