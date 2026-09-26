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
    """Set how often the device takes a full-resolution still (0 = off), or
    turn the motion burst on/off (``burst=on|off``)."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        if "burst" in request.POST:
            on = request.POST.get("burst") == "on"
            device.motion_burst_stills = device.MOTION_BURST_COUNT if on else 0
            device.save(update_fields=["motion_burst_stills"])
            messages.success(request, (
                f"On motion: {device.MOTION_BURST_COUNT} full-resolution stills, then the clip."
                if on else "On motion: video only.") + " The device adopts it on its next check-in.")
            return redirect("devices:detail", pk=pk)
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


class DeviceFreeSpaceView(LoginRequiredMixin, View):
    """Clear every uploaded video and still of this device for deletion ON THE
    DEVICE. The cloud copies stay. The device frees them on its next cleanup
    pass (two keys: it uploaded, and a person cleared it here)."""

    def post(self, request, pk):
        from apps.videos.models import Video

        device = _device_or_403(request.user, pk, "manager")
        v = (Video.objects.filter(device=device, device_delete_requested=False,
                                  device_deleted_at__isnull=True)
             .update(device_delete_requested=True))
        st = (DeviceStill.objects.filter(device=device, device_delete_requested=False,
                                         device_deleted_at__isnull=True)
              .update(device_delete_requested=True))
        messages.success(request, f"Cleared {v} video(s) and {st} still(s) for deletion on "
                                  "the device — the cloud copies are kept. It frees the "
                                  "space on its next check-in.")
        return redirect("devices:detail", pk=pk)


def on_device_counts(device) -> dict:
    """Uploaded files the device still holds, and how many are already cleared."""
    from apps.videos.models import Video

    v = Video.objects.filter(device=device, device_deleted_at__isnull=True)
    s = DeviceStill.objects.filter(device=device, device_deleted_at__isnull=True)
    return {"videos": v.count(), "stills": s.count(),
            "pending": (v.filter(device_delete_requested=True).count()
                        + s.filter(device_delete_requested=True).count())}


def _burst_clip(device, taken_at):
    """The clip a burst was followed by: the first one starting within 30 s."""
    from datetime import timedelta
    from apps.videos.models import Video

    return (Video.objects.filter(device=device,
                                 recorded_at__gte=taken_at - timedelta(seconds=5),
                                 recorded_at__lte=taken_at + timedelta(seconds=30))
            .order_by("recorded_at").first())


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
        shown = list(stills.filter(taken_at__date=day).order_by("taken_at")
                     if day else stills.none())
        # A motion burst is one row: its frames in order, then its clip.
        bursts, singles = {}, []
        for s in shown:
            if s.burst_id:
                bursts.setdefault(s.burst_id, []).append(s)
            else:
                singles.append(s)
        burst_rows = []
        for frames in bursts.values():
            frames.sort(key=lambda s: (s.burst_index or 0, s.taken_at))
            burst_rows.append({"taken_at": frames[0].taken_at,
                               "cards": [still_card(s) for s in frames],
                               "clip": _burst_clip(device, frames[0].taken_at)})
        ctx.update({
            "device": device,
            "days": days,
            "day": day,
            "bursts": burst_rows,
            "cards": [still_card(s) for s in singles],
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
