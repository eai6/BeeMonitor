"""Full-resolution photos: the device settings, "take one now", freeing space.

Photos are rows in the videos table (``Video.kind == "photo"``, memory/40):
a motion burst's photos belong to the clip after them and show on its page;
periodic photos appear in the Processing hub beside the clips. This module is
only the device-side controls. Changing them is control (manager).
"""

from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import redirect
from django.views import View

from .views import _device_or_403


class DeviceStillsSettingView(LoginRequiredMixin, View):
    """Periodic photos every N minutes (``interval``, 0 = off), or the motion
    burst on/off (``burst=on|off``)."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        if "burst" in request.POST:
            on = request.POST.get("burst") == "on"
            device.motion_burst_stills = device.MOTION_BURST_COUNT if on else 0
            device.save(update_fields=["motion_burst_stills"])
            messages.success(request, (
                f"On motion: {device.MOTION_BURST_COUNT} full-resolution photos, then the clip."
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
        messages.success(request, f"Periodic photos: {label.lower()}. The device adopts it "
                                  "on its next check-in.")
        return redirect("devices:detail", pk=pk)


class DeviceTakeStillView(LoginRequiredMixin, View):
    """Ask the device for one photo now. It takes it when no clip is open and
    uploads it like a video; it then appears in the Processing hub's Photos."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.pending_command = "take_still"
        device.command_params = {}
        device.save(update_fields=["pending_command", "command_params"])
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": True})
        messages.success(request, "Asked the device for a photo. It appears with the "
                                  "device's videos once uploaded over WiFi.")
        return redirect("devices:detail", pk=pk)


class DeviceFreeSpaceView(LoginRequiredMixin, View):
    """Clear every uploaded video and photo of this device for deletion ON THE
    DEVICE. The cloud copies stay. The device frees them on its next cleanup
    pass (two keys: it uploaded, and a person cleared it here)."""

    def post(self, request, pk):
        from apps.videos.models import Video

        device = _device_or_403(request.user, pk, "manager")
        rows = Video.everything.filter(device=device, device_delete_requested=False,
                                       device_deleted_at__isnull=True)
        photos = rows.filter(kind=Video.Kind.PHOTO).count()
        n = rows.update(device_delete_requested=True)
        messages.success(request, f"Cleared {n - photos} video(s) and {photos} photo(s) for "
                                  "deletion on the device — the cloud copies are kept. It "
                                  "frees the space on its next check-in.")
        return redirect("devices:detail", pk=pk)


def on_device_counts(device) -> dict:
    """Uploaded files the device still holds, and how many are already cleared."""
    from apps.videos.models import Video

    rows = Video.everything.filter(device=device, device_deleted_at__isnull=True)
    return {"videos": rows.filter(kind=Video.Kind.VIDEO).count(),
            "stills": rows.filter(kind=Video.Kind.PHOTO).count(),
            "pending": rows.filter(device_delete_requested=True).count()}
