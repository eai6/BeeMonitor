"""ROI + reference-object layout history.

The ROI editor saves the device's *current* layout on the Device (that is what
the heartbeat pushes) and also appends a read-only DeviceLayoutVersion. The
version takes effect when a heartbeat delivers it to the device. Analysis looks
up the version that was in use when a clip was recorded (layout_for_video), so
redrawing the layout later never changes how older clips are analysed.
"""

from __future__ import annotations

from datetime import timedelta

from django.db import transaction
from django.utils import timezone

from .models import DeviceLayoutVersion


def _same(v: DeviceLayoutVersion, roi, polygon, nests) -> bool:
    return (v.roi_override == roi and v.roi_polygon == polygon
            and (v.nest_layout or []) == (nests or []))


def record_saved_layout(device, user=None):
    """Append a version for the device's (just saved) layout, unless it matches
    the latest one. Returns the new version or None."""
    roi, polygon, nests = device.roi_override, device.roi_polygon, device.nest_layout or []
    with transaction.atomic():
        latest = (DeviceLayoutVersion.objects.select_for_update()
                  .filter(device=device).order_by("-number").first())
        if latest is not None and _same(latest, roi, polygon, nests):
            return None
        return DeviceLayoutVersion.objects.create(
            device=device, number=(latest.number + 1) if latest else 1,
            roi_override=roi, roi_polygon=polygon, nest_layout=nests,
            saved_by=user if getattr(user, "is_authenticated", False) else None)


def mark_delivered(device, now=None) -> None:
    """Called when a heartbeat response carries the layout to the device: the
    latest version (if not yet delivered) is in use from now."""
    latest = device.layout_versions.order_by("-number").first()
    if latest is not None and latest.applied_at is None:
        DeviceLayoutVersion.objects.filter(pk=latest.pk, applied_at__isnull=True).update(
            applied_at=now or timezone.now())


def _clip_instant(video):
    """The true UTC instant a clip was recorded. recorded_at carries the Pi's
    wall clock labelled UTC; subtracting the device's reported offset recovers
    the real instant. No recorded_at -> the upload time."""
    if video.recorded_at is None:
        return video.created_at
    off = getattr(video.device, "tz_offset_min", None) or 0
    return video.recorded_at - timedelta(minutes=off)


def layout_for_video(video) -> dict:
    """``{roi_override, roi_polygon, nest_layout}`` in use when ``video`` was
    recorded. Devices with no history fall back to their current layout."""
    device = getattr(video, "device", None)
    if device is None:
        return {"roi_override": None, "roi_polygon": None, "nest_layout": []}
    versions = device.layout_versions.all()
    if not versions.exists():
        return {"roi_override": device.roi_override, "roi_polygon": device.roi_polygon,
                "nest_layout": device.nest_layout or []}
    v = (versions.filter(applied_at__isnull=False, applied_at__lte=_clip_instant(video))
         .order_by("-applied_at", "-number").first())
    if v is None:  # recorded before any layout was in use on the device
        return {"roi_override": None, "roi_polygon": None, "nest_layout": []}
    return {"roi_override": v.roi_override, "roi_polygon": v.roi_polygon,
            "nest_layout": v.nest_layout or []}
