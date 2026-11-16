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


EMPTY = {"roi_override": None, "roi_polygon": None, "nest_layout": [], "source": None, "source_label": ""}


def layout_for_video(video) -> dict:
    """The ROI and reference objects a clip is analysed with:
    ``{roi_override, roi_polygon, nest_layout, source, source_label}``.

    Looked up in order (memory/52): one drawn for **this clip**, then its
    **site**'s, then its **unit**'s (the version in use when it was recorded;
    devices with no history fall back to their current layout). ``source`` is
    "video" / "site" / "device", or None when there is none.
    """
    from apps.videos.models import Layout

    own = Layout.current(video=video) if getattr(video, "pk", None) else None
    if own is not None:
        return _as_dict(own, "video", f"this clip (v{own.number})")
    site = getattr(video, "site", None)
    if site is not None:
        at_site = Layout.current(site=site)
        if at_site is not None:
            return _as_dict(at_site, "site", f"site {site.name} (v{at_site.number})")
    return _device_layout(video)


def _as_dict(layout, source, label):
    return {"roi_override": layout.roi_override, "roi_polygon": layout.roi_polygon,
            "nest_layout": layout.nest_layout or [], "source": source, "source_label": label}


def _device_layout(video) -> dict:
    device = getattr(video, "device", None)
    if device is None:
        return dict(EMPTY)
    label = f"unit {device.name}"
    versions = device.layout_versions.all()
    if not versions.exists():
        if not (device.roi_override or device.nest_layout):
            return dict(EMPTY)
        return {"roi_override": device.roi_override, "roi_polygon": device.roi_polygon,
                "nest_layout": device.nest_layout or [], "source": "device", "source_label": label}
    v = (versions.filter(applied_at__isnull=False, applied_at__lte=_clip_instant(video))
         .order_by("-applied_at", "-number").first())
    if v is None:  # recorded before any layout was in use on the device
        return dict(EMPTY)
    return {"roi_override": v.roi_override, "roi_polygon": v.roi_polygon,
            "nest_layout": v.nest_layout or [], "source": "device",
            "source_label": f"{label} (v{v.number})"}
