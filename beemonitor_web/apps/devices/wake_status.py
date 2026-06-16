"""Is a device currently inside its wake window, right now?

The dashboard knows each unit's desired wake schedule (daylight / fixed window /
interval / always-on) and its location, but the AI agents had no way to compare
"now" against that window — so when a unit was offline they could only *guess*
whether it was asleep (expected) or broken (a problem). This resolves the device's
local time and the concrete wake window for today, and answers that question.

Read-only. Shared by the monitoring agent and the device support assistant.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date as _date
from zoneinfo import ZoneInfo

from django.core.cache import cache

from .models import wake_schedule_label


def _operational_zone(device) -> "tuple[ZoneInfo, str]":
    """The device's real local timezone (the clock its WittyPi schedule runs on).

    GPS-derived wins (a field unit's own clock is often misconfigured), then the
    Pi-reported tz_name, then the user's display tz, then UTC.
    """
    name = ""
    if device.lat is not None and device.lon is not None:
        from apps.devices.views import _tz_from_coords  # lazy: avoid import cycle
        name = _tz_from_coords(device.lat, device.lon) or ""
    name = name or (device.tz_name or "").strip() or (device.display_tz or "").strip() or "UTC"
    try:
        return ZoneInfo(name), name
    except Exception:
        return ZoneInfo("UTC"), "UTC"


def _sun_times(lat, lon, day: _date) -> "dict | None":
    """Today's local sunrise/sunset 'HH:MM' for a point, via Open-Meteo. Cached 6h.

    Returns {"sunrise": "HH:MM", "sunset": "HH:MM"} or None (no GPS / fetch fail).
    """
    if lat is None or lon is None:
        return None
    key = f"suntimes:{lat:.3f}:{lon:.3f}:{day.isoformat()}"
    cached = cache.get(key)
    if cached is not None:
        return cached or None
    out = None
    try:
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "daily": "sunrise,sunset", "timezone": "auto",
            "start_date": day.isoformat(), "end_date": day.isoformat(),
        })
        with urllib.request.urlopen(url, timeout=8) as r:
            daily = json.loads(r.read().decode("utf-8")).get("daily", {})
        sr = (daily.get("sunrise") or [None])[0]
        ss = (daily.get("sunset") or [None])[0]
        if sr and ss:  # ISO like "2026-06-13T05:42" — keep the clock part
            out = {"sunrise": sr[11:16], "sunset": ss[11:16]}
    except Exception:
        out = None
    cache.set(key, out or "", 60 * 60 * 6)
    return out


def _hhmm_to_min(s: str) -> "int | None":
    try:
        h, m = s.split(":")
        return int(h) * 60 + int(m)
    except (ValueError, AttributeError):
        return None


def _inside(now_min: int, on_min: int, off_min: int) -> bool:
    """Is now within [on, off)? Handles windows that cross midnight (on > off)."""
    if on_min == off_min:
        return False
    if on_min < off_min:
        return on_min <= now_min < off_min
    return now_min >= on_min or now_min < off_min  # overnight window


def wake_window_status(device) -> dict:
    """Whether ``device`` is, right now, inside its configured wake window.

    ``currently_in_wake_window`` is True / False / "unknown" (the last when it
    can't be resolved — no GPS for a daylight schedule, or a cyclic interval whose
    phase isn't tracked server-side). For an OFFLINE unit, True-but-offline points
    at a real fault, while False explains the silence as expected sleep.
    """
    zone, tzname = _operational_zone(device)
    from django.utils import timezone as djtz
    now_local = djtz.now().astimezone(zone)
    now_min = now_local.hour * 60 + now_local.minute

    spec = device.wake_schedule_dict()
    mode = spec.get("mode", "daylight")
    out = {
        "device_id": device.pk,
        "name": device.name,
        "device_local_time": now_local.strftime("%Y-%m-%d %H:%M"),
        "timezone": tzname,
        "schedule_mode": mode,
        "schedule_label": wake_schedule_label(spec),
        "schedule_applied_to_hardware": device.wake_schedule_apply,
    }

    if mode == "always_on":
        out["window_today"] = "24/7"
        out["currently_in_wake_window"] = True
        return out

    if mode == "window":
        on, off = spec.get("on"), spec.get("off")
        on_m, off_m = _hhmm_to_min(on or ""), _hhmm_to_min(off or "")
        out["window_today"] = f"{on}–{off} (device local)"
        if on_m is None or off_m is None:
            out["currently_in_wake_window"] = "unknown"
            out["note"] = "Window times are malformed in the schedule."
        else:
            out["currently_in_wake_window"] = _inside(now_min, on_m, off_m)
        return out

    if mode == "daylight":
        sun = _sun_times(device.lat, device.lon, now_local.date())
        if not sun:
            out["currently_in_wake_window"] = "unknown"
            out["note"] = ("Daylight schedule, but sunrise/sunset couldn't be "
                           "resolved" + (" — no GPS set for this device."
                           if device.lat is None else " (weather lookup failed)."))
            return out
        on_m, off_m = _hhmm_to_min(sun["sunrise"]), _hhmm_to_min(sun["sunset"])
        out["window_today"] = f"sunrise {sun['sunrise']} – sunset {sun['sunset']} (local)"
        out["currently_in_wake_window"] = _inside(now_min, on_m, off_m)
        return out

    if mode == "interval":
        out["currently_in_wake_window"] = "unknown"
        out["note"] = ("Interval/cyclic schedule — whether it's in an ON or OFF "
                       "phase right now depends on the WittyPi's cycle anchor, "
                       "which isn't tracked server-side.")
        return out

    out["currently_in_wake_window"] = "unknown"
    return out
