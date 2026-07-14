"""User-facing UI for managing devices.

The Django admin has the same capabilities, but admin access is restricted to
staff. End users get this app's pages.
"""

import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.cache import cache
from django.core.exceptions import PermissionDenied
from django.db.models import Q
from django.http import Http404, JsonResponse
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
    """Online iff the device checked in within its expected beat cadence.

    Thin wrapper over ``Device.is_online()`` (the single source of truth): the
    window scales to the unit's own telemetry interval, so a slow-beat cellular
    unit isn't shown offline just for being between beats. ``last_seen_at`` is
    bumped by DeviceKeyAuthentication on every beat.
    """
    return device.is_online()


def _schedule_state(device, metrics) -> str:
    """Reconcile state for the dashboard schedule badge:

      'report_only' — apply is OFF: the device reports the schedule but won't
                      program the WittyPi (so it never "confirms" — by design).
      'active'      — apply ON and the device confirms it's running the desired
                      mode (active_schedule matches).
      'pending'     — apply ON but the device hasn't confirmed yet.

    The device reports ``active_schedule`` (what it actually applied) in metrics.
    """
    if not device.wake_schedule_apply:
        return "report_only"
    active = metrics.get("active_schedule")
    if isinstance(active, dict) and active.get("mode") == device.wake_schedule_dict().get("mode"):
        return "active"
    return "pending"


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
    zone, _ = _display_zone(device)
    pi_off = timedelta(minutes=device.tz_offset_min or 0)
    now_loc = timezone.now().astimezone(zone)
    start_loc = now_loc.replace(minute=0, second=0, microsecond=0)
    # clip true_utc >= start  <=>  recorded_at - pi_off >= start  <=>  recorded_at >= start + pi_off
    thresh = start_loc.astimezone(dt_timezone.utc) + pi_off
    # Bee ACTIVITY excludes clips the on-device confirmation marked NOT a bee
    # (metadata.bee_confirmed=False). They still upload + count for storage, but
    # not as activity. Untagged clips (off-mode / pre-confirmation) still count.
    cloud = (Video.objects.filter(device=device, recorded_at__gte=thresh)
             .exclude(metadata__bee_confirmed=False).count())
    # The device's on-card count is for the Pi's clock hour; only use it when the
    # display tz currently matches the Pi's offset (else it's a different hour).
    latest = device.heartbeats.first()
    dev_hour = ((latest.metrics or {}) if latest else {}).get("clips_this_hour")
    display_off_now = int((now_loc.utcoffset() or timedelta()).total_seconds() // 60)
    if (isinstance(dev_hour, (int, float)) and device.tz_offset_min is not None
            and int(device.tz_offset_min) == display_off_now):
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


def _fetch_weather(lat: float, lon: float, start_date: str, end_date: str,
                   hourly: bool, tz: str = "auto") -> dict:
    """Hourly/daily temperature + precipitation from Open-Meteo (free, no key).

    Uses the *forecast* endpoint with start/end dates: unlike the archive API it
    covers the recent past through today, which is what a live device chart needs.
    Cached for an hour (historical weather doesn't change) so we don't refetch on
    every page load. Returns {} on any failure -> the chart just omits weather.
    """
    cache_key = f"wx:{lat:.3f}:{lon:.3f}:{start_date}:{end_date}:{'h' if hourly else 'd'}:{tz}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start_date,
        "end_date": end_date,
        # Return weather in the device's DISPLAY timezone so its hour/day keys
        # line up with the activity buckets (which are in that same tz).
        "timezone": tz or "auto",
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


def _weather_lookup(buckets, gran, lat, lon, tz="auto") -> dict:
    """Map each activity bucket -> {"temp", "precip"} from Open-Meteo.

    ``buckets`` are display-tz-local datetimes; Open-Meteo is requested in that
    same tz so its "YYYY-MM-DDTHH" / "YYYY-MM-DD" keys line up with them.
    """
    if not buckets:
        return {}
    start = min(buckets).strftime("%Y-%m-%d")
    end = max(buckets).strftime("%Y-%m-%d")
    hourly = gran == "hour"
    data = _fetch_weather(lat, lon, start, end, hourly=hourly, tz=tz)

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
    """Short-lived presigned GET URL for a heartbeat image, or None.

    Thin alias for the shared helper so the presign contract lives in one place
    (config.storage) — see also apps/monitor which presigns the same way.
    """
    from config.storage import presigned_get
    return presigned_get(storage_key)


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
        # Also collect map points (deployment coords, or the latest GPS beat as a
        # fallback) so the page can plot the fleet on a map.
        from django.utils.timesince import timesince
        latest_version = _latest_artifact_version()
        points = []
        any_manage = False
        for device in ctx["devices"]:
            latest = device.heartbeats.first()
            device.online = _is_online(device)
            device.latest_hb = latest
            device.storage_pct = latest.storage_pct if latest else None
            device.video_count = device.videos.count()
            device.my_role = device.role_for(self.request.user)
            device.can_manage = device.my_role in ("owner", "manager")
            any_manage = any_manage or device.can_manage
            # Reported software version (artifact units report their version here)
            # + whether it's behind the latest published artifact, for the fleet
            # update filter/indicator.
            device.code_commit = (latest.metrics or {}).get("code_commit") if latest else None
            device.needs_update = bool(
                latest_version and device.code_commit and device.code_commit != latest_version)
            # An update was queued and the unit isn't on that version yet → show
            # "Updating…" so a refresh doesn't look idle. Cleared by the heartbeat
            # once the device reports the target version (or the request goes stale).
            device.updating = bool(
                device.update_target and device.code_commit != device.update_target)

            lat = device.lat if device.lat is not None else (latest.lat if latest else None)
            lon = device.lon if device.lon is not None else (latest.lon if latest else None)
            if lat is not None and lon is not None:
                points.append({
                    "name": device.name,
                    "lat": lat,
                    "lon": lon,
                    "online": bool(device.is_active and device.online),
                    "active": bool(device.is_active),
                    "location": device.location or "",
                    "last_seen": (f"{timesince(device.last_seen_at)} ago"
                                  if device.last_seen_at else "never"),
                    "url": reverse("devices:detail", args=[device.pk]),
                })
        ctx["map_points"] = json.dumps(points)
        ctx["has_map"] = bool(points)
        ctx["latest_artifact_version"] = latest_version
        ctx["can_fleet_update"] = any_manage
        return ctx


# range key -> (days back, bucket granularity, label format)
_ACTIVITY_RANGES = {
    "24h": (1, "hour", "%H:%M"),
    "7d": (7, "hour", "%b %d %H:%M"),
    "30d": (30, "day", "%b %d"),
    "90d": (90, "day", "%b %d"),
}


def _tz_from_coords(lat, lon):
    """IANA timezone for a lat/lon, via Open-Meteo (which returns it for any
    point). Cached 30 days — a location's timezone doesn't change. None on fail."""
    key = f"tzname:{lat:.3f}:{lon:.3f}"
    cached = cache.get(key)
    if cached is not None:
        return cached or None
    tz = ""
    try:
        url = "https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
            "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
            "timezone": "auto", "forecast_days": 1, "current": "temperature_2m",
        })
        with urllib.request.urlopen(url, timeout=8) as resp:
            tz = json.loads(resp.read().decode("utf-8")).get("timezone") or ""
    except Exception as e:  # pragma: no cover - network hiccup must not 500
        logger.warning("tz-from-coords failed for (%.3f,%.3f): %s", lat, lon, e)
        tz = ""
    cache.set(key, tz, 60 * 60 * 24 * 30)
    return tz or None


def _display_zone(device):
    """The timezone to render this device's activity in. Priority: the user's
    explicit display_tz > the device's GPS location > the Pi-reported tz_name >
    UTC. (GPS beats the Pi clock — a field unit's clock is often misconfigured.)
    Returns (ZoneInfo, name)."""
    name = (device.display_tz or "").strip()
    if not name and device.lat is not None and device.lon is not None:
        name = _tz_from_coords(device.lat, device.lon) or ""
    if not name:
        name = (device.tz_name or "UTC").strip() or "UTC"
    try:
        return ZoneInfo(name), name
    except (ZoneInfoNotFoundError, ValueError):
        return ZoneInfo("UTC"), "UTC"


def _true_utc(device_recorded_at_or_naive, pi_off):
    """Recover the true UTC instant from a clip's stored time.

    recorded_at (and the device histogram keys) carry the Pi's wall-clock labeled
    UTC, so subtracting the Pi's reported offset yields the real instant — works
    whether the Pi clock is local or UTC.
    """
    return device_recorded_at_or_naive - pi_off


def _parse_custom_range(start, end, zone):
    """Parse 'YYYY-MM-DD' start/end (interpreted in the display tz) into a
    (start_loc, end_loc) day-boundary window, or None if either is missing,
    unparseable, or start > end. The window is capped at 366 days to bound the
    bucket count and the weather lookup.
    """
    if not start or not end:
        return None
    try:
        sd = datetime.strptime(start, "%Y-%m-%d").date()
        ed = datetime.strptime(end, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    if sd > ed:
        return None
    start_loc = datetime(sd.year, sd.month, sd.day, 0, 0, 0, tzinfo=zone)
    end_loc = datetime(ed.year, ed.month, ed.day, 23, 59, 59, 999999, tzinfo=zone)
    if (end_loc - start_loc) > timedelta(days=366):
        start_loc = end_loc - timedelta(days=366)
    return start_loc, end_loc


def _build_activity_series(device, range_key: str, confirmed: str = "all",
                           start=None, end=None, min_sec=None, max_sec=None) -> dict:
    """Snippets per clock-hour/day in the device's DISPLAY timezone. Combines
    uploaded clips (recorded_at) with the device's on-card histogram, converting
    each to its true instant then to the display tz. Shared by page + poll.

    `confirmed` filters by the on-device bee-confirmation verdict:
      - "all"         every clip (confirmed + unconfirmed + untagged)
      - "confirmed"   only clips explicitly confirmed a bee (uploaded
                      bee_confirmed=True, matching the Processing page's "Confirmed
                      bee" filter) plus, for on-card preview, the firmware's strict
                      `confirmed_by_hour` histogram. The loose `activity_by_hour`
                      (confirmed + untagged) is NOT used here. Untagged clips fall
                      under "all" only.
      - "unconfirmed" only clips the device marked NOT a bee (bee_confirmed=False)
    """
    from apps.videos.models import Video

    if confirmed not in ("all", "confirmed", "unconfirmed"):
        confirmed = "all"
    zone, zone_name = _display_zone(device)
    pi_off = timedelta(minutes=device.tz_offset_min or 0)
    now_loc = timezone.now().astimezone(zone)

    # A custom from–to window (both dates valid) overrides the preset. Hourly
    # buckets for spans up to 14 days, daily beyond. Otherwise use the preset
    # range, which always ends at "now".
    custom = _parse_custom_range(start, end, zone)
    if custom:
        start_loc, upper_loc = custom
        span_days = (upper_loc.date() - start_loc.date()).days
        gran = "hour" if span_days <= 14 else "day"
        fmt = "%b %d %H:%M" if gran == "hour" else "%b %d"
        range_key = "custom"
    else:
        if range_key not in _ACTIVITY_RANGES:
            range_key = "7d"
        days, gran, fmt = _ACTIVITY_RANGES[range_key]
        start_loc = now_loc - timedelta(days=days)
        upper_loc = now_loc
    since = start_loc.astimezone(dt_timezone.utc) - timedelta(days=1)  # tz slack
    until = upper_loc.astimezone(dt_timezone.utc) + timedelta(days=1)

    def bucket(local_dt):
        if gran == "hour":
            b = local_dt.replace(minute=0, second=0, microsecond=0)
            return b.strftime("%Y-%m-%dT%H"), b
        b = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        return b.strftime("%Y-%m-%d"), b

    counts, key_dt = {}, {}
    # Filter uploaded clips by the device's bee-confirmation verdict. "confirmed"
    # = explicitly confirmed a bee (bee_confirmed=True), matching the Processing
    # page; "unconfirmed" = marked NOT a bee (False); "all" counts everything
    # (incl. untagged clips, which belong to neither confirmed nor unconfirmed).
    qs = Video.objects.filter(device=device, recorded_at__isnull=False,
                              recorded_at__gte=since, recorded_at__lte=until)
    if confirmed == "confirmed":
        qs = qs.filter(metadata__bee_confirmed=True)
    elif confirmed == "unconfirmed":
        qs = qs.filter(metadata__bee_confirmed=False)
    for ra in qs.values_list("recorded_at", flat=True):
        loc = _true_utc(ra, pi_off).astimezone(zone)
        if loc < start_loc or loc > upper_loc:
            continue
        k, b = bucket(loc)
        counts[k] = counts.get(k, 0) + 1
        key_dt[k] = b

    # Device on-card histogram (creation-based) for hour ranges — clips on the
    # card show before they upload. Keys are the Pi's wall-clock hour. The Pi
    # reports confirmed (`activity_by_hour`) and unconfirmed (`unconfirmed_by_hour`)
    # side by side; merge whichever the filter asks for ("all" = their per-hour sum).
    #
    # `activity_by_hour` counts confirmed + UNTAGGED clips (it only excludes the
    # `.unconfirmed` marker), so it's used for "all" but NOT for "confirmed". The
    # firmware also reports `confirmed_by_hour` — a strict subset (positively
    # confirmed bees, `.confirmed` marker) — which the "confirmed" filter uses so
    # the on-card preview is accurate. Devices on older firmware don't send it, so
    # "confirmed" then falls back to uploaded bee_confirmed=True videos only (still
    # exact, just no pre-upload preview). `unconfirmed_by_hour` is strict
    # (== bee_confirmed=False), so the "unconfirmed" filter keeps its histogram.
    if gran == "hour":
        latest = device.heartbeats.first()
        metrics = (latest.metrics or {}) if latest else {}
        conf_hist = metrics.get("activity_by_hour")
        strict_hist = metrics.get("confirmed_by_hour")
        unconf_hist = metrics.get("unconfirmed_by_hour")
        hist = {}
        if confirmed == "all" and isinstance(conf_hist, dict):
            for hk, c in conf_hist.items():
                try:
                    hist[hk] = hist.get(hk, 0) + int(c)
                except (ValueError, TypeError):
                    continue
        if confirmed == "confirmed" and isinstance(strict_hist, dict):
            for hk, c in strict_hist.items():
                try:
                    hist[hk] = hist.get(hk, 0) + int(c)
                except (ValueError, TypeError):
                    continue
        if confirmed in ("all", "unconfirmed") and isinstance(unconf_hist, dict):
            for hk, c in unconf_hist.items():
                try:
                    hist[hk] = hist.get(hk, 0) + int(c)
                except (ValueError, TypeError):
                    continue
        for hk, c in hist.items():
            try:
                naive = datetime.strptime(hk, "%Y-%m-%dT%H").replace(tzinfo=dt_timezone.utc)
            except (ValueError, TypeError):
                continue
            loc = _true_utc(naive, pi_off).astimezone(zone)
            if loc < start_loc or loc > upper_loc:
                continue
            k, b = bucket(loc)
            counts[k] = max(counts.get(k, 0), int(c))
            key_dt.setdefault(k, b)

    # Densify: emit EVERY bucket in the range, filling 0 where there were no
    # clips. An empty hour is real data (no activity), so it belongs in the chart,
    # the data table, and the CSV export — previously zero-count buckets were
    # silently dropped, leaving gaps in all three.
    step = timedelta(hours=1) if gran == "hour" else timedelta(days=1)
    _, cur = bucket(start_loc)
    while True:
        bk, bdt = bucket(cur)
        if bdt > upper_loc:
            break
        counts.setdefault(bk, 0)
        key_dt.setdefault(bk, bdt)
        cur = bdt + step

    # Weather (Open-Meteo, cached) in the SAME display tz so it lines up.
    weather_enabled = device.lat is not None and device.lon is not None
    wkey = "%Y-%m-%dT%H" if gran == "hour" else "%Y-%m-%d"
    wx = {}
    if weather_enabled and key_dt:
        wx = _weather_lookup(list(key_dt.values()), gran, device.lat, device.lon, zone_name)

    # Analyzed foraging per bucket. Entries/exits are cheap per-video DB counts
    # (JobResult). Trips are paired Exit->Entry ACROSS the device's videos on the
    # absolute timeline — because a bee usually returns in a DIFFERENT clip, so a
    # per-video trip count is ~0 and would wrongly read zero here.
    from apps.analysis.models import JobResult
    forage, seen_vid = {}, set()
    analyzed_ct = {}  # analyzed clips per bucket → is this period analyzed at all?
    jr_qs = JobResult.objects.filter(
        job__video__device=device,
        job__video__recorded_at__gte=since,
        job__video__recorded_at__lte=until)
    if confirmed == "confirmed":
        jr_qs = jr_qs.filter(job__video__metadata__bee_confirmed=True)
    elif confirmed == "unconfirmed":
        jr_qs = jr_qs.filter(job__video__metadata__bee_confirmed=False)
    jr = (jr_qs.order_by("job__video_id", "-job__id")
          .values("job__video_id", "job__video__recorded_at", "entry_count", "exit_count"))
    for row in jr:
        vid = row["job__video_id"]
        if vid in seen_vid:
            continue
        seen_vid.add(vid)  # latest job per video only
        ra = row["job__video__recorded_at"]
        if not ra:
            continue
        loc = _true_utc(ra, pi_off).astimezone(zone)
        if loc < start_loc or loc > upper_loc:
            continue
        k, _b = bucket(loc)
        f = forage.setdefault(k, {"entries": 0, "exits": 0, "trips": 0})
        f["entries"] += row["entry_count"] or 0
        f["exits"] += row["exit_count"] or 0
        analyzed_ct[k] = analyzed_ct.get(k, 0) + 1
    # Trips come from persisted DailyForagingSummary rows — a pure DB read
    # (no S3 in this request path; the reconciler sweep keeps rows fresh).
    # User-adjustable duration bounds are a read-time filter on stored trips.
    from apps.analysis import foraging as _foraging
    from apps.pipelines.aggregate import clamp_trip_bounds
    min_sec, max_sec = clamp_trip_bounds(min_sec, max_sec)
    for exit_time in _foraging.device_trips(device, since, until, min_sec, max_sec):
        loc = _true_utc(exit_time, pi_off).astimezone(zone)
        if loc < start_loc or loc > upper_loc:
            continue
        k, _b = bucket(loc)
        forage.setdefault(k, {"entries": 0, "exits": 0, "trips": 0})["trips"] += 1

    series = []
    for k in sorted(counts, key=lambda kk: key_dt[kk]):
        b = key_dt[k]
        w = wx.get(b.strftime(wkey), {})
        fg = forage.get(k, {})
        series.append({
            "iso": b.astimezone(dt_timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "t": b.strftime(fmt),  # display-tz label; the chart shows it as-is
            "v": counts[k],
            "analyzed": analyzed_ct.get(k, 0),  # analyzed clips in this bucket
            "entries": fg.get("entries", 0),
            "exits": fg.get("exits", 0),
            "trips": fg.get("trips", 0),
            "temp": w.get("temp"),
            "precip": w.get("precip"),
        })
    return {
        "activity_series": series,
        "activity_range": range_key,
        "activity_ranges": [{"key": k, "label": k} for k in _ACTIVITY_RANGES],
        "activity_gran": gran,
        "activity_confirmed": confirmed,
        "activity_confirm_filters": [
            {"key": "all", "label": "All"},
            {"key": "confirmed", "label": "Confirmed"},
            {"key": "unconfirmed", "label": "Unconfirmed"},
        ],
        "weather_enabled": weather_enabled,
        "display_tz": zone_name,
        # Echo the active custom window (blank when a preset is in use) so the
        # date inputs stay populated across reloads/polls.
        "activity_custom": bool(custom),
        "activity_start": start_loc.strftime("%Y-%m-%d") if custom else "",
        "activity_end": upper_loc.strftime("%Y-%m-%d") if custom else "",
        # Active trip-duration bounds (clamped), echoed so the inputs persist.
        "activity_min_sec": min_sec,
        "activity_max_sec": max_sec,
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
        # The auto-calibrate job's learned bee-blob-area window, reported in the
        # beat — shown read-only in the Motion tuning card for transparency.
        ctx["motion_calibration"] = metrics.get("motion_calibration")
        # Always show the most RECENT image-bearing beat (regular beats carry no
        # image), so the camera card is never blank once any picture exists.
        image_hb = device.heartbeats.exclude(image_storage_key="").first()
        ctx["image_hb"] = image_hb
        ctx["latest_image_url"] = _presign_image(image_hb.image_storage_key) if image_hb else None
        # The saved hotel ROI + nest layout, drawn (read-only) over the camera image
        # client-side — the device now sends a CLEAN frame, so the dashboard overlays.
        ctx["roi_json"] = json.dumps(device.roi_override)
        ctx["nests_json"] = json.dumps(device.nest_layout or [])
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
        ctx["bee_confirm_modes"] = device.BEE_CONFIRM_MODES  # on-device species filter
        ctx["record_modes"] = device.RECORD_MODES  # capture mode + hour window
        ctx["hours"] = list(range(24))
        # Floating support-assistant widget (read-only AI; shows only if a key is set).
        from apps.setup.assistant import client as _assistant_client
        ctx["assistant_enabled"] = _assistant_client.is_enabled()
        ctx["active_transport"] = metrics.get("active_transport")
        ctx["wifi_scan"] = metrics.get("wifi_scan") or []  # in-range networks (last scan)
        ctx["usb_status"] = _usb_text(metrics.get("usb"))  # last USB result (or None)
        ctx["usb_present"] = metrics.get("usb_present")     # USB plugged in now?
        ctx["activity_hour"] = _activity_this_hour(device)  # clips in this clock hour
        ctx["tz_name"] = device.tz_name
        ctx["common_tzs"] = COMMON_TIMEZONES

        # Power schedule (review/edit) + WittyPi battery telemetry.
        from .models import WAKE_MODES
        ctx["wake_schedule"] = device.wake_schedule_dict()
        ctx["wake_modes"] = WAKE_MODES
        ctx["active_schedule"] = metrics.get("active_schedule")
        ctx["schedule_state"] = _schedule_state(device, metrics)
        ctx["wake_schedule_apply"] = device.wake_schedule_apply
        ctx["next_boot"] = metrics.get("next_boot")
        ctx["next_shutdown"] = metrics.get("next_shutdown")
        ctx["battery_voltage"] = metrics.get("battery_voltage")
        ctx["output_current"] = metrics.get("output_current")
        ctx["power_source"] = metrics.get("power_source")
        ctx["cell_firewall"] = metrics.get("cell_firewall")


        # Videos uploaded by this device (device-scoped slice of /videos/).
        ctx["videos"] = device.videos.all()[:12]
        ctx["video_count"] = device.videos.count()

        # Activity-over-time series for the chart (actual snippets per bucket).
        # `start`/`end` (YYYY-MM-DD) define a custom window that overrides `range`.
        ctx.update(_build_activity_series(
            device, self.request.GET.get("range", "7d"),
            self.request.GET.get("confirmed", "all"),
            start=self.request.GET.get("start"),
            end=self.request.GET.get("end"),
            min_sec=self.request.GET.get("min_sec"),
            max_sec=self.request.GET.get("max_sec")))
        return ctx


def _golden_image_url() -> str:
    """URL the enrollment page links to for the golden image download.

    A direct ``BEEMONITOR_GOLDEN_IMAGE_URL`` wins (e.g. a CDN). Otherwise, if a
    key is configured and the object exists in the (private) models bucket, return
    the presigned-redirect endpoint. '' = no download button. The blob_exists check
    is cached so it doesn't hit S3 on every page load.
    """
    if settings.BEEMONITOR_GOLDEN_IMAGE_URL:
        return settings.BEEMONITOR_GOLDEN_IMAGE_URL
    key = settings.BEEMONITOR_GOLDEN_IMAGE_S3_KEY
    if not key:
        return ""
    cache_key = f"golden_image_present:{key}"
    present = cache.get(cache_key)
    if present is None:
        from config.storage import get_s3_client
        try:
            present = get_s3_client().blob_exists("models", key)
        except Exception as e:  # pragma: no cover - S3 hiccup must not 500 the page
            logger.warning("golden image blob_exists failed: %s", e)
            present = False
        cache.set(cache_key, present, 300)  # 5 min
    return reverse("devices:golden_image") if present else ""


class GoldenImageDownloadView(LoginRequiredMixin, View):
    """Redirect to a short-lived presigned URL for the golden image in S3.

    Keeps the models bucket private (no public-read): each click signs a fresh
    GET URL the browser follows straight to S3. 404 if no image is configured.
    """

    def get(self, request):
        from config.storage import presigned_get
        key = settings.BEEMONITOR_GOLDEN_IMAGE_S3_KEY
        url = presigned_get(key, container="models", expiry_hours=1) if key else None
        if not url:
            raise Http404("Golden image not available.")
        return redirect(url)


class DeviceEnrollmentView(LoginRequiredMixin, TemplateView):
    """Manage enrollment tokens for zero-touch device setup.

    Generate a token, bake it into the SD image, and units self-register on first
    boot — no per-device key copy-paste.
    """

    template_name = "devices/enrollment.html"

    def get_context_data(self, **kwargs):
        from .models import EnrollmentToken
        ctx = super().get_context_data(**kwargs)
        ctx["tokens"] = EnrollmentToken.objects.filter(user=self.request.user)
        ctx["new_token"] = self.request.session.pop("enroll_token_raw", None)
        ctx["api_base"] = settings.BEEMONITOR_DEVICE_API_BASE
        ctx["image_url"] = _golden_image_url()
        return ctx

    def post(self, request, *args, **kwargs):
        from .models import EnrollmentToken
        label = (request.POST.get("label") or "").strip()[:100]
        _, raw = EnrollmentToken.create_token(request.user, label=label)
        request.session["enroll_token_raw"] = raw  # shown once
        messages.success(request, "Enrollment token created — copy it now.")
        return redirect("devices:enrollment")


class EnrollmentTokenRevokeView(LoginRequiredMixin, View):
    """Revoke an enrollment token (stops new devices enrolling with it)."""

    def post(self, request, pk):
        from .models import EnrollmentToken
        tok = get_object_or_404(EnrollmentToken, pk=pk, user=request.user)
        tok.revoked = True
        tok.save(update_fields=["revoked"])
        messages.success(request, "Enrollment token revoked.")
        return redirect("devices:enrollment")


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
        ctx["image_url"] = _golden_image_url()  # golden image (flash)
        ctx["online"] = _is_online(device)  # so a returning page shows current state
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


class DeviceWifiScanView(LoginRequiredMixin, View):
    """Queue a scan of in-range WiFi networks. The device runs nmcli on its next
    command poll and reports the networks in its heartbeat (metrics.wifi_scan),
    which the device page surfaces so a network can be picked to connect."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        device.pending_command = "wifi_scan"
        device.command_params = {}
        device.save(update_fields=["pending_command", "command_params"])
        return JsonResponse({"ok": True})


class DeviceCellularView(LoginRequiredMixin, View):
    """Open the cellular egress firewall for remote debugging (rpi-connect), or
    re-gate it. Rides the telemetry command channel (the one thing always allowed
    on cellular), so it reaches the unit even while the gate is closed. Opening
    auto-reverts on the device after a timeout so an open metered link can't burn
    the data cap.
    """

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        action = request.POST.get("action", "")
        if action == "open":
            try:
                minutes = int(request.POST.get("minutes") or 30)
            except (TypeError, ValueError):
                minutes = 30
            minutes = max(1, min(minutes, 240))
            device.pending_command = "cellular_open"
            device.command_params = {"minutes": minutes}
            msg = (f"Opening cellular for {minutes} min on the next check-in — "
                   "rpi-connect should then reach the unit, and it auto-re-gates "
                   "after. Watch the gate state below.")
        elif action == "gate":
            device.pending_command = "cellular_gate"
            device.command_params = {}
            msg = "Re-gating cellular to telemetry-only on the next check-in."
        else:
            messages.error(request, "Unknown cellular action.")
            return redirect("devices:detail", pk=pk)
        device.save(update_fields=["pending_command", "command_params"])
        messages.success(request, msg)
        return redirect("devices:detail", pk=pk)


def _resolve_edge_descriptor() -> "dict | None":
    """Artifact-update descriptor from S3 (feature 18): read edge/latest.json + that
    version's manifest from the models bucket, presign the bundle, and return
    {version, url, sha256, sig, reqs_hash} for the heartbeat update command. None if
    nothing's published or S3 is unreachable."""
    from config.storage import presigned_get
    try:
        import boto3
        s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
        bucket = settings.AWS_S3_BUCKET_MODELS

        def _obj(key):
            return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())

        version = _obj("edge/latest.json")["version"]
        manifest = _obj(f"edge/beemonitor-edge-{version}.json")
        url = presigned_get(f"edge/beemonitor-edge-{version}.tar.gz",
                            container="models", expiry_hours=2)
        if not url:
            return None
        return {"version": version, "url": url, "sha256": manifest["sha256"],
                "sig": manifest.get("sig", ""), "reqs_hash": manifest.get("reqs_hash", "")}
    except Exception as e:  # pragma: no cover - S3/network hiccup must not 500
        logger.warning("resolve edge descriptor failed: %s", e)
        return None


def _latest_artifact_version() -> "str | None":
    """The currently-published edge artifact version (edge/latest.json), cached
    briefly so the devices list can flag out-of-date units without an S3 read per
    request. None if nothing's published / S3 is unreachable."""
    from django.core.cache import cache
    cached = cache.get("edge_latest_version")
    if cached is not None:
        return cached or None
    version = ""
    try:
        import boto3
        s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
        obj = s3.get_object(Bucket=settings.AWS_S3_BUCKET_MODELS, Key="edge/latest.json")
        version = json.loads(obj["Body"].read()).get("version", "") or ""
    except Exception as e:  # pragma: no cover - S3/network hiccup must not 500
        logger.warning("latest artifact version lookup failed: %s", e)
    cache.set("edge_latest_version", version, 60)
    return version or None


class DeviceFleetUpdateView(LoginRequiredMixin, View):
    """Queue the signed-artifact update on many devices at once.

    Resolve the published bundle descriptor ONCE, then set the same update command
    on every selected device the caller manages. Each device downloads + verifies +
    swaps (and auto-rolls-back) on its next check-in, exactly like the per-device
    update — this just fans it out so you don't go unit by unit.
    """

    def post(self, request):
        ids = request.POST.getlist("device_ids")
        if not ids and request.body:
            try:
                ids = json.loads(request.body).get("device_ids", [])
            except (ValueError, AttributeError):
                ids = []
        try:
            ids = {int(i) for i in ids}
        except (TypeError, ValueError):
            return JsonResponse({"ok": False, "error": "Invalid device selection."}, status=400)
        if not ids:
            return JsonResponse({"ok": False, "error": "No devices selected."}, status=400)

        params = _resolve_edge_descriptor()
        if not params:
            return JsonResponse(
                {"ok": False, "error": "No signed edge artifact is published yet "
                 "(tag a release / check the edge-artifact CI)."}, status=409)

        # Only act on devices the caller manages (manager+) and that are active.
        targets = [d for d in Device.accessible(request.user).filter(pk__in=ids)
                   if d.is_active and d.can(request.user, "manager")]
        now = timezone.now()
        for d in targets:
            d.pending_command = "update"
            d.command_params = params
            d.update_target = params["version"]   # persists the "updating" state
            d.update_requested_at = now
        if targets:
            Device.objects.bulk_update(
                targets, ["pending_command", "command_params",
                          "update_target", "update_requested_at"])
        return JsonResponse({
            "ok": True,
            "version": params["version"],
            "queued": len(targets),
            "skipped": len(ids) - len(targets),
        })


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
        # Artifact update (feature 18): resolve the signed bundle descriptor from
        # S3. Falls back to the git ref flow (the default) for git-layout units.
        if (request.POST.get("mode") or "").strip() == "artifact":
            params = _resolve_edge_descriptor()
            if not params:
                messages.error(request, "No signed edge artifact is published yet "
                               "(tag a release / check the edge-artifact CI).")
                return redirect("devices:detail", pk=pk)
            device.pending_command = "update"
            device.command_params = params
            device.update_target = params["version"]
            device.update_requested_at = timezone.now()
            device.save(update_fields=["pending_command", "command_params",
                                       "update_target", "update_requested_at"])
            messages.success(
                request,
                f"Artifact update to {params['version']} queued — the device will "
                "download, verify the signature, swap the release, and auto-roll-back "
                "if unhealthy. Watch the version below.",
            )
            return redirect("devices:detail", pk=pk)

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


class DeviceBeeConfirmView(LoginRequiredMixin, View):
    """Turn on-device bee confirmation on/off (or observe-only) from the dashboard.

    off  = track all activity (no model loaded);
    tag  = run the model + label clips, but still count + send everything;
    gate = filter — unconfirmed clips aren't counted and their crops aren't sent
           (the clip is still recorded + uploaded either way, so nothing is lost).
    "" = use the unit's env default. The device adopts the change within seconds.
    """

    VALID = {"", "off", "tag", "gate"}

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        mode = (request.POST.get("mode") or "").strip().lower()
        if mode not in self.VALID:
            if is_ajax:
                return JsonResponse({"error": "Invalid bee-confirmation mode."}, status=400)
            messages.error(request, "Invalid bee-confirmation mode.")
            return redirect("devices:detail", pk=pk)
        device.bee_confirm_mode = mode
        device.save(update_fields=["bee_confirm_mode"])
        label = dict(device.BEE_CONFIRM_MODES).get(mode, mode)
        if is_ajax:
            return JsonResponse({"ok": True, "mode": mode, "label": label})
        messages.success(
            request,
            f"Bee confirmation set to “{label}”. The device will adopt it on its "
            "next check-in.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceActivityCropsView(LoginRequiredMixin, View):
    """Set which activities the device sends BioCLIP "review" crops for, over cellular.

    all       = sample + upload a crop for every activity (incl. unconfirmed/shadow
                motion) — max data for cloud taxonomic tagging.
    confirmed = only activities the on-device bee-confirmer accepted (default; lowest
                cellular + compute).
    off       = stop sampling entirely on the recorder (no SD/CPU/cellular spend).
    The device adopts the change on its next check-in.
    """

    VALID = {"all", "confirmed", "off"}

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        mode = (request.POST.get("mode") or "").strip().lower()
        if mode not in self.VALID:
            if is_ajax:
                return JsonResponse({"ok": False, "error": "invalid mode"}, status=400)
            messages.error(request, "Invalid crop mode.")
            return redirect("devices:detail", pk=pk)
        device.activity_crops_mode = mode
        device.save(update_fields=["activity_crops_mode"])
        label = dict(Device.ACTIVITY_CROP_MODES).get(mode, mode)
        if is_ajax:
            return JsonResponse({"ok": True, "mode": mode, "label": label})
        messages.success(
            request,
            f"Review-crop mode set to “{label}”. The device will adopt it on its "
            "next check-in.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceScheduleView(LoginRequiredMixin, View):
    """Set the device's desired WittyPi power schedule (review/edit).

    Builds a spec from the form, validates its *shape* (the device enforces the
    real safety guardrails), and stores it on the device. The unit reconciles to
    it on its next check-in via the heartbeat response. See
    memory/16_remote_scheduling_design.md.
    """

    def post(self, request, pk):
        from .models import clean_wake_schedule
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        mode = (request.POST.get("mode") or "").strip()
        spec: dict = {"mode": mode}
        if mode == "window":
            spec["on"] = (request.POST.get("on") or "").strip()
            spec["off"] = (request.POST.get("off") or "").strip()
        elif mode == "interval":
            spec["wake_every_min"] = request.POST.get("wake_every_min")
            spec["on_minutes"] = request.POST.get("on_minutes")

        cleaned, err = clean_wake_schedule(spec)
        if err:
            if is_ajax:
                return JsonResponse({"error": err}, status=400)
            messages.error(request, err)
            return redirect("devices:detail", pk=pk)

        # Whether the device may actually write this to the WittyPi (else it just
        # reports it). The device still enforces the wake floor, so this is safe.
        apply = request.POST.get("apply") in ("1", "true", "on")
        device.wake_schedule = cleaned
        device.wake_schedule_apply = apply
        device.save(update_fields=["wake_schedule", "wake_schedule_apply"])
        if is_ajax:
            return JsonResponse({
                "ok": True,
                "wake_schedule": cleaned,
                "wake_schedule_apply": apply,
                "schedule_state": _schedule_state(device, {}),
                "label": device.wake_schedule_label,
            })
        messages.success(
            request,
            f"Power schedule set to “{device.wake_schedule_label}”. The device "
            "applies it on its next check-in (and enforces its own safety floor).",
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
        img_hb = device.heartbeats.exclude(image_storage_key="").first()
        image = ({"url": _presign_image(img_hb.image_storage_key),
                  "ts": img_hb.created_at.isoformat()} if img_hb else None)
        # The chart series is only built when asked (?chart=1): the page polls
        # this endpoint every 5s for the cheap telemetry tiles, but refreshes
        # the chart on its own slower cadence — recomputing buckets + weather
        # lookups 12×/minute per open tab was pure waste.
        activity = None
        if request.GET.get("chart") == "1":
            activity = _build_activity_series(device, request.GET.get("range", "7d"),
                                              request.GET.get("confirmed", "all"),
                                              start=request.GET.get("start"),
                                              end=request.GET.get("end"),
                                              min_sec=request.GET.get("min_sec"),
                                              max_sec=request.GET.get("max_sec"))
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
            "wifi_scan": metrics.get("wifi_scan") or [],
            "cell_firewall": metrics.get("cell_firewall"),
            "active_transport": metrics.get("active_transport"),
            "usb_status": _usb_text(metrics.get("usb")),
            "usb_present": metrics.get("usb_present"),
            # Power schedule (desired) + what the device reports it's running, plus
            # the WittyPi battery/input voltage telemetry.
            "wake_schedule_label": device.wake_schedule_label,
            "schedule_state": _schedule_state(device, metrics),
            "wake_schedule_apply": device.wake_schedule_apply,
            "next_boot": metrics.get("next_boot"),
            "next_shutdown": metrics.get("next_shutdown"),
            "battery_voltage": metrics.get("battery_voltage"),
            "output_current": metrics.get("output_current"),
            "power_source": metrics.get("power_source"),
            "image": image,
            "telemetry_interval_label": device.telemetry_interval_label,
            **({"activity_series": activity["activity_series"],
                "activity_gran": activity["activity_gran"],
                "activity_confirmed": activity["activity_confirmed"],
                "weather_enabled": activity["weather_enabled"]}
               if activity is not None else {}),
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


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def _norm_box(b):
    """Validate a normalized [x1,y1,x2,y2]; return it ordered+clamped or None."""
    try:
        x1, y1, x2, y2 = (_clamp01(v) for v in b)
    except (TypeError, ValueError):
        return None
    lo_x, hi_x = sorted((x1, x2))
    lo_y, hi_y = sorted((y1, y2))
    if hi_x - lo_x < 0.01 or hi_y - lo_y < 0.01:  # too small to be real
        return None
    return [round(lo_x, 5), round(lo_y, 5), round(hi_x, 5), round(hi_y, 5)]


# Curated list for the display-tz dropdown (IANA names). "" = auto (device tz).
COMMON_TIMEZONES = [
    "UTC", "America/New_York", "America/Chicago", "America/Denver",
    "America/Phoenix", "America/Los_Angeles", "America/Anchorage",
    "Pacific/Honolulu", "America/Toronto", "America/Sao_Paulo",
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Madrid",
    "Africa/Nairobi", "Asia/Jerusalem", "Asia/Kolkata", "Asia/Shanghai",
    "Asia/Tokyo", "Australia/Sydney", "Pacific/Auckland",
]


class DeviceRecordSettingsView(LoginRequiredMixin, View):
    """Set the recorder's capture mode + daily hour window from the dashboard.

    mode: motion (clips only on motion — default) | continuous (record the
    whole window in ~10-minute clips; heavy on SD card + uploads).
    window: start/end hours (0-23, device local time), both blank = all day;
    start > end wraps past midnight. Gates BOTH modes; the WittyPi power
    schedule stays the outer envelope. The recorder hot-reloads the change
    after the device's next check-in.
    """

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        def _fail(msg):
            if is_ajax:
                return JsonResponse({"error": msg}, status=400)
            messages.error(request, msg)
            return redirect("devices:detail", pk=pk)

        mode = (request.POST.get("mode") or "").strip().lower()
        if mode not in dict(device.RECORD_MODES):
            return _fail("Invalid recording mode.")

        # Clip tail (post-roll) + max clip length (force-rotate). Clamped to sane
        # ranges; blank keeps the current value.
        def _clamp_int(name, lo, hi, cur):
            raw = (request.POST.get(name) or "").strip()
            if raw == "":
                return cur
            try:
                return max(lo, min(hi, int(raw)))
            except (ValueError, TypeError):
                return None
        post_roll = _clamp_int("post_roll", 1, 300, device.record_post_roll)
        max_segment = _clamp_int("max_segment", 30, 3600, device.record_max_segment)
        if post_roll is None or max_segment is None:
            return _fail("Clip tail and max length must be whole numbers of seconds.")

        start_raw = (request.POST.get("start") or "").strip()
        end_raw = (request.POST.get("end") or "").strip()
        window = None
        if start_raw or end_raw:
            try:
                start, end = int(start_raw), int(end_raw)
            except (TypeError, ValueError):
                return _fail("Pick both a start and end hour (or clear both for all-day).")
            if not (0 <= start <= 23 and 0 <= end <= 23):
                return _fail("Hours must be 0–23.")
            if start == end:
                return _fail("Start and end can't be the same hour — clear both to record all day.")
            window = {"start": start, "end": end}

        device.record_mode = mode
        device.record_window = window
        device.record_post_roll = post_roll
        device.record_max_segment = max_segment
        device.save(update_fields=["record_mode", "record_window",
                                   "record_post_roll", "record_max_segment"])

        mode_label = dict(device.RECORD_MODES)[mode]
        window_label = (f"{window['start']:02d}:00–{window['end']:02d}:00"
                        if window else "all day")
        if is_ajax:
            return JsonResponse({"ok": True, "mode": mode, "mode_label": mode_label,
                                 "window_label": window_label,
                                 "post_roll": post_roll, "max_segment": max_segment})
        messages.success(
            request,
            f"Recording set to “{mode_label}”, {window_label}. The device adopts "
            "it on its next check-in.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceVideoUploadModeView(LoginRequiredMixin, View):
    """Set the video upload policy (auto|manual) from the dashboard.

    manual (default) = the uploader holds recorded videos even on WiFi until
    "Upload now" — so tethering the unit to a phone hotspot can't silently
    drain a cellular data plan. auto = upload whenever WiFi is connected.
    The device adopts the change on its next check-in.
    """

    VALID = {"auto", "manual"}

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        mode = (request.POST.get("mode") or "").strip().lower()
        if mode not in self.VALID:
            if is_ajax:
                return JsonResponse({"error": "Invalid upload mode."}, status=400)
            messages.error(request, "Invalid upload mode.")
            return redirect("devices:detail", pk=pk)
        device.video_upload_mode = mode
        device.save(update_fields=["video_upload_mode"])
        label = dict(device.VIDEO_UPLOAD_MODES).get(mode, mode)
        if is_ajax:
            return JsonResponse({"ok": True, "mode": mode, "label": label})
        messages.success(
            request,
            f"Video uploads set to “{label}”. The device adopts it on its next check-in.",
        )
        return redirect("devices:detail", pk=pk)


class DeviceUploadNowView(LoginRequiredMixin, View):
    """One-shot backlog drain: queue an `upload_videos` command for the device.

    Used with manual upload mode — the uploader pushes everything currently
    pending, then goes back to holding. NOTE: `pending_command` is a single
    slot, so this replaces any not-yet-delivered command (and vice versa)."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        device.pending_command = "upload_videos"
        device.command_params = {}
        device.save(update_fields=["pending_command", "command_params"])
        if is_ajax:
            return JsonResponse({"ok": True})
        messages.success(
            request,
            "Upload requested — the device pushes its pending videos on its next "
            "check-in (WiFi required).",
        )
        return redirect("devices:detail", pk=pk)


class DeviceDisplayTzView(LoginRequiredMixin, View):
    """Set the timezone the dashboard displays this device's activity in."""

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        tz = (request.POST.get("display_tz") or "").strip()
        if tz:
            try:
                ZoneInfo(tz)
            except (ZoneInfoNotFoundError, ValueError):
                if is_ajax:
                    return JsonResponse({"error": "Unknown timezone."}, status=400)
                messages.error(request, "Unknown timezone.")
                return redirect("devices:detail", pk=pk)
        device.display_tz = tz
        device.save(update_fields=["display_tz"])
        if is_ajax:
            return JsonResponse({"ok": True, "display_tz": tz})
        messages.success(request, "Display timezone updated.")
        return redirect("devices:detail", pk=pk)


class DeviceRoiEditorView(LoginRequiredMixin, View):
    """Edit the hotel ROI + nest boxes/IDs over the device's latest picture.

    Coordinates are stored NORMALIZED (0..1) so they're resolution-independent;
    the device converts them to its lores (motion ROI) and main (overlay) coords.
    """

    template_name = "devices/roi_editor.html"

    def get(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        img_hb = device.heartbeats.exclude(image_storage_key="").first()
        ctx = {
            "device": device,
            "image_url": _presign_image(img_hb.image_storage_key) if img_hb else None,
            "roi_json": json.dumps(device.roi_override),
            "nests_json": json.dumps(device.nest_layout or []),
        }
        from django.shortcuts import render
        return render(request, self.template_name, ctx)

    def post(self, request, pk):
        device = _device_or_403(request.user, pk, "manager")
        try:
            body = json.loads(request.body or "{}")
        except ValueError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)

        roi = body.get("roi")
        roi_override = _norm_box(roi) if roi else None

        nests = []
        for n in (body.get("nests") or [])[:200]:
            box = _norm_box(n.get("box"))
            if box is None:
                continue
            try:
                nid = int(n.get("id"))
            except (TypeError, ValueError):
                nid = len(nests) + 1
            nests.append({"id": nid, "box": box})

        device.roi_override = roi_override
        device.nest_layout = nests
        device.save(update_fields=["roi_override", "nest_layout"])
        return JsonResponse({"ok": True, "roi": roi_override, "nests": nests})


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
