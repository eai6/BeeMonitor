"""When a clip started, and the real time of a moment in it (memory/60).

Every row of the tracks, events and interactions tables carries a timestamp:
the clip's start plus the seconds into the clip. The start is only as good as
where it came from, so this module is the one place that decides it:

- device clips: the recorder sent the true instant; shown in the unit's zone;
- file-name times: the site's local wall-clock (``Video.localise_recorded_at``
  makes it a true instant once the site has a zone); without a zone it stays a
  wall-clock and its timestamps carry no offset;
- the uploader's typed start, or the file's own metadata: true instants;
- the upload time is not a recording time, so there is no start at all and
  timestamps are blank. Seconds into the clip are still known.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones


@lru_cache(maxsize=1)
def time_zones() -> list[str]:
    """Every IANA zone name, sorted, for the site forms."""
    return sorted(z for z in available_timezones() if "/" in z and not z.startswith(("Etc/", "SystemV/")))


def valid_tz(name: str) -> bool:
    if not name:
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def _zone(name):
    return ZoneInfo(name) if valid_tz(name or "") else None


def clip_start(video) -> datetime | None:
    """The clip's recorded start, in its local time.

    Aware for a true instant (in the site's or unit's zone when known, else
    UTC); naive for a file-name wall-clock whose site has no zone yet; None
    when the clip's start is not known.
    """
    if video is None or video.recorded_at is None:
        return None
    meta = video.metadata or {}
    source = meta.get("recorded_at_source")
    if source == "upload_time":
        return None
    if source == "filename":
        site = video.site if getattr(video, "site_id", None) else None
        zone = _zone(getattr(site, "tz_name", "") if site else "")
        if zone is None:
            return video.recorded_wall()
        return video.recorded_at.astimezone(zone)
    device = video.device if getattr(video, "device_id", None) else None
    site = video.site if getattr(video, "site_id", None) else None
    zone = (_zone(getattr(device, "tz_name", "") if device else "")
            or _zone(getattr(site, "tz_name", "") if site else ""))
    return video.recorded_at.astimezone(zone or dt_timezone.utc)


def iso(dt: datetime | None) -> str:
    """ISO 8601 to the millisecond; no offset for a naive wall-clock."""
    if dt is None:
        return ""
    return dt.isoformat(timespec="milliseconds")


def timestamp_at(video_or_start, seconds) -> str:
    """The real time ``seconds`` into a clip, or "" when its start is unknown.

    Pass the video, or its ``clip_start`` computed once when looping rows.
    """
    start = (video_or_start if isinstance(video_or_start, datetime) or video_or_start is None
             else clip_start(video_or_start))
    if start is None or seconds is None:
        return ""
    try:
        return iso(start + timedelta(seconds=float(seconds)))
    except (TypeError, ValueError, OverflowError):
        return ""
