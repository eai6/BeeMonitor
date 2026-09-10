"""The video review workspace — filters, and the context its rail and grid need.

Choosing which clips to look at is the same task on the Processing hub (what to
analyse) and in an annotation project (what to label), so it is built once here
rather than twice.

It was, briefly, twice: ``annotations.views`` grew its own title/device/site/
year/month/day filtering while this module's ``apply_video_filters`` already
existed and was already shared between the hub and the pipeline runner. The two
had diverged — the hub understood several hotels at once, time-of-day windows,
date ranges and "not yet analysed"; the annotation page understood none of
them. Same disease as a frame rate resolved four different ways.
"""

import re

_NATALIES_RE = re.compile(r"natalies?", re.IGNORECASE)
_SITEA_RE = re.compile(r"SiteA", re.IGNORECASE)


def _sanitize_site(value: str) -> str:
    """Replace occurrences of 'natalies' with 'SiteA' in display strings."""
    if not value:
        return value
    return _NATALIES_RE.sub("SiteA", value)


def _unsanitize_site(value: str) -> str:
    """Reverse-map 'SiteA' back to 'natalies' for DB queries."""
    if not value:
        return value
    return _SITEA_RE.sub("natalies", value)


# GET/POST params the Processing-hub video filter understands.
VIDEO_FILTER_KEYS = ("device", "site", "year", "month", "day", "hour",
                     "hfrom", "hto", "from", "to", "q", "confirmed", "analysis")


def _values(params, key):
    """Every value for ``key`` — QueryDicts repeat keys, plain dicts don't.

    The hub filters on several hotels at once, so ``device`` arrives repeated.
    The per-device scheduler passes a plain dict with one value
    (devices/scheduling.py), and the API a single string, so both shapes have to
    keep working.
    """
    getlist = getattr(params, "getlist", None)
    if getlist is not None:
        return [v for v in getlist(key) if v not in (None, "")]
    value = params.get(key)
    if value in (None, ""):
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def apply_video_filters(qs, params):
    """Apply the Processing-hub video filters to a Video queryset. ``params`` is
    any dict-like with .get() (a GET or POST QueryDict). Shared by the hub list
    and the pipeline "run on all filtered videos" path so they never diverge."""
    from datetime import datetime, time
    from django.utils import timezone as _tz
    from django.utils.dateparse import parse_date, parse_datetime

    q = (params.get("q") or "").strip()
    if q:
        qs = qs.filter(title__icontains=q)
    devices = _values(params, "device")
    if devices:
        # Several hotels at once: clips interleave by time so the same hour can
        # be compared across them.
        qs = qs.filter(device_id__in=devices)
    if params.get("site"):
        qs = qs.filter(site_name=_unsanitize_site(params.get("site")))
    for field in ("year", "month", "day", "hour"):
        val = params.get(field)
        if val:
            try:
                qs = qs.filter(**{field: int(val)})
            except (ValueError, TypeError):
                pass

    # Daily time-of-day window: videos recorded between hfrom:00 (inclusive)
    # and hto:00 (exclusive) EVERY day — combine with from/to for "6–7 pm each
    # day across June". hfrom > hto wraps past midnight (e.g. 22 → 4).
    try:
        hfrom = int(params.get("hfrom")) if params.get("hfrom") not in (None, "") else None
        hto = int(params.get("hto")) if params.get("hto") not in (None, "") else None
    except (ValueError, TypeError):
        hfrom = hto = None
    if hfrom is not None or hto is not None:
        lo = hfrom if hfrom is not None else 0
        hi = hto if hto is not None else 24
        if lo < hi:
            qs = qs.filter(hour__gte=lo, hour__lt=hi)
        elif lo > hi:  # wraps past midnight
            from django.db.models import Q as _Q
            qs = qs.filter(_Q(hour__gte=lo) | _Q(hour__lt=hi))
        # lo == hi selects nothing meaningful -> ignore (treat as no window)

    def _parse_dt(s):
        dt = parse_datetime(s)
        if dt is None:
            d = parse_date(s)
            if d:
                dt = datetime.combine(d, time.min)
        if dt and _tz.is_naive(dt):
            dt = _tz.make_aware(dt, _tz.get_current_timezone())
        return dt

    if params.get("from"):
        dt = _parse_dt(params.get("from"))
        if dt:
            qs = qs.filter(recorded_at__gte=dt)
    if params.get("to"):
        dt = _parse_dt(params.get("to"))
        if dt:
            qs = qs.filter(recorded_at__lte=dt)

    # "Not yet analyzed" — the review question a run usually answers, and the
    # one thing the hub could show but never filter on.
    analysis = params.get("analysis")
    if analysis:
        from apps.analysis.models import Job
        done = Job.objects.filter(status="completed").values("video_id")
        if analysis == "never":
            qs = qs.exclude(pk__in=done)
        elif analysis == "done":
            qs = qs.filter(pk__in=done)

    confirmed = params.get("confirmed")
    if confirmed == "yes":
        qs = qs.filter(metadata__bee_confirmed=True)
    elif confirmed == "no":
        qs = qs.filter(metadata__bee_confirmed=False)
    elif confirmed == "untagged":
        # Neither True nor False — the key is absent. Written as an explicit
        # exclusion of the two tagged sets rather than `exclude(key=...)`,
        # which drops absent-key rows under SQL NULL semantics and would
        # return nothing at all here.
        tagged = qs.model.objects.filter(
            pk__in=qs.values("pk")).filter(
            metadata__bee_confirmed__in=[True, False]).values("pk")
        qs = qs.exclude(pk__in=tagged)
    return qs


# Stable per-hotel colours. The grid interleaves hotels by time, so a card has
# to say which one it came from at a glance.
DEVICE_DOTS = ["#16a34a", "#b45309", "#0e7490", "#7c3aed", "#be123c", "#4d7c0f"]


def device_rows(devices, selected, counts):
    """Rail rows for the hotel filter: the device, its clip count, its colour."""
    selected = {str(s) for s in (selected or [])}
    return [{
        "obj": d,
        "count": counts.get(d.id, 0),
        "selected": str(d.id) in selected,
        "dot": DEVICE_DOTS[i % len(DEVICE_DOTS)],
    } for i, d in enumerate(devices)]


def dots_by_device(rows):
    return {r["obj"].id: r["dot"] for r in rows}


def filter_options(user_videos):
    """Dropdown values drawn from the user's actual clips, not from a range."""
    return {
        "sites": sorted({_sanitize_site(s) for s in
                         user_videos.exclude(site_name="")
                         .values_list("site_name", flat=True)}),
        "years": sorted(set(user_videos.exclude(year=None).values_list("year", flat=True))),
        "months": sorted(set(user_videos.exclude(month=None).values_list("month", flat=True))),
        "days": sorted(set(user_videos.exclude(day=None).values_list("day", flat=True))),
        "hours": sorted(set(user_videos.exclude(hour=None).values_list("hour", flat=True))),
        # The full clock for the time-of-day window, unlike "hours", which
        # lists only hours that happen to have footage.
        "hours24": list(range(24)),
    }


def group_by_day(videos, dots):
    """Clips as day groups, so the grid reads as footage rather than as rows."""
    days = []
    for video in videos:
        video.dot = dots.get(video.device_id, "#9ca3af")
        when = video.recorded_at or video.uploaded_at
        day = when.date() if when else None
        if not days or days[-1]["day"] != day:
            days.append({"day": day, "videos": [], "hotels": set()})
        days[-1]["videos"].append(video)
        if video.device_id:
            days[-1]["hotels"].add(video.device_id)
    return days


def current_filter(params):
    """The filter as a plain dict, for re-rendering the rail's own state."""
    f = {k: (params.get(k) or "") for k in VIDEO_FILTER_KEYS}
    f["device"] = _values(params, "device")
    return f
