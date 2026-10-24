"""Days a project leaves out of sampling, pre-labeling and training (memory/57).

A held-out test set has to be whole days, not frames: frames from one clip or
one morning are near-copies, so a model tested on frames beside its training
frames scores well without having learned much. The project's Settings tab
holds the days as text, one line per site::

    mendels: 2024-04-21, 2024-04-30
    natalies: 2024-05-04
    2024-06-05            # no site: every site

A clip's day is its local recording date: the device's time zone when it has
one, else the date as stored (filename times are stored as written).
"""

import re
from datetime import date

# Clips smaller than this hold no video: a unit that ran out of space still
# wrote the file name (mendels, 2–6 May 2024). Sampling them only fails.
TOO_SMALL_BYTES = 1_000_000

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def parse(text):
    """``(rules, errors)``: rules maps a lower-cased site name ("" = every
    site) to a set of dates; errors are human-readable problems, one per bad
    entry. Nothing in the text can raise."""
    rules, errors = {}, []
    for n, raw in enumerate((text or "").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        site, sep, rest = line.partition(":")
        if not sep or _DATE.fullmatch(site.strip()):
            site, rest = "", line
        site = site.strip().lower()
        for token in re.split(r"[,\s]+", rest.strip()):
            if not token:
                continue
            try:
                day = date.fromisoformat(token)
            except ValueError:
                errors.append(f"line {n}: '{token}' is not a date (use YYYY-MM-DD)")
                continue
            rules.setdefault(site, set()).add(day)
    return rules, errors


def video_site(video):
    if video.site_id:
        return video.site.name
    return video.site_name or ""


def video_day(video):
    """The clip's local recording date, or None when it has no time."""
    when = video.recorded_at
    if when is None:
        return None
    tz_name = getattr(video.device, "tz_name", "") if video.device_id else ""
    if tz_name:
        try:
            from zoneinfo import ZoneInfo
            return when.astimezone(ZoneInfo(tz_name)).date()
        except Exception:  # unknown zone: fall back to the stored date
            pass
    return when.date()


def is_left_out(video, rules):
    if not rules:
        return False
    day = video_day(video)
    if day is None:
        return False
    return day in rules.get("", ()) or day in rules.get(video_site(video).lower(), ())


def too_small(video):
    """True for a clip whose file holds no video (unknown size is kept)."""
    size = video.file_size_bytes
    return size is not None and size < TOO_SMALL_BYTES


def project_rules(project):
    return parse(project.left_out_days)[0]


def left_out_ids(project, videos=None):
    """Pks of the project's clips (or of ``videos``) on left-out days."""
    rules = project_rules(project)
    if not rules:
        return set()
    qs = project.videos.all() if videos is None else videos
    return {v.pk for v in qs.select_related("site", "device") if is_left_out(v, rules)}


def unknown_sites(project, rules):
    """Site names in ``rules`` that none of the project's clips has."""
    named = {s for s in rules if s}
    if not named:
        return []
    have = {n.lower() for n in project.videos.exclude(site=None)
            .values_list("site__name", flat=True)}
    have |= {n.lower() for n in project.videos.values_list("site_name", flat=True) if n}
    return sorted(named - have)


def describe(rules):
    """The rules as JSON-friendly text, for a training job's record."""
    return {(site or "*"): sorted(d.isoformat() for d in days)
            for site, days in sorted(rules.items())}


def split_for_sampling(project, videos):
    """``(keep, left_out, empty)`` lists of the given clips: sampling and
    pre-labeling skip clips on left-out days and clips with no video."""
    rules = project_rules(project)
    keep, left_out, empty = [], [], []
    for v in videos:
        if is_left_out(v, rules):
            left_out.append(v)
        elif too_small(v):
            empty.append(v)
        else:
            keep.append(v)
    return keep, left_out, empty


def skipped_note(left_out, empty):
    parts = []
    if left_out:
        parts.append(f"{len(left_out)} clip(s) on left-out days")
    if empty:
        parts.append(f"{len(empty)} clip(s) with no video in the file")
    return f" Skipped {' and '.join(parts)}." if parts else ""
