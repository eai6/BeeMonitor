"""Is this annotation project's sample representative?

The clip picker answers "what am I looking at". This answers the question the
platform had no version of: *what have I already got, and what am I missing.*
A training set drawn from one hotel at one hour produces a model that works at
that hotel at that hour, and nothing in the interface said so — the project was
a flat list of titles, and a lopsided sample looked exactly like a balanced one.

Two aggregate queries over (device, hour): what exists, and what is in the
project. Everything below is arithmetic on those.
"""

from django.db.models import Count

# Cells are shaded by how much of the project they hold, relative to the
# busiest one — the comparison that matters is between cells, not an absolute.
_SHADES = ("c1", "c2", "c3", "c4")


def _counts(qs):
    # .order_by() clears Video.Meta.ordering. Without it Django folds the
    # ordering field into the GROUP BY and every cell counts 1 — the same trap
    # that made the hub's per-hotel counts all read 1.
    rows = (qs.exclude(hour=None).exclude(device=None).order_by()
            .values("device_id", "hour").annotate(n=Count("id")))
    return {(r["device_id"], r["hour"]): r["n"] for r in rows}


def build(available_qs, project_videos, devices, dots):
    """A device x hour-of-day grid, and what it says.

    ``available_qs`` is every clip the user could add (already filtered by the
    rail, so the map always describes what is on screen); ``project_videos`` is
    what the project already holds.
    """
    have = _counts(project_videos)
    can = _counts(available_qs)

    hours = sorted({h for _d, h in list(have) + list(can)})
    if not hours:
        return None
    # A contiguous span reads as a day; the gaps inside it are the finding.
    hours = list(range(min(hours), max(hours) + 1))

    peak = max(have.values()) if have else 0
    rows = []
    for device in devices:
        cells = []
        for hour in hours:
            n = have.get((device.id, hour), 0)
            avail = can.get((device.id, hour), 0)
            if n:
                # Quartile of the peak, so the busiest cell is always darkest.
                cells.append({"hour": hour, "n": n, "available": avail,
                              "state": _SHADES[min(int(n * 4 / (peak + 1)), 3)]})
            elif avail:
                # The state worth seeing: footage exists and none of it is
                # annotated. Invisible until now.
                cells.append({"hour": hour, "n": 0, "available": avail, "state": "gap"})
            else:
                cells.append({"hour": hour, "n": 0, "available": 0, "state": "none"})
        rows.append({
            "device": device,
            "dot": dots.get(device.id, "#9ca3af"),
            "cells": cells,
            "total": sum(c["n"] for c in cells),
            "available": sum(c["available"] for c in cells),
        })

    return {"hours": hours, "rows": rows, "peak": peak,
            "findings": _findings(rows, hours),
            "in_project": sum(have.values()),
            "gaps": sum(1 for r in rows for c in r["cells"] if c["state"] == "gap")}


def _findings(rows, hours):
    """The three things a lopsided sample looks like, in plain sentences.

    Stated rather than left to be noticed: a grid of numbers is only an
    instrument if it says what it means.
    """
    out = []

    silent = [r for r in rows if r["total"] == 0 and r["available"]]
    for r in silent:
        out.append(
            f"{r['device'].name} contributes nothing — {r['available']} clip"
            f"{'s' if r['available'] != 1 else ''} available, none annotated. "
            "A model trained on this project has never seen it.")

    # Hours nobody annotated but everybody recorded.
    for hour in hours:
        annotated = sum(c["n"] for r in rows for c in r["cells"] if c["hour"] == hour)
        avail = sum(c["available"] for r in rows for c in r["cells"] if c["hour"] == hour)
        if annotated == 0 and avail >= 10:
            out.append(f"Nothing from {hour:02d}:00, though {avail} clips were recorded then.")

    busiest = max(((c["n"], r["device"].name, c["hour"])
                   for r in rows for c in r["cells"]), default=(0, "", 0))
    total = sum(r["total"] for r in rows)
    if busiest[0] and total and busiest[0] / total > 0.35:
        out.append(
            f"{busiest[1]} at {busiest[2]:02d}:00 holds {busiest[0]} of "
            f"{total} clips — over a third of the whole project in one hour.")
    return out[:6]


def _hotel(device_id, site_id, site_name):
    """A clip's hotel: its field unit, else its site. Clips imported from
    uploads or cloud storage have a site and no unit."""
    if device_id:
        return ("unit", device_id)
    if site_id:
        return ("site", site_id)
    return ("name", (site_name or "").lower())


def _candidates(available_qs, exclude_ids, project=None):
    """``[(id, device, site, hotel, hour, recorded_at)]`` that may be picked,
    oldest first: not already in the project, not an empty file, not on the
    project's left-out days (memory/57)."""
    from . import leftout

    rules = leftout.project_rules(project) if project is not None else {}
    qs = (available_qs.exclude(pk__in=exclude_ids)
          .exclude(file_size_bytes__lt=leftout.TOO_SMALL_BYTES)
          .select_related("site", "device").order_by("recorded_at", "id"))
    out = []
    for v in qs:
        if rules and leftout.is_left_out(v, rules):
            continue
        out.append((v.pk, v.device_id, v.site_id, _hotel(v.device_id, v.site_id, v.site_name),
                    v.hour, v.recorded_at))
    return out


def _spaced(items, k):
    """k items evenly spaced through a date-ordered list, so a pick spans the
    whole recorded period instead of bunching at its newest end."""
    n = len(items)
    if k >= n:
        return list(items)
    return [items[int((j + 0.5) * n / k)] for j in range(k)]


def _pick(vid, dev, site, hotel, hour, rec):
    return {"id": vid, "device": dev, "site": site, "hour": hour,
            "date": rec.date().isoformat() if rec else ""}


def draft(available_qs, project_videos, per_cell=2, project=None):
    """A balanced starting selection: up to ``per_cell`` clips per empty cell.

    Deliberately a draft. It picks clips the project is missing across hotels
    and hours — spread across the recorded dates, not the newest — and the user
    still looks at each one before adding it, because "spread evenly" and
    "worth annotating" are different questions and only the second one needs
    eyes. Returns ``[{id, device, site, hour, date}]``.
    """
    have = {(_hotel(d, s, n), h) for d, s, n, h in project_videos.exclude(hour=None).order_by()
            .values_list("device_id", "site_id", "site_name", "hour")}
    in_project = list(project_videos.values_list("pk", flat=True))
    cells = {}
    for c in _candidates(available_qs.exclude(hour=None), in_project, project):
        cells.setdefault((c[3], c[4]), []).append(c)
    picks = []
    for cell in sorted(cells, key=str):
        if cell in have:
            continue                       # already represented
        picks += [_pick(*it) for it in _spaced(cells[cell], per_cell)]
    return picks


def even_spread(available_qs, project_videos, target, project=None):
    """Up to ``target`` clips from everything the filter matches, in
    proportion to the footage: a hotel with 60% of the clips gets 60% of the
    picks, and within a hotel the picks are spaced evenly through time, so
    every day and hour is sampled as much as it was recorded. The sample then
    looks like the footage that will be analysed.
    """
    in_project = list(project_videos.values_list("pk", flat=True))
    by_hotel = {}
    for c in _candidates(available_qs, in_project, project):
        by_hotel.setdefault(c[3], []).append(c)
    total = sum(len(v) for v in by_hotel.values())
    if not total or target <= 0:
        return []
    if target >= total:
        alloc = {h: len(v) for h, v in by_hotel.items()}
    else:
        # Largest remainder: shares add up to exactly ``target``.
        exact = {h: len(v) * target / total for h, v in by_hotel.items()}
        alloc = {h: int(x) for h, x in exact.items()}
        for h in sorted(exact, key=lambda h: exact[h] - alloc[h], reverse=True)[:target - sum(alloc.values())]:
            alloc[h] += 1
    picks = []
    for h in sorted(by_hotel, key=str):
        picks += [_pick(*it) for it in _spaced(by_hotel[h], alloc[h])]
    return picks
