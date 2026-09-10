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


def draft(available_qs, project_videos, per_cell=2, prefer_confirmed=True):
    """A balanced starting selection: up to ``per_cell`` clips per empty cell.

    Deliberately a draft. It picks clips the project is missing across hotels
    and hours; the user still looks at each one before adding it, because
    "spread evenly" and "worth annotating" are different questions and only the
    second one needs eyes.
    """
    have = _counts(project_videos)
    wanted, seen = [], set()

    qs = available_qs.exclude(hour=None).exclude(device=None)
    if prefer_confirmed:
        # A clip the device flagged as a bee is likelier to contain something
        # to label — but it is only an ordering, never a filter, or the sample
        # inherits the detector's blind spots.
        qs = qs.order_by("-metadata__bee_confirmed", "-recorded_at")
    else:
        qs = qs.order_by("-recorded_at")

    per_cell_taken = {}
    for video in qs.iterator():
        cell = (video.device_id, video.hour)
        if have.get(cell):
            continue                       # already represented
        if per_cell_taken.get(cell, 0) >= per_cell:
            continue
        if video.pk in seen:
            continue
        per_cell_taken[cell] = per_cell_taken.get(cell, 0) + 1
        seen.add(video.pk)
        wanted.append(video.pk)
    return wanted
