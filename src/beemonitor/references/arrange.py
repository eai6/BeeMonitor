"""Numbering the reference objects a model found (memory/83).

A detector's boxes have no identities, and a reference's number must mean the
same object on every clip: per-tube totals, trips and reviews are keyed on it.
How they are numbered is the study's to say:

- ``grid``: ``rows`` × ``columns``, numbered row by row from the top left
  (1..columns on the top row), as tubes are numbered by hand. Extra boxes are
  dropped, a missed one is placed where its row and column put it, and a grid
  that can't be told for sure fails with a reason rather than guess.
- ``reading_order``: every box, rows top to bottom then left to right, no
  fixed count: references that aren't a grid.

Boxes are ``[x1, y1, x2, y2, ...]`` in any one unit (pixels or 0..1).
"""

from __future__ import annotations

import math

import numpy as np

# At least this share of the grid's cells found, or it fails.
MIN_FOUND = 0.5
# A row: boxes within this share of a box's height of each other (along the
# row's direction).
SAME_ROW = 0.5
# A row's own spacing is trusted with this many boxes or more, and when within
# this share of the others'; else its missed cells follow the other rows'
# columns.
OWN_FIT = 3
SPACING_SLACK = 0.25


def _centres(boxes):
    b = np.asarray([list(map(float, x[:4])) for x in boxes], float).reshape(-1, 4)
    return b, np.column_stack(((b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2))


def _chain(values, gap):
    """Indices of ``values`` in groups: sorted, a new group past ``gap``."""
    order = np.argsort(values)
    groups = []
    for i in order:
        if groups and values[i] - values[groups[-1][-1]] <= gap:
            groups[-1].append(int(i))
        else:
            groups.append([int(i)])
    return groups


def reading_order(boxes):
    """``boxes`` in reading order: rows top to bottom (centres within half a
    box's height), then left to right."""
    boxes = list(boxes or [])
    if not boxes:
        return []
    b, c = _centres(boxes)
    h = float(np.median(b[:, 3] - b[:, 1])) or 1e-6
    rows = _chain(c[:, 1], h * SAME_ROW)
    return [boxes[i] for row in rows for i in sorted(row, key=lambda i: c[i, 0])]


def _tilt(c, w, h):
    """The rows' direction, as an angle: from each box to its nearest box to
    the right in the same row, the median slope."""
    slopes = []
    for i in range(len(c)):
        d = c - c[i]
        ok = (d[:, 0] > w * 0.5) & (np.abs(d[:, 1]) < h)
        if ok.any():
            j = np.flatnonzero(ok)[np.argmin(d[ok, 0])]
            slopes.append(d[j, 1] / d[j, 0])
    return math.atan(float(np.median(slopes))) if slopes else 0.0


def _fail(reason, found, need):
    return {"ok": False, "reason": reason, "boxes": [], "found": [], "detected": found,
            "filled": 0, "dropped": 0, "cells": need}


def grid(boxes, rows, columns):
    """The boxes as a ``rows`` × ``columns`` grid numbered row by row.

    Returns ``{ok, reason, boxes, found, detected, filled, dropped, cells}``:
    ``boxes`` the grid's boxes in number order (``boxes[0]`` is 1), ``found``
    which of them a box was detected for (the rest placed from the grid),
    ``detected`` how many boxes came in, ``filled`` the cells placed,
    ``dropped`` boxes left out. ``ok`` False with a ``reason`` and no boxes
    when it can't be told."""
    rows, columns = int(rows), int(columns)
    need = rows * columns
    boxes = [list(map(float, x[:4])) for x in boxes or []]
    if rows < 1 or columns < 1:
        return _fail("Rows and columns must be at least 1.", len(boxes), max(need, 0))
    if len(boxes) < max(1, math.ceil(MIN_FOUND * need)):
        return _fail(f"Found {len(boxes)} of {need}: fewer than half.", len(boxes), need)
    b, c = _centres(boxes)
    w = float(np.median(b[:, 2] - b[:, 0])) or 1e-6
    h = float(np.median(b[:, 3] - b[:, 1])) or 1e-6

    # Turned so rows run level: u along a row, v across.
    a = _tilt(c, w, h)
    cos, sin = math.cos(a), math.sin(a)
    u = c[:, 0] * cos + c[:, 1] * sin
    v = -c[:, 0] * sin + c[:, 1] * cos

    # 1. Rows: the fullest ones, top to bottom.
    groups = _chain(v, h * SAME_ROW)
    if len(groups) < rows:
        return _fail(f"Found {len(groups)} rows of {rows}.", len(boxes), need)
    groups = sorted(sorted(groups, key=len, reverse=True)[:rows], key=lambda g: float(np.median(v[g])))

    # 2. The spacing between neighbours in a row, then each box's column.
    step = None
    if columns > 1:
        gaps = np.concatenate([np.diff(np.sort(u[g])) for g in groups if len(g) > 1] or [np.array([])])
        gaps = gaps[gaps > w * 0.5]
        if not len(gaps):
            return _fail("Can't tell the spacing between them.", len(boxes), need)
        step = float(np.median(gaps))
        k = np.maximum(1, np.round(gaps / step))
        step = float(np.median(gaps / k))
    ref = float(np.min(u[max(groups, key=len)]))

    def column(i):
        return int(round((u[i] - ref) / step)) if step else 0

    cols_of = {i: column(i) for g in groups for i in g}
    lo, hi = min(cols_of.values()), max(cols_of.values())
    if hi - lo + 1 < columns:
        return _fail(f"Found {hi - lo + 1} columns of {columns}.", len(boxes), need)
    # The `columns` consecutive columns holding the most boxes.
    counts = np.bincount(np.array(list(cols_of.values())) - lo, minlength=hi - lo + 1)
    start = lo + int(np.argmax([counts[s:s + columns].sum() for s in range(hi - lo + 2 - columns)]))

    # 3. Each row's line, then one box a cell (the nearest the line) and the
    # rest placed on it.
    lines, kept = [], []
    for g in groups:
        g = [i for i in g if start <= cols_of[i] < start + columns]
        if not g:
            return _fail("A row has no box in the grid's columns.", len(boxes), need)
        # One box a cell: the nearest the row's line at the common spacing,
        # then the line fitted to those.
        icept = float(np.median([u[i] - (step or 0.0) * (cols_of[i] - start) for i in g]))
        cell = {}
        for i in g:
            kk = cols_of[i] - start
            off = abs(u[i] - (icept + (step or 0.0) * kk))
            if kk not in cell or off < cell[kk][1]:
                cell[kk] = (i, off)
        slope = step or 0.0
        if step and len(cell) > 1:
            ks = np.array(list(cell), float)
            slope, icept = (float(x) for x in np.polyfit(ks, u[[i for i, _ in cell.values()]], 1))
        lines.append({"slope": slope, "icept": icept, "v": float(np.median(v[g])),
                      "cell": {kk: i for kk, (i, _) in cell.items()}})
        kept.extend(i for i, _ in cell.values())
    if len(kept) < math.ceil(MIN_FOUND * need):
        return _fail(f"Found {len(kept)} of {need} in the grid: fewer than half.", len(boxes), need)

    # A row with few boxes, or an odd spacing, follows the other rows' columns.
    trusted = [ln for ln in lines if len(ln["cell"]) >= min(OWN_FIT, columns)
               and (not step or abs(ln["slope"] / step - 1) <= SPACING_SLACK)]

    def common(kk):
        return float(np.median([t["icept"] + t["slope"] * kk for t in trusted]))

    def u_at(ln, kk):
        if ln in trusted or not trusted:
            return ln["icept"] + ln["slope"] * kk
        return common(kk) + float(np.median([u[i] - common(k2) for k2, i in ln["cell"].items()]))

    out, found = [], []
    for ln in lines:
        for kk in range(columns):
            i = ln["cell"].get(kk)
            if i is not None:
                out.append([float(x) for x in b[i]])
                found.append(True)
                continue
            cu, cv = u_at(ln, kk), ln["v"]
            cx, cy = cu * cos - cv * sin, cu * sin + cv * cos
            out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            found.append(False)
    return {"ok": True, "reason": "", "boxes": out, "found": found, "detected": len(boxes),
            "filled": found.count(False), "dropped": len(boxes) - len(kept), "cells": need}
