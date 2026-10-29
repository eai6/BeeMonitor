"""Tracks drawn over the original video in the browser (memory/46).

The worker's tracking CSV is ~100k rows / 16 MB for a 10-minute clip — far too
much to put in a page, and most of it (floats to 13 places, repeated text
columns) the overlay never reads. This reduces it to what is drawn: per frame,
each track's box in whole pixels and whether the tracker was predicting. Built
once per result and kept beside the tracking CSV as gzip'd JSON (~1 MB).
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import logging

logger = logging.getLogger(__name__)

# Bump when the payload's shape changes: the stored copy is keyed on it.
VERSION = 3  # 3: boxes only — events and references load live (memory/60)


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_rows(tracking_csv: str) -> list[list[int]]:
    """``[[frame, track_id, x1, y1, x2, y2, lost], ...]`` sorted by frame.

    ``lost`` is 1 where the tracker had no detection for the track and only
    predicted where it was. The tracker says so in the ``detected`` column
    (memory/60); the drawn box is the last detected box centred on the
    prediction. Tracking CSVs from before that column existed are drawn
    without the distinction: every box solid, as the file records it.

    A frame can hold a track twice (the motion-gated recorder's lookback pass
    and the live pass both emit it); the later row wins.
    """
    seen: dict[tuple[int, int], list[int]] = {}
    for row in csv.DictReader(io.StringIO(tracking_csv)):
        frame, tid = _f(row.get("frame")), _f(row.get("track_id"))
        box = tuple(_f(row.get(k)) for k in ("x1", "y1", "x2", "y2"))
        if frame is None or tid is None or None in box:
            continue
        frame, tid = int(frame), int(tid)
        detected = _f(row.get("detected"))
        lost = int(detected is not None and not detected)
        x1, y1, x2, y2 = box
        cx, cy = _f(row.get("cx")), _f(row.get("cy"))
        if lost and cx is not None and cy is not None:
            hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
            x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
        seen[(frame, tid)] = [frame, tid, round(x1), round(y1), round(x2), round(y2), lost]
    return sorted(seen.values(), key=lambda r: (r[0], r[1]))


def build(tracking_csv: str, track_rows: list[dict], fps: float) -> dict:
    """The stored overlay payload for one clip: boxes per frame and each track's
    identity and detected span. Events and references are not stored here —
    they are fetched live (``?part=events``) so they always match the run."""
    rows = frame_rows(tracking_csv)
    spans: dict[str, dict] = {}
    for frame, tid, _x1, _y1, _x2, _y2, lost in rows:
        if lost:
            continue
        s = spans.setdefault(str(tid), {"first": frame, "last": frame, "n": 0})
        s["first"], s["last"] = min(s["first"], frame), max(s["last"], frame)
        s["n"] += 1
    tracks = {}
    for t in track_rows or []:
        tid = _f(t.get("track_id"))
        if tid is None:
            continue
        span = spans.get(str(int(tid))) or {}
        tracks[str(int(tid))] = {
            "species": t.get("species") or "",
            "species_confidence": _f(t.get("species_confidence")),
            "species_votes": t.get("species_votes") or "",
            "marker": t.get("marker_id") or "",
            "first": span.get("first"),
            "last": span.get("last"),
            "frames_seen": span.get("n"),
        }
    return {
        "v": VERSION,
        "fps": float(fps or 0) or 25.0,
        "tracks": tracks,
        "rows": [v for r in rows for v in r],  # flat: 7 ints per row
        "row_width": 7,
    }


def stored_path(tracking_path: str) -> str:
    """Where a result's payload is kept: beside its tracking CSV."""
    folder = tracking_path.rsplit("/", 1)[0] if "/" in tracking_path else ""
    return f"{folder}/overlay_v{VERSION}.json.gz" if folder else f"overlay_v{VERSION}.json.gz"


def encode(payload: dict) -> bytes:
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), compresslevel=6)
