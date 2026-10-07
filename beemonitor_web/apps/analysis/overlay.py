"""Tracks drawn over the original video in the browser (memory/46).

The worker's tracking CSV is ~100k rows / 16 MB for a 10-minute clip — far too
much to put in a page, and most of it (floats to 13 places, repeated text
columns) the overlay never reads. This reduces it to what is drawn: per frame,
each track's box in whole pixels and whether the tracker was coasting. Built
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
VERSION = 1


def _f(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def frame_rows(tracking_csv: str) -> list[list[int]]:
    """``[[frame, track_id, x1, y1, x2, y2, lost], ...]`` sorted by frame.

    ``lost`` is 1 where the tracker had no detection for the track: the CSV
    then repeats the last seen box and confidence while cx/cy move with the
    tracker's prediction, so the drawn box is that box centred on the
    prediction — where the tracker thought the insect was.

    A frame can hold a track twice (the motion-gated recorder's lookback pass
    and the live pass both emit it); the later row wins.
    """
    seen: dict[tuple[int, int], list[int]] = {}
    last_box: dict[int, tuple] = {}
    for row in csv.DictReader(io.StringIO(tracking_csv)):
        frame, tid = _f(row.get("frame")), _f(row.get("track_id"))
        box = tuple(_f(row.get(k)) for k in ("x1", "y1", "x2", "y2"))
        if frame is None or tid is None or None in box:
            continue
        frame, tid = int(frame), int(tid)
        key = box + (row.get("confidence"),)
        lost = int(last_box.get(tid) == key)
        last_box[tid] = key
        x1, y1, x2, y2 = box
        cx, cy = _f(row.get("cx")), _f(row.get("cy"))
        if lost and cx is not None and cy is not None:
            hw, hh = (x2 - x1) / 2, (y2 - y1) / 2
            x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
        seen[(frame, tid)] = [frame, tid, round(x1), round(y1), round(x2), round(y2), lost]
    return sorted(seen.values(), key=lambda r: (r[0], r[1]))


def _regions(config: dict) -> list[dict]:
    """The clip's drawn hotel ROI and nests, normalised 0–1 as stored."""
    out = []
    poly = config.get("hotel_polygon")
    roi = config.get("hotel_roi")
    if poly and len(poly) >= 3:
        out.append({"label": "hotel", "points": poly})
    elif roi and len(roi) == 4:
        x1, y1, x2, y2 = roi
        out.append({"label": "hotel", "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]})
    for nest in config.get("nest_layout") or []:
        pts = nest.get("points")
        if not pts and nest.get("box") and len(nest["box"]) == 4:
            x1, y1, x2, y2 = nest["box"]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if pts and len(pts) >= 3:
            out.append({"label": f"nest {nest.get('id', '')}".strip(), "points": pts})
    return out


def build(tracking_csv: str, track_rows: list[dict], event_rows: list[dict],
          fps: float, config: dict | None = None) -> dict:
    """The overlay payload for one clip."""
    rows = frame_rows(tracking_csv)
    tracks = {}
    for t in track_rows or []:
        tid = _f(t.get("track_id"))
        if tid is None:
            continue
        tracks[str(int(tid))] = {
            "species": t.get("species") or "",
            "species_confidence": _f(t.get("species_confidence")),
            "species_votes": t.get("species_votes") or "",
            "species_best_guess": t.get("species_best_guess") or "",
            "marker": t.get("marker_id") or "",
            "first": _f(t.get("first_frame")),
            "last": _f(t.get("last_frame")),
            "frames_seen": _f(t.get("frames_seen")),
        }
    events = []
    for e in event_rows or []:
        frame, subject = _f(e.get("frame")), _f(e.get("subject"))
        if frame is None or subject is None:
            continue  # a reference's event, not an insect's
        events.append([int(frame), int(subject), str(e.get("action") or ""),
                       str(e.get("target") or "")])
    return {
        "v": VERSION,
        "fps": float(fps or 0) or 25.0,
        "tracks": tracks,
        "rows": [v for r in rows for v in r],  # flat: 7 ints per row
        "row_width": 7,
        "events": events,
        "regions": _regions(config or {}),
    }


def stored_path(tracking_path: str) -> str:
    """Where a result's payload is kept: beside its tracking CSV."""
    folder = tracking_path.rsplit("/", 1)[0] if "/" in tracking_path else ""
    return f"{folder}/overlay_v{VERSION}.json.gz" if folder else f"overlay_v{VERSION}.json.gz"


def encode(payload: dict) -> bytes:
    return gzip.compress(json.dumps(payload, separators=(",", ":")).encode(), compresslevel=6)
