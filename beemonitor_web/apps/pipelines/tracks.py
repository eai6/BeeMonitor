"""One row per track: when it was seen, and what it was.

The tracking table is one row per track per frame, and the worker's species /
marker vote rides in ``summary_stats.identification.by_track``. Neither is the
table people actually analyse: "track 12 was an Osmia lignaria, marker 4, here
from 00:12 to 00:31". This builds that for the job page, its download and the
batch download — so species and marker IDs are in the data without a separate
per-crop votes file.

The same identity columns are joined onto tracking, events and interaction
rows, keyed by their track id. Every CSV people download is built here or in
``primitives.export_rows`` (memory/60): times, not frames, and the worker's
internal column names translated.
"""

from __future__ import annotations

# The track's species and marker, as the models called them (memory/60): no
# confidence floor, no "unidentified" — the confidence is a column, and the
# reader filters on it.
IDENTITY_FIELDS = ["species", "species_confidence", "species_votes",
                   "species_vote_share", "marker_id", "marker_votes", "marker_vote_share"]
TRACK_FIELDS = (["track_id", "start_time", "end_time", "start_clip_sec", "end_clip_sec",
                 "duration_sec"] + IDENTITY_FIELDS)

# The Tracking CSV people download: the worker's per-frame table with times
# added and its internal columns translated (memory/60).
TRACKING_FIELDS = (["frame", "timestamp", "clip_sec", "track_id", "detected",
                    "x1", "y1", "x2", "y2", "cx", "cy", "confidence"] + IDENTITY_FIELDS
                   + ["marker_method", "marker_confidence"])
DETECTION_FIELDS = ["frame", "timestamp", "clip_sec", "x1", "y1", "x2", "y2",
                    "cx", "cy", "confidence", "class"]

_ID_KEYS = ("track_id", "track", "tid", "object_id")
_FRAME_KEYS = ("frame_number", "frame", "frame_num", "frame_idx")


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _key(value):
    """A track id as the string by_track is keyed by ("12", not "12.0")."""
    n = _number(value)
    if n is None:
        return str(value) if value not in (None, "") else None
    return str(int(n)) if n == int(n) else str(n)


def _first(row, keys):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return None


def by_track_of(result):
    """``{track_id: vote}`` from a job result (dict or JobResult)."""
    stats = (result.get("summary_stats") if isinstance(result, dict)
             else getattr(result, "summary_stats", None)) or {}
    return (stats.get("identification") or {}).get("by_track") or {}


def identity(vote):
    """Identity columns for one track's vote."""
    vote = vote or {}
    out = dict.fromkeys(IDENTITY_FIELDS, "")
    taxon, conf = vote.get("taxon"), _number(vote.get("taxon_confidence"))
    if taxon:
        out.update(species=taxon,
                   species_confidence=round(conf, 4) if conf is not None else "",
                   species_votes=vote.get("taxon_votes", ""),
                   species_vote_share=vote.get("taxon_vote_share", ""))
    if vote.get("marker") not in (None, ""):
        out.update(marker_id=vote["marker"], marker_votes=vote.get("marker_votes", ""),
                   marker_vote_share=vote.get("marker_vote_share", ""))
    return out


def _detected(row):
    """Whether a tracking row was a detection. A CSV from before the column
    existed says nothing, and every row counts (memory/60)."""
    value = row.get("detected")
    if value in (None, ""):
        return True
    n = _number(value)
    return bool(n) if n is not None else True


def _sec(frame, fps):
    return round(float(frame) / float(fps), 3) if fps and frame is not None else ""


def tracking_export(rows, by_track, fps, start=None):
    """The worker's tracking rows as the Tracking CSV columns."""
    from apps.videos import timing

    out = []
    for row in rows or []:
        frame = _number(_first(row, _FRAME_KEYS))
        sec = _sec(frame, fps)
        tid = _key(_first(row, _ID_KEYS))
        det = row.get("detected")
        line = {
            "frame": int(frame) if frame is not None else "",
            "timestamp": timing.timestamp_at(start, sec) if start and sec != "" else "",
            "clip_sec": sec,
            "track_id": int(_number(tid)) if tid and _number(tid) is not None else tid,
            "detected": "" if det in (None, "") else int(_number(det) or 0),
            **{k: row.get(k, "") for k in ("x1", "y1", "x2", "y2", "cx", "cy")},
            # A predicted row repeats its last detection's confidence in older
            # files; it was not this frame's, so it is not reported as one.
            "confidence": row.get("confidence", "") if _detected(row) else "",
        }
        line.update(identity(by_track.get(tid) if tid else None))
        line["marker_method"] = row.get("bee_id_method", "") or ""
        line["marker_confidence"] = row.get("bee_id_confidence", "") if line["marker_id"] != "" else ""
        out.append(line)
    return out


def detections_export(rows, fps, start=None):
    """The worker's raw detections as the Detections CSV columns."""
    from apps.videos import timing

    out = []
    for row in rows or []:
        frame = _number(_first(row, _FRAME_KEYS))
        sec = _sec(frame, fps)
        out.append({
            "frame": int(frame) if frame is not None else "",
            "timestamp": timing.timestamp_at(start, sec) if start and sec != "" else "",
            "clip_sec": sec,
            **{k: row.get(k, "") for k in ("x1", "y1", "x2", "y2", "cx", "cy", "confidence")},
            "class": row.get("taxon", "") or row.get("class", ""),
        })
    return out


def track_rows(tracking_rows, by_track=None, fps=None, start=None):
    """One row per track: when it was first and last DETECTED, and what it was."""
    from apps.videos import timing

    by_track = by_track or {}
    spans = {}
    for row in tracking_rows or []:
        tid = _key(_first(row, _ID_KEYS))
        frame = _number(_first(row, _FRAME_KEYS))
        if tid is None or frame is None:
            continue
        s = spans.setdefault(tid, None)
        if not _detected(row):
            continue          # a lost insect's predicted position is not when it was seen
        if s is None:
            spans[tid] = s = {"first": frame, "last": frame}
        s["first"], s["last"] = min(s["first"], frame), max(s["last"], frame)
    # A track that voted but left no tracking rows still belongs in the table.
    for tid in by_track:
        spans.setdefault(str(tid), None)

    def order(t):
        n = _number(t)
        return (n is None, n if n is not None else 0, str(t))

    rows = []
    for tid in sorted(spans, key=order):
        s = spans[tid]
        row = dict.fromkeys(TRACK_FIELDS, "")
        n = _number(tid)
        row["track_id"] = int(n) if n is not None and n == int(n) else tid
        if s and fps:
            first, last = _sec(s["first"], fps), _sec(s["last"], fps)
            row.update(start_clip_sec=first, end_clip_sec=last,
                       duration_sec=round(last - first, 3),
                       start_time=timing.timestamp_at(start, first) if start else "",
                       end_time=timing.timestamp_at(start, last) if start else "")
        row.update(identity(by_track.get(tid)))
        rows.append(row)
    return rows


def batch_track_rows(sources, read_csv, provenance):
    """Every clip's tracks, with where they came from.

    ``sources`` are ``aggregate.collect_sources`` dicts; ``read_csv`` reads a
    processed-bucket CSV to dict rows; ``provenance`` gives a clip's columns.
    """
    from apps.videos import timing

    out = []
    for src in sources:
        result = src.get("result") or {}
        path = result.get("tracking_csv_path")
        by_track = by_track_of(result)
        if not path and not by_track:
            continue
        start = timing.clip_start(src["video"]) if src.get("video") else None
        rows = track_rows(read_csv(path) if path else [], by_track, src.get("fps"), start)
        prov = provenance(src)
        for row in rows:
            out.append({**prov, **row})
    return out


def primitive_with_identity(kind, rows, by_track, fps=None, start=None, reviews=None):
    """Saved events / interactions rows as their CSV columns, with the insect's
    species and marker — and, when two insects met, the other one's
    (``with_species``, ``with_marker_id``). A reference has no identity.

    ``reviews`` (``review.event_summaries``) fills an event's review columns."""
    from . import primitives
    from .review import event_key, review_columns

    saved = primitives.normalized(kind, rows)
    out = []
    for raw, row in zip(saved, primitives.export_rows(kind, rows, fps, start)):
        merged = dict(row)
        if kind == "events":
            merged.update(review_columns(reviews or {}, event_key(raw)))
        merged.update(identity(by_track.get(_key(row.get("track_id")))))
        if kind == "interactions":
            near = row.get("relation") == primitives.NEAR
            other = identity(by_track.get(_key(row.get("with")))) if near else {}
            merged.update(with_species=other.get("species", ""),
                          with_marker_id=other.get("marker_id", ""))
        out.append(merged)
    return out
