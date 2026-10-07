"""One row per track: when it was seen, and what it was.

The tracking table is one row per track per frame, and the worker's species /
marker vote rides in ``summary_stats.identification.by_track``. Neither is the
table people actually analyse: "track 12 was an Osmia lignaria, marker 4, here
from 00:12 to 00:31". This builds that, with the pipeline's own species floor
applied, for the job page, its download and the batch download — so species
and marker IDs are in the data without a separate per-crop votes file.

The same identity columns are joined onto tracking, events and interaction
rows (``identity_for_row``), keyed by their track id.
"""

from __future__ import annotations

from datetime import timedelta

UNIDENTIFIED = "unidentified"

IDENTITY_FIELDS = ["species", "species_confidence", "species_votes",
                   "species_vote_share", "species_best_guess",
                   "marker_id", "marker_votes", "marker_vote_share"]
TRACK_FIELDS = (["track_id", "class", "first_frame", "last_frame", "start_sec",
                 "end_sec", "duration_sec", "frames_seen"] + IDENTITY_FIELDS)

_ID_KEYS = ("track_id", "track", "tid", "object_id")
_FRAME_KEYS = ("frame_number", "frame", "frame_num", "frame_idx")
_CLASS_KEYS = ("class", "class_name", "label", "taxon_class")
# Track ids on an events / interactions row: the insect, and its partner.
_SUBJECT_KEYS = ("track_id", "subject", "organism_track_id", "a", "entity1_id")


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


def species_floor(steps):
    """The Identify Species node's minimum mean confidence, 0 when none."""
    for s in steps or []:
        if s.get("block_type") == "identify.species":
            return _number((s.get("config") or {}).get("min_mean_confidence")) or 0.0
    return 0.0


def by_track_of(result):
    """``{track_id: vote}`` from a job result (dict or JobResult)."""
    stats = (result.get("summary_stats") if isinstance(result, dict)
             else getattr(result, "summary_stats", None)) or {}
    return (stats.get("identification") or {}).get("by_track") or {}


def identity(vote, floor=0.0):
    """Identity columns for one track's vote. Below ``floor`` the species is
    unidentified and the model's call is kept as the best guess."""
    vote = vote or {}
    out = dict.fromkeys(IDENTITY_FIELDS, "")
    taxon, conf = vote.get("taxon"), _number(vote.get("taxon_confidence"))
    if taxon:
        out.update(species_confidence=round(conf, 4) if conf is not None else "",
                   species_votes=vote.get("taxon_votes", ""),
                   species_vote_share=vote.get("taxon_vote_share", ""))
        if floor and (conf is None or conf < floor):
            out.update(species=UNIDENTIFIED, species_best_guess=taxon)
        else:
            out["species"] = taxon
    if vote.get("marker") not in (None, ""):
        out.update(marker_id=vote["marker"], marker_votes=vote.get("marker_votes", ""),
                   marker_vote_share=vote.get("marker_vote_share", ""))
    return out


def has_identity(by_track):
    return any(v.get("taxon") or v.get("marker") not in (None, "")
               for v in (by_track or {}).values())


def with_identity(rows, by_track, floor=0.0, keys=_SUBJECT_KEYS):
    """``rows`` with the identity columns of each row's track appended — the
    tracking table's own species columns are replaced, so the floor applies."""
    if not has_identity(by_track):
        return rows
    out = []
    for row in rows:
        tid = _key(_first(row, keys))
        merged = {k: v for k, v in row.items()
                  if k not in ("taxon", "taxon_confidence", "taxon_votes",
                               "taxon_vote_share", "taxon_crops", "bee_id",
                               "bee_id_method", "bee_id_confidence", "bee_id_votes",
                               "bee_id_vote_share")}
        merged.update(identity(by_track.get(tid) if tid else None, floor))
        out.append(merged)
    return out


def track_rows(tracking_rows, by_track=None, floor=0.0, fps=None):
    """One row per track from the per-frame tracking rows (dicts)."""
    by_track = by_track or {}
    spans = {}
    for row in tracking_rows or []:
        tid = _key(_first(row, _ID_KEYS))
        frame = _number(_first(row, _FRAME_KEYS))
        if tid is None or frame is None:
            continue
        s = spans.setdefault(tid, {"first": frame, "last": frame, "n": 0, "classes": {}})
        s["first"], s["last"] = min(s["first"], frame), max(s["last"], frame)
        s["n"] += 1
        cls = _first(row, _CLASS_KEYS)
        if cls is not None:
            s["classes"][cls] = s["classes"].get(cls, 0) + 1
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
        if s:
            row.update(first_frame=int(s["first"]), last_frame=int(s["last"]),
                       frames_seen=s["n"])
            if s["classes"]:
                row["class"] = max(s["classes"], key=s["classes"].get)
            if fps:
                row.update(start_sec=round(s["first"] / fps, 3),
                           end_sec=round(s["last"] / fps, 3),
                           duration_sec=round((s["last"] - s["first"]) / fps, 3))
        row.update(identity(by_track.get(tid), floor))
        rows.append(row)
    return rows


def batch_track_rows(sources, read_csv, provenance):
    """Every clip's tracks, with where they came from and when they started.

    ``sources`` are ``aggregate.collect_sources`` dicts; ``read_csv`` reads a
    processed-bucket CSV to dict rows; ``provenance`` gives a clip's columns.
    """
    out = []
    for src in sources:
        result = src.get("result") or {}
        path = result.get("tracking_csv_path")
        by_track = by_track_of(result)
        if not path and not by_track:
            continue
        floor = species_floor(getattr(src.get("run"), "steps", None))
        rows = track_rows(read_csv(path) if path else [], by_track, floor, src.get("fps"))
        prov = provenance(src)
        for row in rows:
            merged = dict(prov)
            start = row.get("start_sec")
            merged["absolute_time"] = (
                (src["recorded_at"] + timedelta(seconds=start)).isoformat()
                if src.get("recorded_at") and isinstance(start, (int, float)) else "")
            merged.update(row)
            out.append(merged)
    return out


def primitive_with_identity(kind, rows, by_track, floor=0.0):
    """Events / interactions rows with the insect's species and marker.

    An event's subject is the insect; an interaction's ``a`` is, and so is its
    ``b`` when two insects met — that partner's identity goes in ``b_`` columns.
    A reference (a tube, a flower) has no identity and gets blanks.
    """
    from .primitives import tidy_references

    rows = tidy_references(kind, rows)
    if not has_identity(by_track):
        return rows
    subject = "subject" if kind == "events" else "a"
    out = []
    for row in rows:
        merged = dict(row)
        kind_key = f"{subject}_kind" if kind == "events" else "a_kind"
        is_insect = row.get(kind_key, "organism") in ("organism", "", None)
        tid = _key(row.get(subject)) if is_insect else None
        merged.update(identity(by_track.get(tid) if tid else None, floor))
        if kind == "interactions" and row.get("b_kind") == "organism":
            other = identity(by_track.get(_key(row.get("b"))), floor)
            merged.update(b_species=other["species"], b_marker_id=other["marker_id"])
        out.append(merged)
    return out
