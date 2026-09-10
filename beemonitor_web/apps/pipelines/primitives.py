"""The two things the analyze layer actually measures.

Everything an ecologist asks of this system reduces to two facts about a clip:

**Event** — something entered or exited something, at a moment.
**Interaction** — two things were together for a span of time.

Foraging trips, visitation counts, dwell times and insect-to-insect encounters
are all *reads* over those two tables, not separate computations. That matters
because they used to be separate computations: one physical fact (a bee at tube
3 from frame 400 to 700) was produced three times by three code paths —
``ops.compute_visitation`` over the user's ROI, the GPU's
``InteractionAnalyzer`` over detected nests, and the GPU's Entry/Exit event
classifier — each with its own column names, and no way to hold them to
agreeing with each other.

The two primitives are one pass. An interaction's start and end *are* its enter
and exit events, so ``ops.compute_episodes`` runs once and both tables project
out of it. The GPU's own CSVs are normalised into the same two schemas here, so
a downstream consumer never has to care which side of the wire a row came from
-- only what ``source`` says about it.
"""

import logging

logger = logging.getLogger(__name__)

# Canonical column order. These are the CSV headers users download and write
# scripts against, so they are part of the contract: append, never reorder.
EVENT_FIELDS = ("frame", "time_sec", "subject", "subject_kind",
                "action", "target", "target_kind", "source")

INTERACTION_FIELDS = ("start_frame", "end_frame", "start_sec", "end_sec",
                      "duration_sec", "a", "a_kind", "b", "b_kind",
                      "relation", "min_distance", "source")

ENTER, EXIT = "enter", "exit"
ORGANISM, REFERENCE = "organism", "reference"

# How the two sides of an interaction were judged to be together.
INSIDE = "inside"        # the subject's centroid was within the reference's shape
PROXIMITY = "proximity"  # the two were within a distance threshold of each other

# Where a row came from. Following the provenance rule from the consistency
# audit: a number that rests on someone else's assumption has to say so.
DERIVED, GPU = "derived", "gpu"

# The GPU writes Entry/Exit; the primitive spells them lowercase and calls the
# thing they happen to a target rather than a nest.
_GPU_ACTIONS = {"entry": ENTER, "exit": EXIT}


def _sec(frame, fps):
    """Frame number to seconds, or None when there is no usable rate."""
    try:
        return round(int(frame) / float(fps), 3) if fps else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ── Projections from the episode pass ────────────────────────────────────────

def events_from_episodes(episodes, fps, source=DERIVED):
    """An episode's two ends, as enter/exit events.

    A bee inside tube 3 from frame 400 to 700 entered at 400 and exited at 700.
    Emitting both from the same record is what keeps an event table and an
    interaction table from ever disagreeing about the same clip.
    """
    rows = []
    for ep in episodes or []:
        for frame, action in ((ep["start_frame"], ENTER), (ep["end_frame"], EXIT)):
            rows.append({
                "frame": int(frame),
                "time_sec": _sec(frame, fps),
                "subject": ep["track"],
                "subject_kind": ORGANISM,
                "action": action,
                "target": ep["reference"],
                "target_kind": REFERENCE,
                "source": source,
            })
    rows.sort(key=lambda r: (r["frame"], r["action"], str(r["subject"])))
    return rows


def interactions_from_episodes(episodes, fps, source=DERIVED):
    """Each episode as one organism-to-reference interaction."""
    rows = []
    for ep in episodes or []:
        start, end = int(ep["start_frame"]), int(ep["end_frame"])
        rows.append({
            "start_frame": start,
            "end_frame": end,
            "start_sec": _sec(start, fps),
            "end_sec": _sec(end, fps),
            # Frames held, not end-minus-start: a track the detector dropped
            # mid-episode was not present for the gap, and counting the span
            # would inflate every dwell time by exactly the frames we missed.
            "duration_sec": round(ep["frames"] / float(fps), 3) if fps else None,
            "a": ep["track"],
            "a_kind": ORGANISM,
            "b": ep["reference"],
            "b_kind": REFERENCE,
            "relation": INSIDE,
            "source": source,
        })
    rows.sort(key=lambda r: (r["start_frame"], str(r["a"])))
    return rows


def interactions_from_proximity(episodes, fps, source=DERIVED):
    """Each pairwise proximity spell as one organism-to-organism interaction.

    ``min_distance`` rides along in fractions of frame width: it is the whole
    reason the threshold is resolution-independent, so a reader can see how
    close the pair actually came rather than only that they passed a cutoff.
    """
    rows = []
    for ep in episodes or []:
        start, end = int(ep["start_frame"]), int(ep["end_frame"])
        rows.append({
            "start_frame": start,
            "end_frame": end,
            "start_sec": _sec(start, fps),
            "end_sec": _sec(end, fps),
            "duration_sec": round(ep["frames"] / float(fps), 3) if fps else None,
            "a": ep["track"],
            "a_kind": ORGANISM,
            "b": ep["partner"],
            "b_kind": ORGANISM,
            "relation": PROXIMITY,
            "min_distance": round(ep.get("min_distance", 0.0), 4),
            "source": source,
        })
    rows.sort(key=lambda r: (r["start_frame"], str(r["a"]), str(r["b"])))
    return rows


# ── Normalising what the GPU already produces ────────────────────────────────

def events_from_gpu(df, fps):
    """The worker's ``events.csv`` in the Event schema, or [].

    The worker classifies Entry/Exit against the nests it detected, which is a
    different target set from the user's drawn ROI — so these rows coexist with
    the derived ones rather than replacing them, and ``target_kind`` says which
    is which.
    """
    if df is None or len(df) == 0:
        return []
    from . import ops

    cols = {
        "action": ops._pick(df, ["action", "event", "event_type"]),
        "target": ops._pick(df, ["nest", "nest_id", "reference_id", "target"]),
        "frame": ops._pick(df, list(ops._FRAME_COLS)),
        "subject": ops._pick(df, list(ops._ID_COLS)),
    }
    if cols["action"] is None or cols["frame"] is None:
        logger.info("events_from_gpu: CSV lacks action/frame columns — skipped")
        return []

    rows = []
    for _, r in df.iterrows():
        action = _GPU_ACTIONS.get(str(r[cols["action"]]).strip().lower())
        if action is None:
            continue
        try:
            frame = int(float(r[cols["frame"]]))
        except (TypeError, ValueError):
            continue
        target = r[cols["target"]] if cols["target"] else ""
        rows.append({
            "frame": frame,
            "time_sec": _sec(frame, fps),
            "subject": ops._as_native(r[cols["subject"]]) if cols["subject"] else "",
            "subject_kind": ORGANISM,
            "action": action,
            "target": ops._as_native(target),
            # The worker's targets are nests it found itself, not the user's
            # references — kept distinct so a count of ROI visits can never
            # silently absorb them.
            "target_kind": "nest",
            "source": GPU,
        })
    rows.sort(key=lambda r: (r["frame"], r["action"], str(r["subject"])))
    return rows


def interactions_from_gpu(df, fps):
    """The worker's ``interactions.csv`` in the Interaction schema, or [].

    Covers what the local episode pass cannot see: organism-to-organism
    proximity, which needs every pair in every frame and so is computed on the
    GPU while the tracks are already in memory.
    """
    if df is None or len(df) == 0:
        return []
    from . import ops

    cols = {
        "type": ops._pick(df, ["interaction_type", "type", "kind"]),
        "a": ops._pick(df, ["organism_track_id", "entity1_id", "track_id"]),
        "b": ops._pick(df, ["partner_track_id", "entity2_id"]),
        "reference": ops._pick(df, ["reference_id", "nest", "nest_id"]),
        "duration": ops._pick(df, ["duration_seconds", "duration_sec", "duration"]),
        "start": ops._pick(df, ["start_frame", "frame_start", "frame"]),
        "end": ops._pick(df, ["end_frame", "frame_end"]),
    }

    rows = []
    for _, r in df.iterrows():
        kind = str(r[cols["type"]]).strip().lower() if cols["type"] else ""
        to_reference = "reference" in kind or "nest" in kind
        partner = (r[cols["reference"]] if to_reference and cols["reference"]
                   else (r[cols["b"]] if cols["b"] else ""))

        def _num(col, cast=float):
            if not col:
                return None
            try:
                return cast(float(r[col]))
            except (TypeError, ValueError):
                return None

        start, end = _num(cols["start"], int), _num(cols["end"], int)
        duration = _num(cols["duration"])
        if duration is None and None not in (start, end) and fps:
            duration = round((end - start) / float(fps), 3)

        rows.append({
            "start_frame": start,
            "end_frame": end,
            "start_sec": _sec(start, fps) if start is not None else None,
            "end_sec": _sec(end, fps) if end is not None else None,
            "duration_sec": round(duration, 3) if duration is not None else None,
            "a": ops._as_native(r[cols["a"]]) if cols["a"] else "",
            "a_kind": ORGANISM,
            "b": ops._as_native(partner),
            "b_kind": REFERENCE if to_reference else ORGANISM,
            "relation": PROXIMITY,
            "source": GPU,
        })
    rows.sort(key=lambda r: (r["start_frame"] is None, r["start_frame"], str(r["a"])))
    return rows


# ── Rollups: the questions people actually ask, as reads ─────────────────────

def summarize_events(rows):
    """Counts an event table can answer without any further computation."""
    enters = sum(1 for r in rows if r["action"] == ENTER)
    per_target = {}
    for r in rows:
        bucket = per_target.setdefault(str(r["target"]), {
            "id": str(r["target"]), "label": str(r["target"]),
            "kind": r["target_kind"], "enter": 0, "exit": 0, "subjects": set(),
        })
        bucket[r["action"]] += 1
        if r["subject"] not in ("", None):
            bucket["subjects"].add(r["subject"])

    per_reference = sorted(
        ({"id": b["id"], "label": b["label"], "kind": b["kind"],
          "enter": b["enter"], "exit": b["exit"],
          "events": b["enter"] + b["exit"], "subjects": len(b["subjects"])}
         for b in per_target.values()),
        key=lambda r: (-r["events"], r["id"]),
    )
    return {
        "event_count": len(rows),
        "enter_count": enters,
        "exit_count": len(rows) - enters,
        "subjects": len({r["subject"] for r in rows if r["subject"] not in ("", None)}),
        "per_reference": per_reference,
        "rows": rows,
    }


def summarize_interactions(rows):
    """Counts an interaction table can answer, including the visitation view.

    ``organism_reference`` *is* the visitation count — a visit is an insect
    interacting with a reference. It is reported here rather than computed by a
    separate analyzer.
    """
    durations = [r["duration_sec"] for r in rows if r.get("duration_sec") is not None]
    to_reference = [r for r in rows if r["b_kind"] == REFERENCE]

    per_ref = {}
    for r in to_reference:
        bucket = per_ref.setdefault(str(r["b"]), {
            "id": str(r["b"]), "label": str(r["b"]),
            "interactions": 0, "partners": set(), "duration_sec": 0.0,
        })
        bucket["interactions"] += 1
        if r["a"] not in ("", None):
            bucket["partners"].add(r["a"])
        bucket["duration_sec"] += float(r.get("duration_sec") or 0)

    per_reference = sorted(
        ({"id": b["id"], "label": b["label"], "interactions": b["interactions"],
          "partners": len(b["partners"]),
          "duration_sec": round(b["duration_sec"], 2)}
         for b in per_ref.values()),
        key=lambda r: (-r["interactions"], r["id"]),
    )
    return {
        "interaction_count": len(rows),
        "organism_organism": len(rows) - len(to_reference),
        "organism_reference": len(to_reference),
        "total_duration_sec": round(sum(durations), 2) if durations else None,
        "per_reference": per_reference,
        "rows": rows,
    }


def label_references(rows, refs, key="b"):
    """Swap reference ids for their human labels, in place. Returns ``rows``.

    The primitives carry ids because ids are what stay stable across clips; the
    page wants "Tube 3". Kept as a separate step so the tables people download
    are joinable and the tables people read are legible.
    """
    labels = {}
    for ref in refs or []:
        if isinstance(ref, dict) and ref.get("id") is not None:
            labels[str(ref["id"])] = ref.get("label") or str(ref["id"])
    if not labels:
        return rows
    for row in rows:
        if key in row and str(row[key]) in labels:
            row.setdefault("%s_label" % key, labels[str(row[key])])
    return rows
