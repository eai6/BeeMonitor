"""The two things the analyze layer measures (memory/35, rules from memory/60).

**Event** — an insect came out of a reference (exit), went into one (enter), or
started and ended in one without leaving it (stay).
**Interaction** — an insect was in a reference, or near another insect, for a
span of time.

Both are computed here, on the web, from the tracking table and the references
the PIPELINE defines — one source of truth. The GPU worker measures (detections,
tracks); it no longer writes events or interactions of its own.

Rows are SAVED in frames (``frame``, ``start_frame`` …) and exported in time
(``timestamp``, ``clip_sec`` …) by ``export_rows``: a clip's start time can be
corrected later (a site's time zone set), and then every download follows
without anything being recounted.
"""

import logging
import re

logger = logging.getLogger(__name__)

# The CSV columns people download, in order. Identity and provenance columns
# are added around these by the exporters.
EVENT_FIELDS = ("timestamp", "clip_sec", "track_id", "action", "reference",
                "duration_sec", "at_edge", "ml_confidence")

INTERACTION_FIELDS = ("start_time", "end_time", "start_clip_sec", "end_clip_sec",
                      "duration_sec", "track_id", "with", "relation", "min_distance_bl")

EXIT, ENTER, STAY = "exit", "enter", "stay"

# How the two sides of an interaction were together.
IN_REFERENCE = "in_reference"   # the insect's centre was in the reference (plus padding)
NEAR = "near"                   # two insects within the radius, in body lengths


def _sec(frame, fps):
    """Frame number to seconds into the clip, or None without a usable rate."""
    try:
        return round(int(frame) / float(fps), 3) if fps else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# ── Saved rows, from the ops passes ──────────────────────────────────────────

def events_from_computed(events, fps):
    """``ops.compute_events`` output as saved event rows."""
    rows = []
    for ev in events or []:
        row = {
            "frame": int(ev["frame"]),
            "track_id": ev["track"],
            "action": ev["action"],
            "reference": ev["reference"],
            "at_edge": ev.get("at_edge"),
        }
        if ev["action"] == STAY:
            row["end_frame"] = int(ev["end_frame"])
            row["duration_sec"] = (round((ev["end_frame"] - ev["frame"]) / float(fps), 3)
                                   if fps else None)
        rows.append(row)
    return rows


def interactions_from_episodes(episodes, fps):
    """Each reference episode as one ``in_reference`` interaction."""
    return [{
        "start_frame": int(ep["start_frame"]),
        "end_frame": int(ep["end_frame"]),
        # Detected frames held, not end minus start: a gap the detector missed
        # was not time we saw the insect there.
        "duration_sec": round(ep["frames"] / float(fps), 3) if fps else None,
        "track_id": ep["track"],
        "with": ep["reference"],
        "relation": IN_REFERENCE,
    } for ep in episodes or []]


def interactions_from_proximity(episodes, fps):
    """Each pairwise spell as one ``near`` interaction."""
    return [{
        "start_frame": int(ep["start_frame"]),
        "end_frame": int(ep["end_frame"]),
        "duration_sec": round(ep["frames"] / float(fps), 3) if fps else None,
        "track_id": ep["track"],
        "with": ep["partner"],
        "relation": NEAR,
        "min_distance_bl": round(ep["min_distance_bl"], 3),
    } for ep in episodes or []]


# ── Rows saved before memory/60 ──────────────────────────────────────────────
#
# Runs counted under the old rules keep their saved rows (they are left as they
# are, labelled). Their keys are mapped onto the current ones so every page and
# download reads one shape. Nothing is converted that can't be: an old
# ``min_distance`` was a fraction of frame width, not body lengths, so it is
# dropped rather than relabelled.

_OLD_ACTIONS = {"entry": ENTER, "enter": ENTER, "exit": EXIT, "stay": STAY}
_OLD_RELATIONS = {"inside": IN_REFERENCE, "proximity": NEAR}


def normalize_event(row):
    if "track_id" in row and "reference" in row:
        return row
    out = {
        "frame": row.get("frame"),
        "track_id": row.get("subject", ""),
        "action": _OLD_ACTIONS.get(str(row.get("action", "")).lower(), row.get("action")),
        "reference": reference_id(row.get("target")),
        "at_edge": None,
    }
    return out


def normalize_interaction(row):
    if "track_id" in row and "with" in row:
        return row
    to_reference = row.get("b_kind") in ("reference", "nest")
    return {
        "start_frame": row.get("start_frame"),
        "end_frame": row.get("end_frame"),
        "duration_sec": row.get("duration_sec"),
        "track_id": row.get("a", ""),
        "with": reference_id(row.get("b")) if to_reference else row.get("b"),
        "relation": _OLD_RELATIONS.get(row.get("relation"),
                                       IN_REFERENCE if to_reference else NEAR),
    }


def normalized(kind, rows):
    fn = normalize_event if kind == "events" else normalize_interaction
    return [fn(r) for r in rows or []]


# ── Rollups ──────────────────────────────────────────────────────────────────

def summarize_events(rows):
    """Counts an event table answers without further computation."""
    per_ref = {}
    for r in rows:
        bucket = per_ref.setdefault(str(r["reference"]), {
            "id": str(r["reference"]), "label": str(r["reference"]),
            ENTER: 0, EXIT: 0, STAY: 0, "subjects": set(),
        })
        if r["action"] in (ENTER, EXIT, STAY):
            bucket[r["action"]] += 1
        if r["track_id"] not in ("", None):
            bucket["subjects"].add(r["track_id"])

    per_reference = sorted(
        ({"id": b["id"], "label": b["label"], "enter": b[ENTER], "exit": b[EXIT],
          "stay": b[STAY], "events": b[ENTER] + b[EXIT] + b[STAY],
          "subjects": len(b["subjects"])}
         for b in per_ref.values()),
        key=lambda r: (-r["events"], r["id"]),
    )
    return {
        "event_count": len(rows),
        "enter_count": sum(1 for r in rows if r["action"] == ENTER),
        "exit_count": sum(1 for r in rows if r["action"] == EXIT),
        "stay_count": sum(1 for r in rows if r["action"] == STAY),
        "subjects": len({r["track_id"] for r in rows if r["track_id"] not in ("", None)}),
        "per_reference": per_reference,
        "rows": rows,
    }


def summarize_interactions(rows):
    """Counts an interaction table answers, including time in each reference."""
    durations = [r["duration_sec"] for r in rows if r.get("duration_sec") is not None]
    in_reference = [r for r in rows if r["relation"] == IN_REFERENCE]

    per_ref = {}
    for r in in_reference:
        bucket = per_ref.setdefault(str(r["with"]), {
            "id": str(r["with"]), "label": str(r["with"]),
            "interactions": 0, "partners": set(), "duration_sec": 0.0,
        })
        bucket["interactions"] += 1
        if r["track_id"] not in ("", None):
            bucket["partners"].add(r["track_id"])
        bucket["duration_sec"] += float(r.get("duration_sec") or 0)

    per_reference = sorted(
        ({"id": b["id"], "label": b["label"], "interactions": b["interactions"],
          "partners": len(b["partners"]), "duration_sec": round(b["duration_sec"], 2)}
         for b in per_ref.values()),
        key=lambda r: (-r["interactions"], r["id"]),
    )
    return {
        "interaction_count": len(rows),
        "organism_organism": len(rows) - len(in_reference),
        "organism_reference": len(in_reference),
        "total_duration_sec": round(sum(durations), 2) if durations else None,
        "per_reference": per_reference,
        "rows": rows,
    }


# ── Export: frames -> time ───────────────────────────────────────────────────

def export_rows(kind, rows, fps, start=None):
    """Saved rows as the CSV columns people download.

    ``start`` is the clip's recorded start (``apps.videos.timing.clip_start``):
    None when unknown, and then the timestamp columns are blank while the
    seconds-into-clip columns are still filled.
    """
    from apps.videos import timing

    def _at(frame):
        sec = _sec(frame, fps) if frame not in (None, "") else None
        return sec, (timing.timestamp_at(start, sec) if sec is not None and start else "")

    out = []
    for row in normalized(kind, rows):
        if kind == "events":
            sec, ts = _at(row.get("frame"))
            out.append({
                "timestamp": ts, "clip_sec": sec,
                "track_id": row.get("track_id", ""),
                "action": row.get("action", ""),
                "reference": row.get("reference", ""),
                "duration_sec": row.get("duration_sec", ""),
                "at_edge": _flag(row.get("at_edge")),
                "ml_confidence": row.get("ml_confidence", ""),
            })
        else:
            s_sec, s_ts = _at(row.get("start_frame"))
            e_sec, e_ts = _at(row.get("end_frame"))
            out.append({
                "start_time": s_ts, "end_time": e_ts,
                "start_clip_sec": s_sec, "end_clip_sec": e_sec,
                "duration_sec": row.get("duration_sec", ""),
                "track_id": row.get("track_id", ""),
                "with": row.get("with", ""),
                "relation": row.get("relation", ""),
                "min_distance_bl": row.get("min_distance_bl", ""),
            })
    return out


def _flag(value):
    """true / false, or blank when it could not be known."""
    if value is None or value == "":
        return ""
    return "true" if value else "false"


# ── Reference ids ────────────────────────────────────────────────────────────

_REFERENCE_ID = re.compile(r"^(?:nest|region|reference|tube)[ _-]?(.+)$", re.IGNORECASE)


def reference_id(value):
    """Every reference's id, in one vocabulary: ``reference_<n>``."""
    if value is None:
        return value
    text = str(value).strip()
    m = _REFERENCE_ID.match(text)
    if m:
        return f"reference_{m.group(1)}"
    if text.isdigit():
        return f"reference_{text}"
    return value
