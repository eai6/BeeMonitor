"""Reviewing a period across clips, and the trips in it (memory/62 §4, §4b).

A period is one device (or one site) between two local date-times. Every clip
in it with a known recording start and a finished run that has an Events step
is placed on ONE time axis, in the clip's local wall-clock time, so an exit
in one clip and the entry in a later one sit on the same tube lane.

Trips are suggestions, never results: each exit is paired with the next entry
into the same tube later in the period (the same marker first, when markers
were read). A trip becomes data only when a reviewer calls it
(``TripReview``), and only reviewed pairs are ever exported.
"""

from __future__ import annotations

from datetime import datetime, timedelta

MAX_CLIPS = 200
MAX_DAYS = 7


def parse_local(value):
    """A from / to value as a naive local datetime, or None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace(" ", "T"))
    except ValueError:
        return None
    return dt.replace(tzinfo=None)


def local_start(video):
    """The clip's start as a naive local wall-clock, or None when unknown."""
    from apps.videos import timing

    start = timing.clip_start(video)
    return None if start is None else start.replace(tzinfo=None)


def visible_video_ids(user, candidate_ids):
    """Of these clips, the ones the user may see: their own or shared devices'
    (``Video.accessible``), or clips of batches they were invited to."""
    from apps.analysis.models import Job
    from apps.videos.models import Video

    from .review import member_job_ids

    ids = set(Video.accessible(user).filter(pk__in=candidate_ids).values_list("pk", flat=True))
    ids |= set(Job.objects.filter(pk__in=member_job_ids(user), video_id__in=candidate_ids)
               .values_list("video_id", flat=True))
    return ids


def runs_for_videos(videos):
    """``{video pk: (job, run)}``: each clip's latest completed run that has an
    Events step, found through the clip's finished GPU jobs."""
    from apps.analysis.models import Job

    from .aggregate import analyzer_outputs, run_job_id
    from .models import PipelineRun

    jobs = list(Job.objects.filter(video__in=videos, status="completed")
                .select_related("result").order_by("-created_at"))
    if not jobs:
        return {}
    when = [j.created_at for j in jobs if j.created_at]
    runs = PipelineRun.objects.filter(
        status=PipelineRun.Status.COMPLETED,
        user_id__in={j.user_id for j in jobs},
        started_at__gte=min(when) - timedelta(days=1),
        started_at__lte=max(when) + timedelta(days=1))
    by_job = {}
    for run in runs:
        jid = run_job_id(run)
        if jid is not None and any(k == "events" for k, _o in analyzer_outputs(run)):
            by_job.setdefault(jid, run)
    out = {}
    for job in jobs:                        # newest first
        if job.video_id not in out and job.pk in by_job:
            out[job.video_id] = (job, by_job[job.pk])
    return out


def gather(user, device=None, site=None, start=None, end=None):
    """Every reviewable clip of a device or site in [start, end).

    Returns ``{"clips": [...], "skipped": [{"title", "reason"}], "notes": [...],
    "start", "end"}``; each clip is ``{video, job, run, start (naive local),
    offset (seconds from the period start), duration, fps, can_review}`` in
    time order."""
    from apps.videos.models import Video

    from .ops import fps_with_source
    from .review import can_review

    notes = []
    if end - start > timedelta(days=MAX_DAYS):
        end = start + timedelta(days=MAX_DAYS)
        notes.append(f"A period is at most {MAX_DAYS} days; showing the first {MAX_DAYS}.")
    qs = Video.objects.select_related("device", "site")
    qs = qs.filter(device=device) if device is not None else qs.filter(site=site)
    # recorded_at is a UTC instant (or a file name's wall-clock stored as UTC):
    # a day either side covers every time zone; the exact cut is in local time.
    candidates = list(qs.filter(recorded_at__gte=start - timedelta(days=1),
                                recorded_at__lte=end + timedelta(days=1)))
    visible = visible_video_ids(user, [v.pk for v in candidates])
    candidates = [v for v in candidates if v.pk in visible]

    placed, skipped = [], []
    for v in candidates:
        s = local_start(v)
        if s is None:
            skipped.append({"title": v.title, "reason": "recording time unknown"})
            continue
        if start <= s < end:
            placed.append((s, v))
    placed.sort(key=lambda p: (p[0], p[1].pk))
    if len(placed) > MAX_CLIPS:
        notes.append(f"A period shows at most {MAX_CLIPS} clips; narrow the range to see the rest.")
        placed = placed[:MAX_CLIPS]

    runs = runs_for_videos([v for _s, v in placed])
    clips = []
    for s, v in placed:
        if v.pk not in runs:
            skipped.append({"title": v.title, "reason": "no finished run with an Events step"})
            continue
        job, run = runs[v.pk]
        result = getattr(job, "result", None)
        fps = fps_with_source((result.summary_stats if result else None) or {}, v)[0]
        clips.append({"video": v, "job": job, "run": run, "start": s,
                      "offset": (s - start).total_seconds(),
                      "duration": float(v.duration_seconds or 0), "fps": float(fps or 25),
                      "can_review": can_review(user, run)})
    return {"clips": clips, "skipped": skipped, "notes": notes, "start": start, "end": end}


def clip_events(clip, user=None):
    """A gathered clip's events with their period time, identity and votes:
    ``{key, frame, t (s from period start), clock, track_id, action,
    reference, end_frame, at_edge, species, marker, mine, accepted,
    rejected}``."""
    from . import primitives, review, tracks
    from .aggregate import analyzer_outputs

    run, fps = clip["run"], clip["fps"]
    by_track = tracks.by_track_of(clip["job"].result) if getattr(clip["job"], "result", None) else {}
    summaries = review.event_summaries(run)
    mine = review.user_votes(run, user) if user is not None else {}
    out = []
    for raw in (r for k, o in analyzer_outputs(run) if k == "events" for r in (o.get("rows") or [])):
        e = primitives.normalize_event(raw)
        try:
            frame = int(e["frame"])
        except (TypeError, ValueError, KeyError):
            continue
        key = review.event_key(e)
        ident = tracks.identity(by_track.get(str(e.get("track_id"))))
        sec = frame / fps if fps else 0.0
        s = summaries.get(key) or {}
        out.append({
            "key": key, "frame": frame, "t": round(clip["offset"] + sec, 3),
            "clock": (clip["start"] + timedelta(seconds=sec)).isoformat(timespec="milliseconds"),
            "track_id": e.get("track_id"), "action": e.get("action"),
            "reference": str(e.get("reference") or ""), "end_frame": e.get("end_frame"),
            "at_edge": e.get("at_edge"), "species": ident.get("species", ""),
            "species_confidence": ident.get("species_confidence", ""),
            "marker": ident.get("marker_id", ""), "mine": mine.get(key),
            "accepted": s.get("accepted", 0), "rejected": s.get("rejected", 0)})
    out.sort(key=lambda e: e["t"])
    return out


def suggest_trips(events):
    """Each exit → the next entry into the same tube later in the period.

    ``events`` are ``(clip_index, event)`` pairs from every clip. When the exit
    and an entry both carry a marker, the next entry by the same marker wins;
    otherwise the next entry by any track. Never an entry earlier than the exit.
    """
    entries = {}
    for ci, e in events:
        if e["action"] == "enter":
            entries.setdefault(e["reference"], []).append((ci, e))
    for lst in entries.values():
        lst.sort(key=lambda p: p[1]["t"])
    trips = []
    for ci, e in sorted(events, key=lambda p: p[1]["t"]):
        if e["action"] != "exit":
            continue
        later = [p for p in entries.get(e["reference"], []) if p[1]["t"] > e["t"]]
        if not later:
            continue
        pick = None
        if e.get("marker"):
            pick = next((p for p in later if p[1].get("marker") == e["marker"]), None)
        pick = pick or later[0]
        trips.append({"exit": (ci, e), "entry": pick,
                      "same_marker": bool(e.get("marker")) and pick[1].get("marker") == e["marker"],
                      "gap_sec": round(pick[1]["t"] - e["t"], 1)})
    return trips


# ── Trip votes ───────────────────────────────────────────────────────────────

def set_trip_vote(user, exit_run, exit_key, entry_run, entry_key, verdict, reason="", relinked=False):
    """Store, change or (verdict None) take back one reviewer's call on the
    trip starting at this exit. Raises ValueError on a bad request."""
    from .models import TripReview
    from .review import TRIP_REASONS

    if verdict in (None, "", "none"):
        TripReview.objects.filter(exit_run=exit_run, exit_key=exit_key, user=user).delete()
        return None
    if verdict not in TripReview.Verdict.values:
        raise ValueError("verdict must be accepted or rejected")
    reason = (reason or "").strip().lower()
    if reason and reason not in TRIP_REASONS:
        raise ValueError("unknown reason")
    if relinked:
        reason = "relinked"
    if verdict == "accepted" and not entry_key:
        raise ValueError("an accepted trip needs its entry")

    def _split(key):
        action, reference, track, frame = (str(key).split("|") + ["", "", "", ""])[:4]
        try:
            frame = int(frame)
        except (TypeError, ValueError):
            frame = None
        return reference, track, frame

    ref, exit_track, exit_frame = _split(exit_key)
    _r, entry_track, entry_frame = _split(entry_key) if entry_key else ("", "", None)
    TripReview.objects.update_or_create(
        exit_run=exit_run, exit_key=exit_key, user=user,
        defaults={"entry_run": entry_run if entry_key else None, "entry_key": entry_key or "",
                  "reference": ref, "exit_track": exit_track, "exit_frame": exit_frame,
                  "entry_track": entry_track or "", "entry_frame": entry_frame,
                  "verdict": verdict, "reason": reason, "relinked": bool(relinked)})
    return verdict


TRIP_FIELDS = ["reference", "exit_time", "entry_time", "duration_sec", "exit_clip", "exit_track",
               "entry_clip", "entry_track", "species", "marker_id", "review", "reviewed_by",
               "review_notes"]


def trip_rows(trip_reviews):
    """One row per reviewed (exit, entry) pair — only what reviewers called."""
    from apps.analysis.models import JobResult

    from . import tracks
    from .aggregate import run_job_id, run_video_id
    from .ops import fps_with_source
    from .review import combined
    from apps.videos.models import Video

    reviews = list(trip_reviews.select_related("user", "exit_run", "entry_run"))
    runs = {}
    for r in reviews:
        for run in (r.exit_run, r.entry_run):
            if run is not None:
                runs[run.pk] = run
    videos = {v.pk: v for v in Video.everything.filter(
        pk__in=[run_video_id(r) for r in runs.values() if run_video_id(r)])}
    results = {j: res for j, res in JobResult.objects.filter(
        job_id__in=[run_job_id(r) for r in runs.values() if run_job_id(r)])
        .values_list("job_id", "summary_stats")}

    def _clip(run):
        if run is None:
            return None, None, 25.0, {}
        video = videos.get(run_video_id(run))
        stats = results.get(run_job_id(run)) or {}
        fps = fps_with_source(stats, video)[0] if video else 25.0
        by_track = tracks.by_track_of({"summary_stats": stats})
        return video, local_start(video) if video else None, float(fps or 25), by_track

    def _time(start, fps, frame):
        if start is None or frame is None:
            return "", None
        t = start + timedelta(seconds=frame / fps)
        return t.isoformat(timespec="milliseconds"), t

    groups = {}
    for r in reviews:
        groups.setdefault((r.exit_run_id, r.exit_key, r.entry_run_id, r.entry_key), []).append(r)
    rows = []
    for (_er, _ek, _nr, _nk), rs in groups.items():
        first = rs[0]
        ev, es, efps, by_track = _clip(first.exit_run)
        nv, ns, nfps, _bt = _clip(first.entry_run)
        exit_iso, exit_t = _time(es, efps, first.exit_frame)
        entry_iso, entry_t = _time(ns, nfps, first.entry_frame)
        ident = tracks.identity(by_track.get(str(first.exit_track)))
        notes = []
        for r in rs:
            if r.reason and r.reason not in notes:
                notes.append(r.reason)
        rows.append({
            "reference": first.reference, "exit_time": exit_iso, "entry_time": entry_iso,
            "duration_sec": (round((entry_t - exit_t).total_seconds(), 2)
                             if exit_t and entry_t else ""),
            "exit_clip": ev.title if ev else "", "exit_track": first.exit_track,
            "entry_clip": nv.title if nv else "", "entry_track": first.entry_track,
            "species": ident.get("species", ""), "marker_id": ident.get("marker_id", ""),
            "review": combined([r.verdict for r in rs]),
            "reviewed_by": ";".join(sorted(r.user.username for r in rs)),
            "review_notes": "; ".join(notes)})
    rows.sort(key=lambda r: (r["exit_time"], r["reference"]))
    return rows
