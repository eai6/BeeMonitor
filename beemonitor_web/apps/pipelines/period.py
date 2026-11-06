"""Reviewing a period across clips, and the trips in it (memory/62 §4, §4b).

A period is one device (or one site) between two local date-times. Every clip
in it with a known recording start and a finished run that has an Events step
is placed on ONE time axis, in the clip's local wall-clock time, so an exit
in one clip and the entry in a later one sit on the same tube lane.

Each exit is paired with the next entry into the same tube later in the period
(the same marker first, when markers were read). A trip is the pipeline's
suggestion until a reviewer calls it (``TripReview``); the Trips CSV has both,
as the events CSV does (``trip_rows``).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from . import runsummary

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
        gpu_job_id__in=[j.pk for j in jobs],
        started_at__gte=min(when) - timedelta(days=1),
        started_at__lte=max(when) + timedelta(days=1)).order_by("-started_at")
    by_job = {}                             # a job re-used by later runs: the newest
    for run in runsummary.slim_runs(runs):  # chosen from slim copies (memory/76)
        jid = run_job_id(run)
        if jid is not None and any(k == "events" for k, _o in analyzer_outputs(run)):
            by_job.setdefault(jid, run)
    out = {}
    for job in jobs:                        # newest first
        if job.video_id not in out and job.pk in by_job:
            out[job.video_id] = (job, by_job[job.pk])
    return _whole(out)


def _whole(out):
    """``out`` with each chosen run read whole: the period page shows their
    events with each insect's identity."""
    from .models import PipelineRun

    full = PipelineRun.objects.in_bulk([run.pk for _job, run in out.values()])
    return {vid: (job, full.get(run.pk, run)) for vid, (job, run) in out.items()}


def batch_runs(batch_id, light=False):
    """``{video pk: (job, run)}`` for a batch's completed runs with an Events
    step. A clip can have several runs (a re-run re-uses its GPU job), and a
    period opened from a batch must show, and take votes on, that batch's.
    ``light``: the jobs without their results, for thousands of clips."""
    from apps.analysis.models import Job

    from .aggregate import analyzer_outputs, run_job_id, run_video_id
    from .models import PipelineRun

    runs = {}
    for run in runsummary.slim_runs(
            PipelineRun.objects.filter(batch_id=batch_id, status=PipelineRun.Status.COMPLETED)):
        jid = run_job_id(run)
        if jid is not None and any(k == "events" for k, _o in analyzer_outputs(run)):
            runs[jid] = run
    out = {}
    jobs = Job.objects.filter(pk__in=runs, status="completed")
    for job in jobs.defer("config") if light else jobs.select_related("result"):
        if run_video_id(runs[job.pk]) == job.video_id:
            out[job.video_id] = (job, runs[job.pk])
    return out


def gather(user, device=None, site=None, start=None, end=None, prefer=None):
    """Every reviewable clip of a device or site in [start, end).

    Returns ``{"clips": [...], "skipped": [{"title", "reason"}], "notes": [...],
    "start", "end"}``; each clip is ``{video, job, run, start (naive local),
    offset (seconds from the period start), duration, fps, can_review}`` in
    time order. ``prefer`` (``{video pk: (job, run)}``, from ``batch_runs``)
    overrides the latest run for the clips it names."""
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
    shown = {v.pk for _s, v in placed}
    runs.update(_whole({vid: jr for vid, jr in (prefer or {}).items() if vid in shown}))
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


def gather_batch(user, batch_id, device=None, site=None):
    """Every clip of one unit or site in a batch, as ``gather`` but with no
    limit on dates or clips: trips are reviewed across all of them. Light: read
    the clips whole a few at a time, with ``whole_clips``."""
    from apps.videos.models import Video

    from .review import can_review

    chosen = batch_runs(batch_id, light=True)
    qs = Video.objects.select_related("device", "site").filter(pk__in=list(chosen))
    videos = list(qs.filter(device=device) if device is not None else qs.filter(site=site))
    visible = visible_video_ids(user, [v.pk for v in videos])
    placed, skipped = [], []
    for v in videos:
        s = local_start(v)
        if v.pk not in visible:
            continue
        if s is None:
            skipped.append({"title": v.title, "reason": "recording time unknown"})
        else:
            placed.append((s, v))
    placed.sort(key=lambda p: (p[0], p[1].pk))
    start = placed[0][0] if placed else datetime.now()
    clips = []
    for s, v in placed:
        job, run = chosen[v.pk]
        clips.append({"video": v, "job": job, "run": run, "start": s,
                      "offset": (s - start).total_seconds(),
                      "duration": float(v.duration_seconds or 0), "fps": 25.0,
                      "can_review": can_review(user, run)})
    end = placed[-1][0] + timedelta(hours=1) if placed else start
    return {"clips": clips, "skipped": skipped, "notes": [], "start": start, "end": end,
            "light": True}


def whole_clips(data, chunk=25):
    """Each of ``data``'s clips with its run and job read whole, a few at a
    time, so a site's thousands of clips are never all in memory (memory/76).
    Sets each clip's frame rate. Yields ``(index, clip)``."""
    from apps.analysis.models import Job

    from .models import PipelineRun
    from .ops import fps_with_source

    clips = data["clips"]
    if not data.get("light"):
        yield from enumerate(clips)
        return
    for at in range(0, len(clips), chunk):
        part = clips[at:at + chunk]
        runs = PipelineRun.objects.in_bulk([c["run"].pk for c in part])
        jobs = Job.objects.select_related("result").in_bulk([c["job"].pk for c in part])
        for i, c in enumerate(part, at):
            job = jobs[c["job"].pk]
            result = getattr(job, "result", None)
            c["fps"] = float(fps_with_source((result.summary_stats if result else None) or {},
                                             c["video"])[0] or 25)
            yield i, {**c, "job": job, "run": runs[c["run"].pk]}


def clip_events(clip, user=None):
    """A gathered clip's events with their period time, identity and votes:
    ``{key, frame, t (s from period start), clock, track_id, action,
    reference, end_frame, at_edge, species, marker, mine, accepted,
    rejected}``."""
    from . import primitives, review, tracks
    from .aggregate import analyzer_outputs

    run, fps = clip["run"], clip["fps"]
    from .retrack.run import effective_result

    result = effective_result(clip["job"], run) if getattr(clip["job"], "result", None) else None
    by_track = tracks.by_track_of(result) if result is not None else {}
    summaries = review.event_summaries(run)
    # The event's call, whoever made it (memory/74).
    calls = review.current_calls(run, user)
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
            "marker": ident.get("marker_id", ""), "mine": calls.get(key, (None,))[0],
            "called_by": calls.get(key, (None, None))[1],
            "accepted": s.get("accepted", 0), "rejected": s.get("rejected", 0)})
    out.sort(key=lambda e: e["t"])
    return out


def added_events(clip, rows):
    """A gathered clip's added events (``review.added_rows``) on the period's
    axis, in the shape of ``clip_events``, so a trip can start or end on one."""
    from .review import event_key

    fps = float(clip["fps"] or 25)
    out = []
    for a in rows:
        sec = a["frame"] / fps
        out.append({"id": a["id"], "key": event_key(a), "frame": a["frame"],
                    "end_frame": a["end_frame"], "action": a["action"],
                    "reference": str(a["reference"] or ""), "track_id": a["track_id"],
                    "by": a["user"], "t": round(clip["offset"] + sec, 3), "marker": "",
                    "clock": (clip["start"] + timedelta(seconds=sec)).isoformat(timespec="milliseconds")})
    return out


def _trips_step(pipeline):
    return next((s for s in (pipeline.steps if pipeline else None) or []
                 if s.get("block_type") == "analyze.trips"), None)


def has_trips(pipeline):
    """Whether the pipeline has a Trips node, so its trips can be downloaded."""
    return _trips_step(pipeline) is not None


def trip_rules(pipeline):
    """The pipeline's Trips node as ``{min_seconds, max_seconds}`` (None = no
    limit). Read from the pipeline as it is now: trips run across clips, so
    they aren't part of what a clip's run froze."""
    def number(config, key, scale=1):
        try:
            return float(config.get(key)) * scale
        except (TypeError, ValueError):
            return None

    config = (_trips_step(pipeline) or {}).get("config") or {}
    return {"min_seconds": number(config, "min_trip_seconds"),
            "max_seconds": number(config, "max_trip_minutes", 60)}


def suggest_trips(events, rules=None):
    """Each exit → the entry that ends its trip, one to one, within a day.

    ``events`` are ``(clip_index, event)`` pairs from every clip. An exit pairs
    with the next exit-or-entry at its tube, when that is an entry; another exit
    first means the entry between them was missed, and no trip is claimed. When
    the exit carries a marker, the next unused entry by that marker wins. No
    entry ends two trips, and a pair outside ``rules`` (``trip_rules``) is none.
    """
    rules = rules or {}
    tubes = {}
    for ci, e in events:
        if e["action"] in ("exit", "enter"):
            tubes.setdefault((e["reference"], (e.get("clock") or "")[:10]), []).append((ci, e))
    trips = []
    for lst in tubes.values():
        lst.sort(key=lambda p: p[1]["t"])
        used = set()
        for i, (ci, e) in enumerate(lst):
            if e["action"] != "exit":
                continue
            later = range(i + 1, len(lst))
            at = None
            if e.get("marker"):
                at = next((j for j in later if lst[j][1]["action"] == "enter" and j not in used
                           and lst[j][1].get("marker") == e["marker"]), None)
            if at is None and i + 1 < len(lst) and lst[i + 1][1]["action"] == "enter" \
                    and i + 1 not in used:
                at = i + 1
            if at is None:
                continue
            pick = lst[at]
            gap = pick[1]["t"] - e["t"]
            if gap <= 0 or (rules.get("min_seconds") is not None and gap < rules["min_seconds"]) \
                    or (rules.get("max_seconds") is not None and gap > rules["max_seconds"]):
                continue
            used.add(at)
            trips.append({"exit": (ci, e), "entry": pick,
                          "same_marker": bool(e.get("marker")) and pick[1].get("marker") == e["marker"],
                          "gap_sec": round(gap, 1)})
    trips.sort(key=lambda t: t["exit"][1]["t"])
    return trips


# ── Trip types (memory/78) ───────────────────────────────────────────────────

def hops_of(run):
    """``{event key: {track, from, to}}``: the exits and entries of a run that
    belong to a tube hop, an exit whose track's next exit or entry is an entry.
    One track is the only evidence used, so a hop split across two tracks or
    two clips isn't seen. Reads the slim copy as well as the whole context."""
    from . import primitives
    from .aggregate import analyzer_outputs
    from .review import event_key

    by_track = {}
    for raw in (r for k, o in analyzer_outputs(run) if k == "events" for r in (o.get("rows") or [])):
        e = primitives.normalize_event(raw)
        try:
            frame = int(e["frame"])
        except (TypeError, ValueError, KeyError):
            continue
        if e.get("action") in ("exit", "enter") and e.get("track_id") not in (None, ""):
            by_track.setdefault(str(e["track_id"]), []).append((frame, e))
    hops = {}
    for track, rows in by_track.items():
        rows.sort(key=lambda r: r[0])
        for (_f, a), (_g, b) in zip(rows, rows[1:]):
            if a["action"] == "exit" and b["action"] == "enter":
                hops[event_key(a)] = hops[event_key(b)] = {
                    "track": track, "from": str(a.get("reference") or ""),
                    "to": str(b.get("reference") or "")}
    return hops


def auto_trip_type(exit_key, entry_key, exit_hops, entry_hops):
    """A trip's type from the tracks: a Tube hop when either end is one."""
    from .models import TripReview

    hop = exit_key in (exit_hops or {}) or entry_key in (entry_hops or {})
    return TripReview.Type.TUBE_HOP.value if hop else TripReview.Type.DIRECT.value


def hop_text(exit_key, entry_key, exit_hops, entry_hops):
    """Why a trip is a Tube hop, in words, or ""."""
    def tube(ref):
        return ref.replace("reference_", "")

    hop = (exit_hops or {}).get(exit_key) or (entry_hops or {}).get(entry_key)
    if not hop:
        return ""
    if hop["from"] == hop["to"]:
        return f"Track {hop['track']} left tube {tube(hop['from'])} and went back in without leaving view."
    return f"Track {hop['track']} left tube {tube(hop['from'])} and went into tube {tube(hop['to'])}."


# ── Scoring the pipeline's trips (memory/79) ─────────────────────────────────

def _scored_events(run, video, start, votes, added):
    """A clip's exits and entries for ``suggest_trips``, from the slim copy, as
    ``(detected, confirmed)``: all it detected, then those without the rejected
    and with the added. No markers: the slim copy has none."""
    from . import primitives
    from .aggregate import analyzer_outputs
    from .ops import fps_with_source
    from .review import combined, event_key

    stats = next((v["result"].get("summary_stats") or {} for v in (run.context or {}).values()
                  if isinstance(v, dict) and isinstance(v.get("result"), dict)), {})
    fps = float(fps_with_source(stats, video)[0] or 25)

    def event(key, action, reference, frame):
        at = start + timedelta(seconds=frame / fps)
        return (run.pk, {"key": key, "action": action, "reference": str(reference or ""),
                         "t": (at - datetime(2000, 1, 1)).total_seconds(),
                         "clock": at.isoformat(), "marker": ""})

    detected, confirmed = [], []
    for raw in (r for k, o in analyzer_outputs(run) if k == "events" for r in (o.get("rows") or [])):
        e = primitives.normalize_event(raw)
        try:
            frame = int(e["frame"])
        except (TypeError, ValueError, KeyError):
            continue
        if e.get("action") not in ("exit", "enter"):
            continue
        key = event_key(e)
        row = event(key, e["action"], e.get("reference"), frame)
        detected.append(row)
        if combined(votes.get(key, [])) != "rejected":
            confirmed.append(row)
    for a in added:
        if a.action in ("exit", "enter"):
            confirmed.append(event(event_key({"action": a.action, "reference": a.reference,
                                              "track_id": a.track_id, "frame": a.frame}),
                                   a.action, a.reference, a.frame))
    return detected, confirmed


def _trip_places(runs):
    """The clips of ``runs`` by place (device, else site): a list of lists of
    ``(run, video, local start)``. Pairing runs within a place."""
    from apps.videos.models import Video

    from .aggregate import run_video_id

    by_video = {}
    for run in runs:
        if run_video_id(run):
            by_video.setdefault(run_video_id(run), run)
    places = {}
    for v in Video.objects.filter(pk__in=list(by_video)).select_related("device", "site"):
        place = ("device", v.device_id) if v.device_id else ("site", v.site_id) if v.site_id else None
        start = local_start(v)
        if place and start is not None:
            places.setdefault(place, []).append((by_video[v.pk], v, start))
    return list(places.values())


def _trip_sets(places):
    """Per place ``(pipeline, listed, kept)``: the trips of the detected events
    and of the reviewed events, each ``{(exit run, exit key, entry run, entry
    key): trip}``, and the ``(run, key)`` of the events that survived review."""
    from .models import AddedEvent
    from .review import review_votes, shared_job_runs

    chosen = {c[0].pk: c[0] for clips in places for c in clips}
    if not chosen:
        return
    votes = review_votes(chosen.values())
    owners = shared_job_runs(chosen.values(), "added_events")
    added = {}
    for a in AddedEvent.objects.filter(run_id__in=list(owners)):
        added.setdefault(owners[a.run_id].pk, []).append(a)
    rules = trip_rules(next(iter(chosen.values())).pipeline)

    def paired(events):
        return {(t["exit"][0], t["exit"][1]["key"], t["entry"][0], t["entry"][1]["key"]): t
                for t in suggest_trips(events, rules)}

    for clips in places:
        detected, confirmed = [], []
        for run, video, start in clips:
            d, c = _scored_events(run, video, start, votes.get(run.pk, {}), added.get(run.pk, []))
            detected += d
            confirmed += c
        yield paired(detected), paired(confirmed), {(pk, e["key"]) for pk, e in confirmed}


def _trip_call(mine):
    """The trip's call from its ``TripReview``s: accepted, rejected, disputed,
    or blank for none or "can't tell"."""
    from .review import combined

    if any(c.reason == "can't tell" for c in mine):
        return ""
    return combined([c.verdict for c in mine])


def _trip_kind(k, mine, hops_at):
    return next((c.trip_type for c in mine if c.trip_type), "") or auto_trip_type(
        k[1], k[3], hops_at(k[0]), hops_at(k[2]))


def _hops_cache(places):
    chosen = {c[0].pk: c[0] for clips in places for c in clips}
    hops = {}

    def hops_at(pk):
        if pk not in hops:
            hops[pk] = hops_of(chosen[pk])
        return hops[pk]
    return hops_at


def trip_scores(runs, reviewed, calls):
    """The pipeline's trips against the trips reviewers confirmed.

    ``runs`` read slim; ``reviewed``: the pks of the clips marked reviewed;
    ``calls``: ``{(run pk, exit key, entry key): [TripReview]}``. Pairing runs
    over every clip of a unit or site, then only trips with both ends in
    reviewed clips count. The true trips are those of the reviewed events that
    someone answered Yes. A pipeline trip is right when it is one of them and
    wrong when it isn't, by a No or because its events didn't survive review; a
    true trip the pipeline didn't make is missed. A trip of the reviewed events
    with no answer yet waits, and counts for nothing.

    ``{clips, right, wrong, missed, waiting, types: {type: the same four}}``.
    """
    from .models import TripReview

    def tally():
        return {"right": 0, "wrong": 0, "missed": 0, "waiting": 0}

    out = {"clips": len(reviewed), **tally(),
           "types": {value: tally() for value in TripReview.Type.values}}
    if not reviewed:
        return out
    places = [clips for clips in _trip_places(runs) if any(c[0].pk in reviewed for c in clips)]
    hops_at = _hops_cache(places)

    def in_reviewed(trips):
        return {k for k in trips if k[0] in reviewed and k[2] in reviewed}

    def count(field, keys):
        for k in keys:
            kind = _trip_kind(k, calls.get((k[0], k[1], k[3]), []), hops_at)
            out[field] += 1
            out["types"][kind][field] += 1

    def answer(k):
        call = _trip_call(calls.get((k[0], k[1], k[3]), []))
        return "" if call == "disputed" else call

    for pipeline, listed, _kept in _trip_sets(places):
        pipeline, listed = in_reviewed(pipeline), in_reviewed(listed)
        true = {k for k in listed if answer(k) == "accepted"}
        waiting = {k for k in listed if not answer(k)}
        count("right", pipeline & true)
        count("wrong", pipeline - true - waiting)
        count("missed", true - pipeline)
        count("waiting", waiting)
    return out


# ── Trip votes ───────────────────────────────────────────────────────────────

def _same_trip(exit_run, exit_key, entry_key):
    from .models import TripReview
    from .review import shared_job_runs

    return TripReview.objects.filter(
        exit_run_id__in=list(shared_job_runs([exit_run], "trip_exits")), exit_key=exit_key,
        entry_key=entry_key or "")


def _check_type(trip_type):
    from .models import TripReview

    if trip_type and trip_type not in TripReview.Type.values:
        raise ValueError("unknown trip type")


def set_trip_type(exit_run, exit_key, entry_key, trip_type):
    """A reviewer's type on a trip already called; blank goes back to the
    automatic one. Raises ValueError on a bad type."""
    _check_type(trip_type)
    _same_trip(exit_run, exit_key, entry_key).update(trip_type=trip_type or "")


def set_trip_vote(user, exit_run, exit_key, entry_run, entry_key, verdict, reason="", relinked=False,
                  before_review=True, trip_type=None):
    """Make, change or (verdict None) take back the call on this exit and
    entry, whoever made it before (memory/74). ``trip_type`` None keeps the
    type a reviewer gave it. Raises ValueError on a bad request."""
    from .models import TripReview
    from .review import TRIP_REASONS

    same = _same_trip(exit_run, exit_key, entry_key)
    _check_type(trip_type)
    if trip_type is None:
        trip_type = same.exclude(trip_type="").values_list("trip_type", flat=True).first() or ""
    if verdict in (None, "", "none"):
        same.delete()
        return None
    if verdict not in TripReview.Verdict.values:
        raise ValueError("verdict must be accepted or rejected")
    # The reviewer's own words; the two the metrics know are matched by case.
    reason = (reason or "").strip()[:300]
    if reason.lower() in TRIP_REASONS:
        reason = reason.lower()
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
    same.exclude(exit_run=exit_run, user=user).delete()
    TripReview.objects.update_or_create(
        exit_run=exit_run, exit_key=exit_key, entry_key=entry_key or "", user=user,
        defaults={"entry_run": entry_run if entry_key else None,
                  "before_review": bool(before_review), "trip_type": trip_type or "",
                  "reference": ref, "exit_track": exit_track, "exit_frame": exit_frame,
                  "entry_track": entry_track or "", "entry_frame": entry_frame,
                  "verdict": verdict, "reason": reason, "relinked": bool(relinked)})
    return verdict


TRIP_FIELDS = ["reference", "exit_time", "entry_time", "duration_sec", "exit_clip", "exit_track",
               "entry_clip", "entry_track", "species", "marker_id", "source", "review",
               "reviewed_by", "review_notes", "trip_type", "trip_type_set_by", "clip_reviewed"]


def trip_rows(runs, reviewed, calls):
    """The Trips CSV: every trip of the detected events (``source`` pipeline)
    and of the reviewed events (``review``), scored as ``trip_scores`` scores
    them, so the file gives its numbers.

    ``review`` is the call on the trip, or for a pipeline trip the reviewed
    events don't make, why: ``event_rejected``, or ``pairing_changed`` when both
    its events stand but a rejected or added one changed what pairs with what.
    ``clip_reviewed``: both ends in clips marked reviewed. Arguments as
    ``trip_scores``; each clip is read whole in turn for its species.
    """
    from apps.analysis.models import JobResult

    from . import tracks
    from .aggregate import _provenance, run_job_id, whole_runs

    places = _trip_places(runs)
    clips = {c[0].pk: c for place in places for c in place}
    hops_at = _hops_cache(places)
    where = {pk: _provenance({"video": c[1], "title": c[1].title or f"Video {c[1].pk}"})
             for pk, c in clips.items()}

    def stamp(event):
        return datetime.fromisoformat(event["clock"]).isoformat(timespec="milliseconds")

    def track(key):
        return (str(key).split("|") + ["", "", ""])[2]

    rows, identify = [], {}
    for pipeline, listed, kept in _trip_sets(places):
        for k in {**pipeline, **listed}:
            t = listed.get(k) or pipeline[k]
            e, n = t["exit"][1], t["entry"][1]
            called = calls.get((k[0], k[1], k[3]), [])
            mine = called if k in listed else []
            if k in listed:
                review = _trip_call(mine)
            elif (k[0], k[1]) in kept and (k[2], k[3]) in kept:
                review = "pairing_changed"
            else:
                review = "event_rejected"
            notes = []
            for c in mine:
                if c.reason and c.reason not in notes:
                    notes.append(c.reason)
            chosen = next((c.trip_type for c in called if c.trip_type), "")
            row = {**where[k[0]], "reference": e["reference"], "exit_time": stamp(e),
                   "entry_time": stamp(n), "duration_sec": round(n["t"] - e["t"], 2),
                   "exit_clip": where[k[0]]["video_title"], "exit_track": track(k[1]),
                   "entry_clip": where[k[2]]["video_title"], "entry_track": track(k[3]),
                   "species": "", "marker_id": "",
                   "source": "pipeline" if k in pipeline else "review", "review": review,
                   "reviewed_by": ";".join(sorted(c.user.username for c in mine)),
                   "review_notes": "; ".join(notes),
                   "trip_type": _trip_kind(k, called, hops_at),
                   "trip_type_set_by": "reviewer" if chosen else "auto",
                   "clip_reviewed": "true" if k[0] in reviewed and k[2] in reviewed else "false"}
            rows.append(row)
            identify.setdefault(k[0], []).append(row)
    for run in whole_runs([clips[pk][0] for pk in identify]):
        # The run's own (re-tracked) stats first (memory/69).
        stats = _run_stats(run) or JobResult.objects.filter(job_id=run_job_id(run)).values_list(
            "summary_stats", flat=True).first() or {}
        by_track = tracks.by_track_of({"summary_stats": stats})
        for row in identify[run.pk]:
            ident = tracks.identity(by_track.get(row["exit_track"]))
            row.update(species=ident.get("species", ""), marker_id=ident.get("marker_id", ""))
    rows.sort(key=lambda r: (r["exit_time"], r["reference"]))
    return rows


def _run_stats(run):
    """A re-tracked run's own summary stats, else None (memory/69)."""
    from .retrack.run import tracking_outputs

    for sid in tracking_outputs(run):
        result = run.context[sid].get("result") or {}
        if result.get("retracked"):
            return result.get("summary_stats") or {}
    return None
