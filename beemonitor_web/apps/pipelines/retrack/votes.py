"""Review votes follow their events after a recount or a re-track (memory/69).

An event's key holds its track id, so after a re-track a reviewed event can
come back under a new key. Each vote copied its event's action, reference and
frame, so it is moved to the new event of the same action at the same
reference within half a second — when exactly one matches. The rest stay as
they are: shown as no longer detected, as after any recount.

A vote saved on this run is moved in place. One saved on another run of the
same GPU job (an older batch, which keeps its own tracks) is copied onto this
run, so that batch keeps counting it under its own events.
"""

from __future__ import annotations


def _events(run):
    from ..aggregate import analyzer_outputs
    from ..review import event_key

    out = {}
    for kind, o in analyzer_outputs(run):
        if kind != "events":
            continue
        for r in o.get("rows") or []:
            key = event_key(r)
            out[key] = r
    return out


def _filtered_keys(run):
    from ..aggregate import analyzer_outputs

    return {k for kind, o in analyzer_outputs(run) if kind == "events"
            for k in (o.get("filtered_out") or {}).get("event_keys") or []}


def _matcher(events, fps, same_tracks):
    """``match(action, reference, frame, track)`` -> ``(frame, key, track)``.

    When the tracks are the ones the votes were cast on (a recount, not a
    re-track), the event of the same track, action and reference is the same
    event even if its frame moved (e.g. counted from the whole trajectory).
    Otherwise — or when that finds nothing — the one event of that action and
    reference within half a second."""
    from ..primitives import normalize_event

    tolerance = max(1, round(float(fps or 25) * 0.5))
    index = {}
    for key, row in events.items():
        r = normalize_event(row)
        try:
            frame = int(float(r.get("frame")))
        except (TypeError, ValueError):
            continue
        index.setdefault((r.get("action"), str(r.get("reference") or "")), []).append(
            (frame, key, str(r.get("track_id") or "")))

    def match(action, reference, frame, track=None):
        hits = index.get((action, str(reference or "")), [])
        if same_tracks and track not in (None, ""):
            own = [h for h in hits if h[2] == str(track)]
            if own:
                return min(own, key=lambda h: abs(h[0] - (frame or 0)))
        if frame is None:
            return None
        near = [h for h in hits if abs(h[0] - frame) <= tolerance]
        return near[0] if len(near) == 1 else None

    return match


def move_votes(run, fps, same_tracks=False) -> dict:
    """Move or copy this run's event and trip votes onto its new events.
    ``same_tracks``: the run's tracks are the ones the votes were cast on
    (after a recount rather than a re-track). ``{moved, gone}``."""
    from ..models import EventReview, TripReview
    from ..review import shared_job_runs

    events = _events(run)
    if not events:
        return {"moved": 0, "gone": 0}
    # Left out by an insect-group filter, not gone: their votes wait where they
    # are, for the filter coming off (memory/65).
    known = set(events) | _filtered_keys(run)
    match = _matcher(events, fps, same_tracks)
    moved = gone = 0

    owners = shared_job_runs([run], "event_reviews")
    for v in EventReview.objects.filter(run_id__in=list(owners)):
        if owners[v.run_id].pk != run.pk or v.event_key in known:
            continue
        hit = match(v.action, v.reference, v.frame, v.track_id)
        if hit is None:
            gone += 1
            continue
        frame, key, track = hit
        taken = EventReview.objects.filter(run=run, event_key=key, user_id=v.user_id)
        if v.run_id == run.pk:
            if taken.exclude(pk=v.pk).exists():
                continue
            EventReview.objects.filter(pk=v.pk).update(event_key=key, track_id=track, frame=frame)
        elif not taken.exists():
            EventReview.objects.create(run=run, event_key=key, action=v.action,
                                       reference=v.reference, track_id=track, frame=frame,
                                       user_id=v.user_id, verdict=v.verdict, reason=v.reason)
        moved += 1

    owners = shared_job_runs([run], "trip_exits")
    for t in TripReview.objects.filter(exit_run_id__in=list(owners)):
        if owners[t.exit_run_id].pk != run.pk:
            continue
        new_exit = (None if t.exit_key in known
                    else match("exit", t.reference, t.exit_frame, t.exit_track))
        same_job_entry = t.entry_run_id is not None and owners.get(t.entry_run_id) is run
        new_entry = None
        if same_job_entry and t.entry_key and t.entry_key not in known:
            new_entry = match("enter", t.reference, t.entry_frame, t.entry_track)
        if t.exit_key not in known and new_exit is None:
            gone += 1
            continue
        if new_exit is None and new_entry is None:
            continue
        fields = {}
        if new_exit:
            fields.update(exit_frame=new_exit[0], exit_key=new_exit[1], exit_track=new_exit[2])
        if new_entry:
            fields.update(entry_frame=new_entry[0], entry_key=new_entry[1],
                          entry_track=new_entry[2], entry_run=run)
        exit_key = fields.get("exit_key", t.exit_key)
        taken = TripReview.objects.filter(exit_run=run, exit_key=exit_key, user_id=t.user_id)
        if t.exit_run_id == run.pk:
            if taken.exclude(pk=t.pk).exists():
                continue
            TripReview.objects.filter(pk=t.pk).update(**fields)
        elif not taken.exists():
            TripReview.objects.create(
                exit_run=run, exit_key=exit_key,
                entry_run=fields.get("entry_run", t.entry_run),
                entry_key=fields.get("entry_key", t.entry_key), reference=t.reference,
                exit_track=fields.get("exit_track", t.exit_track),
                exit_frame=fields.get("exit_frame", t.exit_frame),
                entry_track=fields.get("entry_track", t.entry_track),
                entry_frame=fields.get("entry_frame", t.entry_frame),
                user_id=t.user_id, verdict=t.verdict, reason=t.reason, relinked=t.relinked)
        moved += 1
    return {"moved": moved, "gone": gone}
