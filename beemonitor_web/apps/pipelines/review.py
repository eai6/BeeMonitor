"""Reviewers' calls on detected events and suggested trips (memory/62).

A review never changes what was detected: it sits beside it. Events are
identified by themselves (``event_key``), so a recount can't attach a vote to
the wrong row; a vote whose event a recount removed is kept and reported as no
longer detected.

Who may review a run: whoever launched it, and the people they invited to its
batch as reviewers (``BatchMember``). Viewers and the public link watch only.
"""

from __future__ import annotations

from . import primitives

REASONS = ("not an event", "wrong tube", "wrong action", "can't tell")
TRIP_REASONS = ("relinked", "didn't come back", "can't tell")


def event_key(row) -> str:
    """``action|reference|track_id|frame`` for a saved (or normalised) event row."""
    row = primitives.normalize_event(row)
    return "|".join(str(row.get(k) if row.get(k) is not None else "")
                    for k in ("action", "reference", "track_id", "frame"))


# ── Access ───────────────────────────────────────────────────────────────────

def member_role(user, run):
    """The user's BatchMember role on the run's batch, or None."""
    from .models import BatchMember

    if not getattr(user, "is_authenticated", False) or not run.batch_id:
        return None
    return (BatchMember.objects.filter(batch_id=run.batch_id, user=user)
            .values_list("role", flat=True).first())


def can_view(user, run) -> bool:
    if not getattr(user, "is_authenticated", False):
        return False
    return run.user_id == user.pk or member_role(user, run) is not None


def can_review(user, run) -> bool:
    if not getattr(user, "is_authenticated", False):
        return False
    from .models import BatchMember

    return run.user_id == user.pk or member_role(user, run) == BatchMember.Role.REVIEWER


# ── Event votes ──────────────────────────────────────────────────────────────

def set_event_vote(run, user, key, verdict, reason=""):
    """Store, change or (``verdict`` None) take back one reviewer's call.
    Returns the stored verdict or None. Raises ValueError on a bad request."""
    from .models import EventReview

    if verdict in (None, "", "none"):
        EventReview.objects.filter(run=run, event_key=key, user=user).delete()
        return None
    if verdict not in EventReview.Verdict.values:
        raise ValueError("verdict must be accepted or rejected")
    reason = (reason or "").strip().lower()
    if reason and reason not in REASONS:
        raise ValueError("unknown reason")
    action, reference, track_id, frame = (key.split("|") + ["", "", "", ""])[:4]
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        frame = None
    EventReview.objects.update_or_create(
        run=run, event_key=key, user=user,
        defaults={"verdict": verdict, "reason": reason if verdict == "rejected" else "",
                  "action": action, "reference": reference, "track_id": track_id,
                  "frame": frame})
    return verdict


def combined(verdicts):
    """accepted / rejected / disputed (reviewers disagree) / "" (none)."""
    kinds = set(verdicts)
    if not kinds:
        return ""
    return kinds.pop() if len(kinds) == 1 else "disputed"


def event_summaries(run):
    """``{event_key: {review, reviewed_by, review_notes, accepted, rejected,
    votes: {username: verdict}}}`` for every reviewed event of the run."""
    from .models import EventReview

    out = {}
    for r in (EventReview.objects.filter(run=run).select_related("user")
              .order_by("user__username")):
        s = out.setdefault(r.event_key, {"votes": {}, "notes": []})
        s["votes"][r.user.username] = r.verdict
        if r.reason and r.reason not in s["notes"]:
            s["notes"].append(r.reason)
    for s in out.values():
        verdicts = list(s["votes"].values())
        s.update(review=combined(verdicts),
                 reviewed_by=";".join(s["votes"]),
                 review_notes="; ".join(s.pop("notes")),
                 accepted=verdicts.count("accepted"),
                 rejected=verdicts.count("rejected"))
    return out


def user_votes(run, user):
    """``{event_key: verdict}`` for this reviewer."""
    from .models import EventReview

    if not getattr(user, "is_authenticated", False):
        return {}
    return dict(EventReview.objects.filter(run=run, user=user)
                .values_list("event_key", "verdict"))


def event_review_counts(run, rows=None):
    """How many of the run's current events are reviewed: ``{events,
    reviewed, accepted, rejected, disputed, gone}`` — ``gone`` counts votes on
    events a recount removed."""
    if rows is None:
        from .aggregate import analyzer_outputs

        rows = [r for k, out in analyzer_outputs(run) if k == "events"
                for r in (out.get("rows") or [])]
    keys = {event_key(r) for r in rows}
    summaries = event_summaries(run)
    current = {k: s for k, s in summaries.items() if k in keys}
    calls = [s["review"] for s in current.values()]
    from .models import AddedEvent

    added = AddedEvent.objects.filter(run=run).count()
    precision, recall = quality(calls.count("accepted"), calls.count("rejected"), added)
    return {"events": len(keys), "reviewed": len(current),
            "accepted": calls.count("accepted"), "rejected": calls.count("rejected"),
            "disputed": calls.count("disputed"), "added": added,
            "precision": precision, "recall": recall,
            "gone": len(summaries) - len(current)}


def review_columns(summaries, key):
    """The Events CSV's review columns for one event."""
    s = summaries.get(key) or {}
    return {"review": s.get("review", ""), "reviewed_by": s.get("reviewed_by", ""),
            "review_notes": s.get("review_notes", "")}


def run_for_job(job):
    from .executors import _run_for_job

    return _run_for_job(job)


def run_for_review(job, run_pk=None):
    """The run a vote on this job is for: ``run_pk`` when the page named the
    run it shows (a clip's GPU job can be shared by several runs), else the
    job's newest run."""
    from .executors import pin_run

    return pin_run(job, run_pk) or run_for_job(job)


def summaries_for_job(job):
    run = run_for_job(job)
    return event_summaries(run) if run is not None else {}


def added_for_job(job):
    run = run_for_job(job)
    return added_rows(run) if run is not None else []


def shared_job_runs(runs, reviewed_by):
    """``{run pk: the run of ``runs`` it counts for}``: ``runs`` themselves and
    every other run with reviews (``reviewed_by``: "event_reviews",
    "trip_exits" or "added_events") that shares one of their GPU jobs.

    A re-run re-uses its clips' GPU jobs, and a period page opened without its
    batch saves votes to the clip's newest run: the same tracks and the same
    event keys, so those votes are this run's too."""
    from .aggregate import run_job_id
    from .models import PipelineRun

    runs = list(runs)
    out = {r.pk: r for r in runs}
    by_job = {j: r for r in runs if (j := run_job_id(r)) is not None}
    for other in (PipelineRun.objects.filter(**{f"{reviewed_by}__isnull": False})
                  .exclude(pk__in=list(out)).distinct().only("pk", "context")):
        own = by_job.get(run_job_id(other))
        if own is not None:
            out[other.pk] = own
    return out


def _event_keys(run):
    from .aggregate import analyzer_outputs

    return {event_key(r) for k, o in analyzer_outputs(run) if k == "events"
            for r in (o.get("rows") or [])}


def trip_reviews_for(runs):
    """Trip calls on exits in these runs, wherever they were saved
    (``shared_job_runs``; from another run only if this run has that exit): the
    latest per reviewer and exit. Each carries ``own_run``, the run of ``runs`` it counts for."""
    from .models import TripReview

    owners = shared_job_runs(runs, "trip_exits")
    keys, latest = {}, {}
    for t in TripReview.objects.filter(exit_run_id__in=list(owners)):
        own = owners[t.exit_run_id]
        if t.exit_run_id != own.pk:         # saved on another run: if this one has it
            if own.pk not in keys:
                keys[own.pk] = _event_keys(own)
            if t.exit_key not in keys[own.pk]:
                continue
        k = (own.pk, t.exit_key, t.user_id)
        if k not in latest or t.updated_at > latest[k].updated_at:
            t.own_run = own
            latest[k] = t
    return list(latest.values())


def review_votes(runs):
    """``{run pk: {event_key: [verdict per reviewer]}}`` for these runs.

    Votes saved on another run of the same GPU job count too
    (shared_job_runs); each reviewer's latest call on an event wins."""
    from .models import EventReview

    runs = list(runs)
    owners = shared_job_runs(runs, "event_reviews")
    latest = {}
    for run_id, key, user_id, verdict, when in (
            EventReview.objects.filter(run_id__in=list(owners))
            .values_list("run_id", "event_key", "user_id", "verdict", "updated_at")):
        k = (owners[run_id].pk, key, user_id)
        if k not in latest or when > latest[k][1]:
            latest[k] = (verdict, when)
    votes = {}
    for (run_pk, key, _u), (verdict, _w) in latest.items():
        votes.setdefault(run_pk, {}).setdefault(key, []).append(verdict)
    return votes


def batch_review_counts(runs, actions=None):
    """``{run pk: counts}`` (as ``event_review_counts``) for many runs in two
    queries' worth of work; None for a run with no Events step.

    ``actions`` limits the counts to those event actions (memory/70: e.g.
    exits and entries for foraging); None counts all. ``scored`` lists each
    reviewed event's ``[action, ml_confidence, call]`` where it has a score,
    for the cut-off panel."""
    from .aggregate import analyzer_outputs
    from .models import EventReview

    from django.db.models import Count

    from .models import AddedEvent

    runs = list(runs)
    votes = review_votes(runs)
    added_owners = shared_job_runs(runs, "added_events")
    added = {}
    added_qs = AddedEvent.objects.filter(run_id__in=list(added_owners))
    if actions is not None:
        added_qs = added_qs.filter(action__in=list(actions))
    for run_id, n in (added_qs.values("run_id").annotate(n=Count("id"))
                      .values_list("run_id", "n")):
        own = added_owners[run_id].pk
        added[own] = added.get(own, 0) + n
    # Clips a reviewer marked done (ClipReview): counted as reviewed even
    # with nothing to vote on.
    from .models import ClipReview
    done_owners = shared_job_runs(runs, "clip_reviews")
    done = {done_owners[r].pk for r in ClipReview.objects.filter(
        run_id__in=list(done_owners)).values_list("run_id", flat=True)}
    out = {}
    for run in runs:
        outs = [o for k, o in analyzer_outputs(run) if k == "events"]
        if not outs:
            out[run.pk] = None
            continue
        all_keys = {event_key(r) for o in outs for r in (o.get("rows") or [])}
        rows = {event_key(r): r for o in outs for r in (o.get("rows") or [])
                if actions is None or primitives.normalize_event(r).get("action") in actions}
        keys = set(rows)
        mine = votes.get(run.pk, {})
        # Votes on events whose action the metrics leave out (e.g. stays).
        other_votes = sum(1 for k in mine if k in all_keys and k not in keys)
        calls = [combined(v) for k, v in mine.items() if k in keys]
        scored = [[primitives.normalize_event(rows[k]).get("action"),
                   float(rows[k]["ml_confidence"]), combined(v)]
                  for k, v in mine.items()
                  if k in keys and rows[k].get("ml_confidence") not in (None, "")]
        # Reviewed events on tracks an insect-group filter now leaves out
        # (memory/65): how the reviewers called what the filter took away.
        gone = {k for o in outs for k in ((o.get("filtered_out") or {}).get("event_keys") or [])}
        left = [combined(v) for k, v in mine.items() if k in gone]
        out[run.pk] = {"events": len(keys), "reviewed": len(calls),
                       "accepted": calls.count("accepted"), "rejected": calls.count("rejected"),
                       "disputed": calls.count("disputed"), "added": added.get(run.pk, 0),
                       "scored": scored,
                       "clip_reviewed": run.pk in done, "other_votes": other_votes,
                       "left_out": {"reviewed": len(left), "accepted": left.count("accepted"),
                                    "rejected": left.count("rejected")}}
    return out


def batch_group_summary(runs, per_run=None):
    """What the batch's insect-group filters left out (memory/65): tracks and
    events by group, and how reviewers had called the events it took away.
    None when no run in the batch leaves anything out."""
    from .aggregate import analyzer_outputs
    from .regroup import group_step

    from .recount import CURRENT_RULES

    groups, tracks, events, filtered_runs = {}, 0, 0, 0
    # Each clip's own setting: after a change, clips counted before and after
    # it are reported apart instead of merged (memory/65).
    settings, options, pending = {}, {}, 0
    for run in runs:
        _i, step = group_step(run)
        if step is None:
            continue
        if run.counting_rules < CURRENT_RULES:
            pending += 1
        out = (run.context or {}).get(step["id"]) or {}
        in_line = any((s.get("inputs") or {}).get("tracks") == step["id"] for s in run.steps or [])
        if not (out.get("group_filter") and in_line):
            continue
        filtered_runs += 1
        setting = tuple(sorted(out.get("leave_out") or []))
        settings[setting] = settings.get(setting, 0) + 1
        options.setdefault(setting, out.get("group_options") or [])
        for g in out.get("group_counts") or []:
            if not g.get("counted"):
                row = groups.setdefault(g["group"], {"group": g["group"], "tracks": 0, "events": 0})
                row["tracks"] += g["tracks"]
                tracks += g["tracks"]
        for kind, o in analyzer_outputs(run):
            if kind != "events":
                continue
            for g in (o.get("filtered_out") or {}).get("by_group") or []:
                row = groups.setdefault(g["group"], {"group": g["group"], "tracks": 0, "events": 0})
                row["events"] += g["events"]
                events += g["events"]
    if not filtered_runs:
        return None
    left = {"reviewed": 0, "accepted": 0, "rejected": 0}
    for counts in (per_run or {}).values():
        for k in left:
            left[k] += ((counts or {}).get("left_out") or {}).get(k, 0)
    main = max(settings, key=settings.get)
    return {"tracks": tracks, "events": events,
            "left_out": list(main),
            "kept": [g for g in options[main] if g not in main],
            "clips": filtered_runs,
            # Clips counted with a different filter from most of the batch.
            "other_clips": filtered_runs - settings[main],
            # Clips queued for a recount after an Apply.
            "pending": pending,
            "by_group": sorted(groups.values(), key=lambda g: -g["tracks"]),
            "reviewed": left}


def batch_review_summary(runs, per_run=None, actions=None):
    """The batch page's review metrics (memory/62): events (precision and
    recall from reviewers' calls and added events) and trips. None when
    nothing in the batch has been reviewed."""
    runs = list(runs)
    per_run = per_run if per_run is not None else batch_review_counts(runs, actions)
    ev = {"accepted": 0, "rejected": 0, "disputed": 0, "added": 0, "reviewed": 0,
          "events": 0, "clips": 0, "scored": []}
    for counts in per_run.values():
        if not counts:
            continue
        ev["events"] += counts["events"]
        for k in ("accepted", "rejected", "disputed", "added", "reviewed"):
            ev[k] += counts[k]
        ev["scored"] += counts.get("scored") or []
        if counts["reviewed"] or counts["added"] or counts.get("clip_reviewed"):
            ev["clips"] += 1
    ev["precision"], ev["recall"] = quality(ev["accepted"], ev["rejected"], ev["added"])
    # The actions counted (memory/70), and the scored reviewed events for the
    # cut-off panel: [action, ml_confidence, call].
    ev["actions"] = list(actions) if actions is not None else None

    by_exit = {}
    for r in trip_reviews_for(runs):
        by_exit.setdefault((r.own_run.pk, r.exit_key), []).append(r)
    trips = {"reviewed": len(by_exit), "accepted": 0, "relinked": 0, "rejected": 0,
             "didnt_come_back": 0, "cant_tell": 0, "disputed": 0}
    for calls in by_exit.values():
        call = combined([c.verdict for c in calls])
        if call == "disputed":
            trips["disputed"] += 1
        elif call == "accepted":
            trips["accepted"] += 1
            if any(c.relinked for c in calls):
                trips["relinked"] += 1
        else:
            trips["rejected"] += 1
            reasons = {c.reason for c in calls}
            if "didn't come back" in reasons:
                trips["didnt_come_back"] += 1
            if "can't tell" in reasons:
                trips["cant_tell"] += 1
    # How often the suggested pairing was right as it stood.
    as_suggested = trips["accepted"] - trips["relinked"]
    trips["suggestion_right"] = (round(100 * as_suggested / trips["reviewed"])
                                 if trips["reviewed"] else None)
    if not (ev["reviewed"] or ev["added"] or ev["clips"] or trips["reviewed"]) and actions is None:
        return None
    return {"events": ev, "trips": trips}


def member_job_ids(user):
    """GPU job pks of every run in a batch the user was invited to
    (``BatchMember``): their clips' results, overlay, crops and CSVs open for
    them, and nothing else does."""
    from .aggregate import run_job_id
    from .models import BatchMember, PipelineRun

    if not getattr(user, "is_authenticated", False):
        return []
    batches = BatchMember.objects.filter(user=user).values_list("batch_id", flat=True)
    return [j for run in PipelineRun.objects.filter(batch_id__in=batches).only("context")
            if (j := run_job_id(run)) is not None]


# ── Missed events (memory/62 §3b) ────────────────────────────────────────────

ACTIONS = ("exit", "enter", "stay")
DUPLICATE_SECONDS = 2.0


class Duplicate(Exception):
    """A detected event of the same tube and action is this close: probably the
    same event, so the reviewer is asked before adding it again."""

    def __init__(self, row):
        super().__init__("A detected event of the same tube and action is within "
                         f"{DUPLICATE_SECONDS:g} s")
        self.row = row


def _detected_rows(run):
    from .aggregate import analyzer_outputs

    return [primitives.normalize_event(r) for k, out in analyzer_outputs(run) if k == "events"
            for r in (out.get("rows") or [])]


def add_event(run, user, *, frame, action, reference, track_id="", end_frame=None,
              note="", fps=25.0, force=False):
    """Record an event the system missed. Raises ValueError on a bad request
    and Duplicate when a detected event of the same tube and action is within
    DUPLICATE_SECONDS (unless ``force``)."""
    from .models import AddedEvent

    if action not in ACTIONS:
        raise ValueError("action must be exit, enter or stay")
    reference = primitives.reference_id(str(reference or "").strip())
    if not reference:
        raise ValueError("pick the tube (reference)")
    try:
        frame = int(frame)
        end_frame = int(end_frame) if end_frame not in (None, "") else None
    except (TypeError, ValueError):
        raise ValueError("frame must be a number")
    if action == "stay" and (end_frame is None or end_frame < frame):
        raise ValueError("a stay needs an end at or after its start")
    if action != "stay":
        end_frame = None
    if not force:
        window = DUPLICATE_SECONDS * float(fps or 25)
        for row in _detected_rows(run):
            try:
                near = abs(int(row.get("frame")) - frame) <= window
            except (TypeError, ValueError):
                continue
            if near and row.get("action") == action and str(row.get("reference")) == reference:
                raise Duplicate(row)
    return AddedEvent.objects.create(
        run=run, user=user, frame=frame, end_frame=end_frame, action=action,
        reference=reference, track_id=str(track_id or "").strip()[:32], note=(note or "")[:200])


def remove_added(run, user, pk):
    """The person who added it, or the run's owner, may take it back."""
    from .models import AddedEvent

    q = AddedEvent.objects.filter(run=run, pk=pk)
    if run.user_id != user.pk:
        q = q.filter(user=user)
    return q.delete()[0] > 0


def added_rows(run):
    """The run's added events, oldest first, as dicts the pages and CSVs read."""
    from .models import AddedEvent

    return [{"id": a.pk, "frame": a.frame, "end_frame": a.end_frame, "action": a.action,
             "reference": a.reference, "track_id": a.track_id, "user": a.user.username,
             "user_id": a.user_id, "note": a.note}
            for a in AddedEvent.objects.filter(run=run).select_related("user")]


def quality(accepted, rejected, added):
    """Precision and recall from reviews, as percentages, or None when there is
    nothing to divide by. Precision = right among the checked detections;
    recall = found among the events people confirmed or added."""
    precision = round(100 * accepted / (accepted + rejected)) if accepted + rejected else None
    recall = round(100 * accepted / (accepted + added)) if accepted + added else None
    return precision, recall
