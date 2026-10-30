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
    return {"events": len(keys), "reviewed": len(current),
            "accepted": calls.count("accepted"), "rejected": calls.count("rejected"),
            "disputed": calls.count("disputed"),
            "gone": len(summaries) - len(current)}


def review_columns(summaries, key):
    """The Events CSV's review columns for one event."""
    s = summaries.get(key) or {}
    return {"review": s.get("review", ""), "reviewed_by": s.get("reviewed_by", ""),
            "review_notes": s.get("review_notes", "")}


def run_for_job(job):
    from .executors import _run_for_job

    return _run_for_job(job)


def summaries_for_job(job):
    run = run_for_job(job)
    return event_summaries(run) if run is not None else {}


def batch_review_counts(runs):
    """``{run pk: counts}`` (as ``event_review_counts``) for many runs in two
    queries' worth of work; None for a run with no Events step."""
    from .aggregate import analyzer_outputs
    from .models import EventReview

    runs = list(runs)
    votes = {}
    for run_id, key, verdict in (EventReview.objects.filter(run__in=runs)
                                 .values_list("run_id", "event_key", "verdict")):
        votes.setdefault(run_id, {}).setdefault(key, []).append(verdict)
    out = {}
    for run in runs:
        outs = [o for k, o in analyzer_outputs(run) if k == "events"]
        if not outs:
            out[run.pk] = None
            continue
        keys = {event_key(r) for o in outs for r in (o.get("rows") or [])}
        mine = votes.get(run.pk, {})
        calls = [combined(v) for k, v in mine.items() if k in keys]
        out[run.pk] = {"events": len(keys), "reviewed": len(calls),
                       "accepted": calls.count("accepted"), "rejected": calls.count("rejected"),
                       "disputed": calls.count("disputed")}
    return out


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
