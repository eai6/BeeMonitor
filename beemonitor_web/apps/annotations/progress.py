"""Where each clip has got to, and where the project has.

The page used to describe a project by verbs — sample, annotate, auto-label,
export — numbered 1 to 4 as though you run them once, top to bottom. The real
loop is per clip: add a few, sample those, label those, review those, repeat.
So the page needs to say what STATE things are in, and the numbers for that
were already in the database, just never asked for together.

A clip moves through four stages. Each is a superset of the work in the last:

    added -> sampled -> labelled -> reviewed

``frames`` is what sampling produced, ``labelled`` how many of those carry
boxes, ``reviewed`` how many a person or the LLM has signed off. A clip is
named by the furthest stage it has actually reached, so "sampled" means
"sampled and nothing more" — the state you would act on next.
"""

from django.db.models import Count, Q

# Order matters: a clip is named by the last stage it satisfies.
STAGES = ("failed", "new", "empty", "sampled", "labelled", "reviewed")

STAGE_LABELS = {
    "new": "Not sampled",
    # Sampled, and nothing worth keeping was found (an empty trigger, or no
    # insect moving). Done — not the same as never sampled.
    "empty": "No activity",
    "sampled": "Sampled",
    # Labelled by SAM 3 (or a person), not yet checked.
    "labelled": "To review",
    "reviewed": "Reviewed",
    "failed": "Failed",
}


def per_video(project, failed_ids=()):
    """``{video_id: {...}}`` — frames, labelled, reviewed and the stage name.

    One query. The alternative — a count per clip — is 59 queries on this
    project and 3,250 on a big one.
    """
    from .models import Annotation

    rows = (Annotation.objects.filter(project=project).order_by()
            .values("video_id")
            .annotate(frames=Count("id"),
                      # A frame carries boxes when it is not the empty
                      # placeholder sampling wrote.
                      labelled=Count("id", filter=~Q(boxes=[])),
                      reviewed=Count("id", filter=Q(reviewed=True))))

    failed_ids = {int(v) for v in failed_ids or ()}
    out = {}
    for r in rows:
        out[r["video_id"]] = _shape(r["video_id"], r["frames"], r["labelled"],
                                    r["reviewed"], failed_ids)
    # Clips whose sampling finished with no frames have no rows to aggregate;
    # without this they read as "Not sampled" and get sampled again and again.
    for vid in empty_ids(project) - set(out):
        out[vid] = _shape(vid, 0, 0, 0, failed_ids, sampled=True)
    # And a clip whose only run failed has no rows either.
    for vid in failed_ids - set(out):
        out[vid] = _shape(vid, 0, 0, 0, failed_ids)
    return out


def empty_ids(project):
    """Clips a finished sampling run left with no frames."""
    from .models import FrameSamplingTask
    return set(FrameSamplingTask.objects.filter(
        project=project, status=FrameSamplingTask.Status.COMPLETED,
    ).values_list("video_id", flat=True))


def _shape(video_id, frames, labelled, reviewed, failed_ids, sampled=False):
    if video_id in failed_ids:
        stage = "failed"
    elif not frames:
        stage = "empty" if sampled else "new"
    elif reviewed and reviewed >= labelled and labelled:
        stage = "reviewed"
    elif labelled:
        stage = "labelled"
    else:
        stage = "sampled"
    return {
        "frames": frames, "labelled": labelled, "reviewed": reviewed,
        "unlabelled": max(frames - labelled, 0),
        "stage": stage, "stage_label": STAGE_LABELS[stage],
        # Percentages for the row's progress bar, so the template does no maths.
        "pct_labelled": round(100 * labelled / frames) if frames else 0,
        "pct_reviewed": round(100 * reviewed / frames) if frames else 0,
    }


def decorate(videos, states, failed_ids=()):
    """Attach a state to each video, inventing one for clips with no frames."""
    failed_ids = {int(v) for v in failed_ids or ()}
    for v in videos:
        v.progress = states.get(v.pk) or _shape(v.pk, 0, 0, 0, failed_ids)
    return videos


def summary(project, states, video_count, failed_ids=()):
    """The four stages, as the funnel across the top of the page.

    Each stage carries what is DONE and what is outstanding, because the
    outstanding half is what you are deciding about — a project page that only
    reports totals tells you how much work happened, never what to do next.
    """
    frames = sum(s["frames"] for s in states.values())
    labelled = sum(s["labelled"] for s in states.values())
    reviewed = sum(s["reviewed"] for s in states.values())
    sampled_clips = sum(1 for s in states.values() if s["frames"])
    empty_clips = sum(1 for s in states.values() if s["stage"] == "empty")
    counts = stage_counts(states, video_count, failed_ids)

    return {
        "clips": video_count,
        "clips_sampled": sampled_clips,
        "clips_empty": empty_clips,
        "clips_unsampled": max(video_count - sampled_clips - empty_clips, 0),
        "frames": frames,
        "labelled": labelled,
        "unlabelled": max(frames - labelled, 0),
        "reviewed": reviewed,
        "unreviewed": max(labelled - reviewed, 0),
        "pct_labelled": round(100 * labelled / frames) if frames else 0,
        "pct_reviewed": round(100 * reviewed / frames) if frames else 0,
        "pct_sampled": round(100 * sampled_clips / video_count) if video_count else 0,
        "stage_counts": counts,
    }


def stage_counts(states, video_count, failed_ids=()):
    """How many clips sit in each stage — the selector's own numbers."""
    counts = {s: 0 for s in STAGES}
    seen = 0
    for state in states.values():
        counts[state["stage"]] += 1
        seen += 1
    # Clips with no Annotation rows at all never appear in the aggregate.
    counts["new"] += max(video_count - seen, 0)
    return counts


def filter_ids(states, video_count, stage):
    """The video ids in one stage, or None for "no stage filter".

    ``new`` is the awkward one: those clips have no rows to aggregate, so they
    cannot be found in ``states`` and the caller has to subtract instead.
    """
    if stage not in STAGE_LABELS:
        return None
    return {vid for vid, s in states.items() if s["stage"] == stage}
