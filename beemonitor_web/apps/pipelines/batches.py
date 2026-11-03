"""A batch's name (memory/64). A batch is the runs sharing a ``batch_id``;
``Batch`` holds a name only once someone gives it one, and a batch with none
shows its pipeline's title."""

from __future__ import annotations

NAME_MAX = 120


def names(batch_ids):
    """``{batch_id: name}`` for the batches that have one."""
    from .models import Batch

    return dict(Batch.objects.filter(batch_id__in=list(batch_ids)).exclude(name="")
                .values_list("batch_id", "name"))


def title(batch_id, pipeline=None):
    """What a batch is called: its name, else its pipeline's title."""
    return (names([batch_id]).get(batch_id)
            or (pipeline.title if pipeline is not None else "") or "Batch")


def rename(batch_id, user, name):
    """Set (or, blank, clear) a batch's name. Returns the stored name."""
    from .models import Batch

    name = " ".join((name or "").split())[:NAME_MAX]
    Batch.objects.update_or_create(batch_id=batch_id,
                                   defaults={"name": name, "created_by": user})
    return name


# The event actions review metrics can count (memory/70).
ACTIONS = ("exit", "enter", "stay")


def review_actions(batch_id):
    """The actions the batch's review metrics count; all of them by default."""
    from .models import Batch

    chosen = (Batch.objects.filter(batch_id=batch_id).values_list("review_actions", flat=True)
              .first()) or []
    chosen = [a for a in chosen if a in ACTIONS]
    return chosen or list(ACTIONS)


def set_review_actions(batch_id, user, actions):
    """Save which actions count; at least one. Returns what was saved."""
    from .models import Batch

    chosen = [a for a in ACTIONS if a in set(actions or [])] or list(ACTIONS)
    row, _ = Batch.objects.get_or_create(batch_id=batch_id, defaults={"created_by": user})
    row.review_actions = [] if len(chosen) == len(ACTIONS) else chosen
    row.save(update_fields=["review_actions", "updated_at"])
    return chosen


def neighbours(run):
    """Where a clip sits in its batch, for previous/next on its results page:
    ``{position, total, prev, next, batch_url}`` (prev/next are
    ``{url, title}`` or None), in recording order; None outside a batch."""
    from django.urls import reverse

    from apps.videos.models import Video

    from .aggregate import run_video_id
    from .models import PipelineRun

    if run is None or not run.batch_id:
        return None
    # Not the contexts (memory/75): each run's job is its own column.
    runs = list(PipelineRun.objects.filter(batch_id=run.batch_id,
                                           status=PipelineRun.Status.COMPLETED)
                .only("pk", "steps", "batch_id", "gpu_job_id"))
    videos = Video.everything.in_bulk([v for r in runs if (v := run_video_id(r))])
    clips = []
    for r in runs:
        video, job = videos.get(run_video_id(r)), r.gpu_job_id
        if video is None or not job:
            continue
        clips.append((video.recorded_at is None, video.recorded_at, video.title or "", str(r.pk),
                      job, video.title or f"Video {video.pk}"))
    clips.sort(key=lambda c: c[:4])
    at = next((i for i, c in enumerate(clips) if c[3] == str(run.pk)), None)
    if at is None:
        return None

    def link(i):
        if not 0 <= i < len(clips):
            return None
        c = clips[i]
        return {"url": reverse("analysis:results", args=[c[4]]) + f"?run={c[3]}", "title": c[5]}

    return {"position": at + 1, "total": len(clips), "prev": link(at - 1), "next": link(at + 1),
            "batch_url": reverse("pipelines:batch_detail", args=[run.batch_id]),
            "batch_name": title(run.batch_id, getattr(run, "pipeline", None))}

