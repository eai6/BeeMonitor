"""What a batch page lists of every clip, without reading their results
(memory/81).

A 6,306-clip batch page took 24 s on average: to list the clips and fill the
clip filter it read every run's steps and slim results, and counted every
event's review. ``PipelineRun.listing`` is the little the list needs (the
clip, the errors, the counts), kept beside ``context`` on every save; the
page reads it, the clips' own rows, their jobs and who reviewed them, and
nothing per clip that grows with the clip's events. The batch-wide totals
that do need every clip's results are worked out apart (batchstats).
"""

from __future__ import annotations

import logging
import threading

from types import SimpleNamespace

logger = logging.getLogger(__name__)

# Bump when what is kept changes: pages fill older ones from the summary.
VERSION = 2   # 2: the reference the analyzers are wired to (memory/82)
# A long traceback is no use in a list; the run's page has it whole.
ERROR_CHARS = 2000


def build(run) -> dict:
    """``run``'s listing, from its steps and ``context`` (whole or slim)."""
    from . import aggregate, references

    errors = [str(e)[:ERROR_CHARS] for out in (run.context or {}).values()
              if isinstance(out, dict) and (e := out.get("error"))]
    counts = aggregate.primitive_counts(run)
    return {"v": VERSION, "video": aggregate.run_video_id(run), "errors": errors,
            "events": counts.get("events"), "interactions": counts.get("interactions"),
            # The layout its events were counted against (references.py).
            "layout": references.counted(run)}


def is_current(listing) -> bool:
    return isinstance(listing, dict) and listing.get("v") == VERSION


def run_error(status, error_message, listing) -> str:
    """aggregate.run_error, from a listing."""
    if status != "failed":
        return ""
    candidates = list((listing or {}).get("errors") or [])
    real = [c for c in candidates if "Upstream step failed" not in c]
    return (real or candidates or [error_message or ""])[0]


def fill(pks) -> dict:
    """Listings for the runs ``pks`` that had none current, built from their
    slim summary (the whole context only where that's missing too) and
    saved, a few at a time. ``{pk: listing}``."""
    from . import runsummary
    from .models import PipelineRun

    pks, out = list(pks), {}
    for i in range(0, len(pks), 200):
        for pk, steps, summary in (PipelineRun.objects.filter(pk__in=pks[i:i + 200])
                                   .values_list("pk", "steps", "summary")):
            if runsummary.is_current(summary):
                context = summary["context"]
            else:
                context = PipelineRun.objects.filter(pk=pk).values_list("context", flat=True)[0]
            out[pk] = build(SimpleNamespace(pk=pk, steps=steps or [], context=context or {}))
            PipelineRun.objects.filter(pk=pk).update(listing=out[pk])
    return out


_refilling = threading.Lock()


def refill(pks):
    """``fill`` these in one thread of this process. One at a time, whatever
    page asked: the set of out-of-date listings changes with every recount,
    and a thread per page load ran several over a 6,000-clip batch at once
    and starved the web (memory/82). The thread reads the ones still out of
    date as it goes, so it picks up the rest too."""
    import sys

    pks = list(pks)
    if not pks:
        return
    if sys.argv[1:2] == ["test"]:
        fill(pks)
        return
    if not _refilling.acquire(blocking=False):
        return                                   # one is running already

    def work():
        import time

        from django.db import connection

        from .models import PipelineRun
        try:
            todo = pks
            while todo:
                for i in range(0, len(todo), 200):
                    fill(todo[i:i + 200])
                    time.sleep(0.5)          # leave the web its one vCPU
                todo = [pk for pk, lst in PipelineRun.objects.filter(pk__in=todo)
                        .values_list("pk", "listing") if not is_current(lst)]
        except Exception:
            logger.exception("refilling %s listings failed", len(pks))
        finally:
            _refilling.release()
            connection.close()

    threading.Thread(target=work, name="listing-refill", daemon=True).start()


def rows(qs) -> list[dict]:
    """The batch page's rows (aggregate.batch_rows's keys) for every run of
    ``qs``, read from the listings, oldest clip first. A row's ``run`` and
    ``result`` hold only what the list shows, not a run or a JobResult."""
    from apps.analysis.models import Job, JobResult
    from apps.annotations.leftout import video_site
    from apps.videos.models import Video

    runs = [SimpleNamespace(pk=pk, status=status, error_message=msg, listing=lst or {},
                            user_id=user_id, started_at=started, gpu_job_id=job_id)
            for pk, status, msg, lst, user_id, started, job_id
            in qs.values_list("pk", "status", "error_message", "listing", "user_id",
                              "started_at", "gpu_job_id").iterator(chunk_size=1000)]
    # None yet: built now. An older version: shown as it is, and rebuilt in
    # the background, as a version bump touches every clip of a 6,000-clip
    # batch, longer than a request may take (memory/82).
    filled = fill([r.pk for r in runs if not r.listing])
    refill([r.pk for r in runs if r.listing and not is_current(r.listing)])
    for r in runs:
        r.listing = filled.get(r.pk, r.listing)
    video_ids = [v for r in runs if (v := r.listing.get("video")) is not None]
    videos = {v.pk: v for v in Video.everything.filter(pk__in=video_ids)
              .select_related("device", "site")
              .only("pk", "title", "kind", "recorded_at", "duration_seconds", "site_name",
                    "site", "device", "metadata", "uploaded_at",
                    "site__name", "site__tz_name", "device__name", "device__location",
                    "device__tz_name", "device__tz_offset_min")}
    job_ids = [r.gpu_job_id for r in runs if r.gpu_job_id]
    seconds = dict(Job.objects.filter(pk__in=job_ids).values_list("pk", "execution_seconds"))
    tracks = dict(JobResult.objects.filter(job_id__in=job_ids).values_list("job_id", "unique_tracks"))
    reviewers = _reviewers(qs, runs)

    out = []
    for run in runs:
        video = videos.get(run.listing.get("video"))
        error = run_error(run.status, run.error_message, run.listing)
        names = reviewers.get(run.pk, [])
        out.append({
            "run": run,
            "video": video,
            "job": None,
            "result": (SimpleNamespace(unique_tracks=tracks[run.gpu_job_id])
                       if run.gpu_job_id in tracks else None),
            "primitives": {"events": run.listing.get("events"),
                           "interactions": run.listing.get("interactions")},
            "site": video_site(video) if video else "",
            # Anyone's call on it, a missed event added, or the clip marked
            # reviewed: who is the Reviewed by filter.
            "reviewed": bool(names),
            "reviewers": names,
            "status": run.status,
            "error": error,
            "cancelled": run.status == "failed" and "Cancelled by user" in error,
            "when": video.recorded_at if video else None,
            "gpu_seconds": float(seconds.get(run.gpu_job_id) or 0.0),
            "layout": run.listing.get("layout"),
        })
    out.sort(key=lambda r: (r["when"] is None, r["when"]))
    return out


def _reviewers(qs, runs) -> dict:
    """review.reviewers_by_run for the runs of ``qs``, by their GPU jobs in
    one query a kind of review, not by lists of thousands of run ids."""
    from .models import AddedEvent, ClipReview, EventReview, TripReview

    jobs = qs.exclude(gpu_job_id=None).values("gpu_job_id")
    names = {}
    for model, field in ((EventReview, "run"), (AddedEvent, "run"),
                         (ClipReview, "run"), (TripReview, "exit_run")):
        for job_id, name in (model.objects.filter(**{f"{field}__gpu_job_id__in": jobs})
                             .values_list(f"{field}__gpu_job_id", "user__username").distinct()):
            names.setdefault(job_id, set()).add(name)
    return {r.pk: sorted(names.get(r.gpu_job_id, ()), key=str.lower) for r in runs}
