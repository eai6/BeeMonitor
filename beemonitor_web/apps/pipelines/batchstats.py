"""A big batch's totals, worked out apart from its page (memory/81).

The downloads on offer, the reference summary, the review metrics, what the
insect groups left out, where an update has got to and what changed in the
pipeline all read every clip's results. On a 6,306-clip batch that was most
of a 24 s page load, repeated on every load. They are worked out once here,
in the background, stored on ``BatchStats`` with a fingerprint of what they
were worked out from, and worked out again only when that changes: a run of
the batch finished, was recounted or re-run, a review was cast, the pipeline
or the batch's counted actions were edited.

One web process at a time works a batch's totals out (``working_since``),
and not again within ``REST_SECONDS`` of the last time: while the recount
worker goes through a batch every recount changes it, and redoing them after
each one kept two copies of the whole batch in memory (memory/82).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading

from django.core.serializers.json import DjangoJSONEncoder

logger = logging.getLogger(__name__)

# Bump when what is worked out changes: every batch's totals are redone.
VERSION = 3   # 3: a batch counts its own reviews (memory/84)

_working = set()
_lock = threading.Lock()
# Totals worked out this recently are shown (marked as being redone) rather
# than worked out again.
REST_SECONDS = 10 * 60
# A process that died while working: its claim is given up after this long.
CLAIM_SECONDS = 15 * 60


def totals(runs, rows, batch_id, pipeline, counted, photos, can_share=True) -> dict:
    """The batch page's totals from ``runs`` (slim) and their ``rows``
    (aggregate.batch_rows)."""
    from collections import Counter

    from . import aggregate, regroup
    from . import review as review_mod
    from .views import _backfill_interactions_paths, _update_status

    sources, skipped = aggregate.collect_sources(runs)
    _backfill_interactions_paths(sources)
    downloads = aggregate.available_downloads(sources, runs)
    if photos:
        skipped = []
        downloads = [{"kind": "photos", "label": "Photos", "analyzed": False,
                      "hint": "One row per insect: photo, time taken, box, class, "
                              "species and its confidence."}]
    plain = [r["run"] for r in rows]
    per_run = {r["run"].pk: r.get("review") for r in rows}
    return {
        "downloads": downloads,
        "skipped_sources": skipped,
        "skipped_reasons": Counter(s["reason"] for s in skipped).most_common(),
        "references": aggregate.reference_summary(runs, rows),
        "review_summary": (review_mod.batch_review_summary(plain, per_run, counted)
                           if not photos else None),
        "group_summary": (review_mod.batch_group_summary(plain, per_run)
                          if not photos else None),
        "update_status": _update_status(plain) if can_share and not photos else None,
        "changes": (regroup.batch_changes(plain, pipeline)
                    if can_share and pipeline is not None and not photos else None),
        "reviewed_clips": sum(1 for r in rows if r["status"] == "completed"
                              and review_mod.clip_touched(r.get("review"))),
    }


def fingerprint(batch_id, pipeline, counted) -> str:
    """What the totals depend on, as a hash: a few aggregate queries."""
    from django.db.models import Count, Max, Q, Sum

    from .models import AddedEvent, ClipReview, EventReview, PipelineRun, TripReview

    runs = PipelineRun.objects.filter(batch_id=batch_id)
    parts = [VERSION, str(batch_id), sorted(counted) if counted is not None else None,
             getattr(pipeline, "pk", None), getattr(pipeline, "updated_at", None)]
    parts.append(runs.aggregate(
        n=Count("id"), done=Count("id", filter=Q(status=PipelineRun.Status.COMPLETED)),
        failed=Count("id", filter=Q(status=PipelineRun.Status.FAILED)),
        results=Max("results_at"), started=Max("started_at"), completed=Max("completed_at"),
        rules=Sum("counting_rules")))
    # This batch's reviews (review.shared_job_runs, memory/84).
    for model, run_field, when in ((EventReview, "run", "updated_at"),
                                   (TripReview, "exit_run", "updated_at"),
                                   (AddedEvent, "run", "created_at"),
                                   (ClipReview, "run", "created_at")):
        parts.append(model.objects.filter(**{f"{run_field}__batch_id": batch_id})
                     .aggregate(n=Count("id"), last=Max(when), top=Max("id")))
    return hashlib.sha256(json.dumps(parts, cls=DjangoJSONEncoder, sort_keys=True)
                          .encode()).hexdigest()


def stored(batch_id, pipeline, counted):
    """``(totals or None, current)``: the batch's stored totals and whether
    they still match it. Starts working them out when they don't."""
    from .models import BatchStats

    from django.utils import timezone

    fp = fingerprint(batch_id, pipeline, counted)
    row = BatchStats.objects.filter(batch_id=batch_id).first()
    if row is not None and row.fingerprint == fp:
        return row.data, True
    if row is not None and row.data and not _testing() and \
            (timezone.now() - row.computed_at).total_seconds() < REST_SECONDS:
        return row.data, False                   # redone once they've rested
    done = start(batch_id, counted, fp)
    if done is not None:
        return done, True
    return (row.data if row is not None else None), False


def start(batch_id, counted, fp):
    """Work the totals out in a thread of this process, once at a time; in
    the test run, now (its database is one connection's), returning them."""
    if _testing():
        return compute(batch_id, counted, fp)
    key = str(batch_id)
    with _lock:
        if key in _working:
            return None
        _working.add(key)
    if not _claim(batch_id):
        with _lock:
            _working.discard(key)
        return None                              # another process is on it
    threading.Thread(target=_work, args=(batch_id, counted, fp),
                     name=f"batchstats-{key[:8]}", daemon=True).start()
    return None


def _work(batch_id, counted, fp):
    from django.db import connection

    from .models import BatchStats

    try:
        compute(batch_id, counted, fp)
    except Exception:
        logger.exception("batch totals for %s failed", batch_id)
    finally:
        with _lock:
            _working.discard(str(batch_id))
        try:
            BatchStats.objects.filter(batch_id=batch_id).update(working_since=None)
        finally:
            connection.close()


def _testing() -> bool:
    return sys.argv[1:2] == ["test"]


def _claim(batch_id) -> bool:
    """Claim working the batch's totals out, for this process: no other web
    process is, or the one that was gave up long ago."""
    from datetime import timedelta

    from django.db.models import Q
    from django.utils import timezone

    from .models import BatchStats

    now = timezone.now()
    BatchStats.objects.get_or_create(batch_id=batch_id, defaults={
        "fingerprint": "", "data": {}, "computed_at": now})
    return bool(BatchStats.objects.filter(batch_id=batch_id).filter(
        Q(working_since=None) | Q(working_since__lt=now - timedelta(seconds=CLAIM_SECONDS)))
        .update(working_since=now))


def compute(batch_id, counted, fp=None) -> dict:
    """Work out and store the totals of the whole batch."""
    from django.utils import timezone

    from . import aggregate, runsummary
    from .models import BatchStats, PipelineRun
    from .registry import pipeline_input_kind

    qs = PipelineRun.objects.filter(batch_id=batch_id).defer("context").order_by("started_at", "id")
    runs = runsummary.use_slim(runsummary.share_steps(qs))
    if not runs:
        return {}
    pipeline = runs[0].pipeline
    fp = fp or fingerprint(batch_id, pipeline, counted)
    rows = aggregate.batch_rows(runs, counted)
    data = totals(runs, rows, batch_id, pipeline, counted,
                  pipeline_input_kind(runs[0].steps) == "photo")
    data = json.loads(json.dumps(data, cls=DjangoJSONEncoder))
    BatchStats.objects.update_or_create(batch_id=batch_id, defaults={
        "fingerprint": fp, "data": data, "computed_at": timezone.now()})
    return data
