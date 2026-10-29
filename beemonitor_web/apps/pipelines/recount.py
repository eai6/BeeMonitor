"""Recount a finished run's events and interactions from its saved tracks.

No GPU: the tracking table is already saved, and events and interactions are a
pure function of it and the reference geometry. Used when the rules change
(memory/60: exit / enter / stay from detected rows) or a clip's layout was
drawn after it was analysed.

Every finished run is recounted, backfilling the current rules from the
tracks it already saved (the user's call, 2026-10-29). A clip tracked before
the worker wrote the ``detected`` column can't say which rows were detections,
so its rows are all used and its outputs carry a note saying track starts and
ends may include the tracker's predicted positions — nothing is guessed to
fill the gap. The reference steps are resolved again, so the current layout
is the one counted against.

The new outputs replace the old ones in ``run.context``, which is what the batch
table, the CSV downloads, studies and the results page all read, so they all
move together.
"""

from __future__ import annotations

import logging

from django.utils import timezone

logger = logging.getLogger(__name__)

RECOUNTED = ("analyze.events", "analyze.interactions",
             # Retired blocks run as the primitives now (memory/60).
             "analyze.foraging_trips", "analyze.visitation", "analyze.interaction")
_REFERENCES = ("reference.", "roi.")


def recount_run(run) -> dict:
    """Recount one run in place. Returns ``{kind: count}`` for what was
    recounted; an analyzer that errors keeps its old output."""
    from . import executors

    steps = run.steps or []
    if not any(s.get("block_type") in RECOUNTED for s in steps):
        return {}
    context = dict(run.context or {})
    counts = {}
    for i, step in enumerate(steps):
        bt = str(step.get("block_type", ""))
        sid = step.get("id")
        if bt.startswith(_REFERENCES):
            out = executors.run_local_step(step, run, context, i)
            if out and not out.get("error"):
                context[sid] = out
        elif bt in RECOUNTED:
            out = executors.run_local_step(step, run, context, i)
            if not out or out.get("error"):
                logger.warning("recount: run %s step %s failed: %s", run.pk, sid,
                               (out or {}).get("error"))
                continue
            out["counted_at"] = timezone.now().isoformat()
            context[sid] = out
            kind = out.get("table_kind")
            counts[kind] = counts.get(kind, 0) + len(out.get("rows") or [])
    if counts:
        run.context = context
        type(run).objects.filter(pk=run.pk).update(context=context)
        sync_job_counts(run)
    return counts


def sync_job_counts(run):
    """Copy what the run's analyzers counted onto its JobResult, so the jobs
    list, clip page, dashboard and analytics read the pipeline's numbers and
    never the worker's (memory/60)."""
    from apps.analysis.models import JobResult

    from . import aggregate

    job_id = aggregate.run_job_id(run)
    if not job_id:
        return
    events = interactions = None
    for kind, out in aggregate.analyzer_outputs(run):
        if kind == "events":
            events = (events or {"total": 0, "enter": 0, "exit": 0})
            events["total"] += out.get("event_count") or 0
            events["enter"] += out.get("enter_count") or 0
            events["exit"] += out.get("exit_count") or 0
        elif kind == "interactions":
            interactions = (interactions or 0) + (out.get("interaction_count") or 0)
    fields = {}
    if events is not None:
        fields.update(total_events=events["total"], entry_count=events["enter"],
                      exit_count=events["exit"])
    if interactions is not None:
        fields["interaction_count"] = interactions
    if fields:
        JobResult.objects.filter(job_id=job_id).update(**fields)


# Bumped whenever the counting rules change; runs below it are recounted.
CURRENT_RULES = 4
# Sized to work through a backlog of every finished run after a rules change
# within the reconciler's 120 s interval.
SWEEP_BATCH = 60
SWEEP_SECONDS = 90


def sweep(batch=SWEEP_BATCH, budget=SWEEP_SECONDS) -> int:
    """Recount finished runs counted with older rules, newest first, a bounded
    batch per reconcile pass. Each run is marked whether or not it had
    anything to recount, so it's visited once."""
    import time

    from .models import PipelineRun

    deadline = time.monotonic() + budget
    done = 0
    ids = list(PipelineRun.objects.filter(status=PipelineRun.Status.COMPLETED,
                                          counting_rules__lt=CURRENT_RULES)
               .order_by("-completed_at").values_list("pk", flat=True)[:batch])
    for pk in ids:
        if time.monotonic() > deadline:
            break
        # Claimed first, so two workers don't recount the same run.
        if not PipelineRun.objects.filter(pk=pk, counting_rules__lt=CURRENT_RULES).update(
                counting_rules=CURRENT_RULES):
            continue
        run = PipelineRun.objects.get(pk=pk)
        try:
            recount_run(run)
        except Exception:
            logger.exception("recount of run %s failed", pk)
        done += 1
    return done
