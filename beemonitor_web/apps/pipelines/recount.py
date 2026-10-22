"""Recount a finished run's events and interactions from its saved tracks.

No GPU: the tracking table is already saved, and events and interactions are a
pure function of it and the reference geometry. Used when the rules change
(memory/55: events became entries and emergences) or a clip's layout was drawn
after it was analysed. The reference steps are resolved again, so the current
layout is the one counted against.

The new outputs replace the old ones in ``run.context``, which is what the batch
table, the CSV downloads, studies and the results page all read, so they all
move together.
"""

from __future__ import annotations

import logging

from django.utils import timezone

logger = logging.getLogger(__name__)

RECOUNTED = ("analyze.events", "analyze.interactions")
_REFERENCES = ("reference.", "roi.")


def recount_run(run, min_seconds=None) -> dict:
    """Recount one run in place. ``min_seconds`` overrides the interactions
    minimum where the node saved none or 0. Returns ``{kind: count}`` for
    what was recounted; an analyzer that errors keeps its old output."""
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
            if bt == "analyze.interactions" and min_seconds is not None:
                cfg = dict(step.get("config") or {})
                if not float(cfg.get("min_seconds") or 0):
                    cfg["min_seconds"] = min_seconds
                step = {**step, "config": cfg}
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
    return counts


# Bumped whenever the counting rules change; runs below it are recounted.
CURRENT_RULES = 2
SWEEP_BATCH = 20
SWEEP_SECONDS = 60


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
            recount_run(run, min_seconds=_default_min_seconds())
        except Exception:
            logger.exception("recount of run %s failed", pk)
        done += 1
    return done


def _default_min_seconds():
    from .executors import DEFAULT_MIN_SECONDS
    return DEFAULT_MIN_SECONDS
