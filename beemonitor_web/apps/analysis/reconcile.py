"""System-wide background reconciliation.

The processing architecture is fire-and-forget: Django writes a payload to S3,
invokes the SageMaker async endpoint, and results land back in S3. Nothing
pushes completions to Django — until this module, state only converged while a
browser had a page open (the poll endpoints) and daemon threads survived
(every deploy restarts the App Runner container and kills them mid-flight).
That made the pipeline look unstable: jobs and runs stuck "Processing"
whenever a deploy or a closed laptop interrupted the chain.

This reconciler is the single convergence loop, run from a daemon thread in
the web process itself (no extra infra, restarts with every deploy, which is
fine — each pass is idempotent and stateless):

  * every PROCESSING Job is checked against S3 (results, failures, respawn of
    never-spawned jobs, hard timeout) — same code path the browser poller uses;
  * every in-flight PipelineRun is re-driven against its jobs' real state.

Multiple gunicorn workers / App Runner instances may each run the loop;
passes are cheap (a handful of S3 GETs) and idempotent, so overlap is safe.
"""

import logging
import os
import random
import sys
import threading
import time

logger = logging.getLogger(__name__)

_INTERVAL_SECONDS = int(os.environ.get("BEEMONITOR_RECONCILE_INTERVAL", "120"))
_started = False
_start_lock = threading.Lock()


def reconcile_all(limit: int = 500) -> dict:
    """One idempotent convergence pass over all users. Returns pass stats."""
    from django.db import close_old_connections

    from django.contrib.auth import get_user_model

    from apps.analysis.models import Job
    from apps.analysis.views import _poll_sagemaker_results, poll_annotate_jobs
    from apps.pipelines import engine
    from apps.pipelines.models import PipelineRun

    close_old_connections()

    jobs = list(
        Job.objects.filter(status=Job.Status.PROCESSING)
        .order_by("-started_at")[:limit]
    )
    resolved = _poll_sagemaker_results(jobs) if jobs else 0

    run_user_ids = list(
        PipelineRun.objects.filter(
            status__in=[PipelineRun.Status.PENDING, PipelineRun.Status.RUNNING],
        ).values_list("user_id", flat=True).distinct()[:50]
    )
    for uid in run_user_ids:
        engine.reconcile_user_runs(uid)

    # On-demand annotated-video renders in flight.
    User = get_user_model()
    annotate_user_ids = list(
        Job.objects.filter(config__annotate__status="processing")
        .values_list("user_id", flat=True).distinct()[:50]
    )
    for uid in annotate_user_ids:
        user = User.objects.filter(pk=uid).first()
        if user:
            poll_annotate_jobs(user)

    # SageMaker training jobs (advance without an open browser).
    training = {"checked": 0, "completed": 0}
    try:
        from apps.training.views import poll_training_jobs
        training = poll_training_jobs()  # all users
    except Exception:
        logger.exception("training poll failed")

    # Pre-annotation tasks (durable — finalize writes Annotation rows).
    preannot_done = 0
    try:
        from apps.annotations.views import poll_preannotation_tasks
        preannot_done = poll_preannotation_tasks()  # all users
    except Exception:
        logger.exception("pre-annotation poll failed")

    # Auto-adaptation runs (relabel → train → evaluate) advance headlessly.
    try:
        from apps.training import orchestrator
        from apps.training.models import AdaptationRun
        active = [AdaptationRun.Status.RELABELING, AdaptationRun.Status.TRAINING,
                  AdaptationRun.Status.EVALUATING]
        for adapt_run in AdaptationRun.objects.filter(status__in=active)[:50]:
            try:
                orchestrator.advance_run(adapt_run)
            except Exception:
                logger.exception("adaptation advance failed for run %s", adapt_run.pk)
    except Exception:
        logger.exception("adaptation reconcile failed")

    # Foraging summaries flagged stale by job completions (or never computed):
    # recompute a bounded batch per tick, newest day first, so device charts
    # converge without any S3 work in request paths.
    trips_recomputed = 0
    try:
        from apps.analysis import foraging
        trips_recomputed = foraging.sweep_stale()
    except Exception:
        logger.exception("foraging summary sweep failed")

    return {"jobs_checked": len(jobs), "jobs_resolved": resolved,
            "run_users": len(run_user_ids), "annotate_users": len(annotate_user_ids),
            "training_completed": training.get("completed", 0),
            "preannot_finalized": preannot_done,
            "trips_recomputed": trips_recomputed}


def _loop():
    # Let the app finish booting; jitter staggers multiple workers.
    time.sleep(30 + random.uniform(0, 30))
    while True:
        try:
            stats = reconcile_all()
            if stats["jobs_checked"] or stats["run_users"]:
                logger.info("reconcile pass: %s", stats)
        except Exception:  # the loop must survive anything
            logger.exception("background reconcile pass failed")
        time.sleep(_INTERVAL_SECONDS)


# Don't spin up the loop for one-shot management commands.
_MANAGEMENT_COMMANDS = {
    "migrate", "makemigrations", "collectstatic", "shell", "test",
    "createsuperuser", "loaddata", "dumpdata", "check", "showmigrations",
}


def start_background_reconciler():
    """Start the reconcile loop once per process (called from AppConfig.ready)."""
    global _started
    if os.environ.get("BEEMONITOR_RECONCILE_DISABLED") == "1":
        return
    if any(cmd in sys.argv for cmd in _MANAGEMENT_COMMANDS):
        return
    with _start_lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_loop, name="bm-reconcile", daemon=True).start()
    logger.info("background reconciler started (interval %ss)", _INTERVAL_SECONDS)
