"""Recurring per-device pipeline launches.

A ``DevicePipelineSchedule`` says "run this pipeline over this device's new videos
every N hours". There is no cron, Celery beat, or scheduled task in this deployment
— the background reconciler daemon (``apps.analysis.reconcile._loop``, ~120s) is the
clock, and it calls :func:`run_due_schedules` once per pass.

Two properties matter because several gunicorn workers / App Runner instances each
run that loop:

* **Claim before launching.** A schedule is claimed with a compare-and-swap on
  ``last_run_at`` (``filter(pk=..., last_run_at=<observed>).update(...)``), the same
  technique ``analysis.views._drain_queue`` uses to claim QUEUED jobs. Only the
  worker whose UPDATE matched a row proceeds, so a schedule can't double-launch.
* **Never raise.** A broken schedule records ``last_error`` and is skipped; it must
  not take down a reconcile pass that also polls jobs, training and annotations.

The launch itself is ``pipelines.engine.launch_batch`` — the exact path the
Processing hub uses — so scheduled runs are indistinguishable from manual ones
(same batch pages, same combined CSVs, same QUEUED/_drain_queue backpressure).
"""

import logging

from django.utils import timezone

logger = logging.getLogger(__name__)


def _videos_for(schedule):
    """The device's videos recorded inside this schedule's window."""
    from apps.analysis.views import apply_video_filters
    from apps.videos.models import Video

    params = {"device": str(schedule.device_id)}
    start = schedule.window_start()
    if start is not None:
        params["from"] = start.isoformat()

    qs = apply_video_filters(Video.manageable(schedule.user), params)
    return qs.order_by("-recorded_at", "-uploaded_at")


def run_schedule(schedule):
    """Launch one due schedule. Returns the number of runs started (0 on no-op).

    Assumes the caller already claimed the schedule (see :func:`run_due_schedules`).
    """
    from django.conf import settings

    from apps.pipelines import engine

    max_batch = getattr(settings, "PIPELINE_MAX_BATCH", 50000)
    qs = _videos_for(schedule)
    videos = list(qs[:max_batch] if max_batch else qs)
    if not videos:
        schedule.last_launched_count = 0
        schedule.last_error = ""
        schedule.save(update_fields=["last_launched_count", "last_error"])
        return 0

    batch_id, launched, invalid = engine.launch_batch(
        schedule.pipeline, videos, schedule.user,
    )
    if launched:
        # The GPU steps were created QUEUED. One kick promotes up to the global
        # SageMaker cap now; the reconciler drains the rest on later passes.
        try:
            from apps.analysis.views import _drain_queue

            _drain_queue()
        except Exception:
            logger.exception("inline drain after scheduled launch failed")

    schedule.last_batch_id = batch_id if launched else None
    schedule.last_launched_count = len(launched)
    schedule.last_error = (
        f"{invalid} video(s) skipped — the pipeline is not valid for them."
        if invalid and not launched else ""
    )
    schedule.save(update_fields=["last_batch_id", "last_launched_count", "last_error"])
    logger.info(
        "device schedule %s launched %d run(s) (batch %s, %d skipped)",
        schedule.pk, len(launched), batch_id, invalid,
    )
    return len(launched)


def run_due_schedules(limit=50):
    """One idempotent pass over every due schedule. Returns pass stats.

    Called from ``apps.analysis.reconcile.reconcile_all``. Never raises.
    """
    from .models import DevicePipelineSchedule

    stats = {"due": 0, "launched_runs": 0, "errors": 0}
    try:
        now = timezone.now()
        candidates = list(
            DevicePipelineSchedule.objects
            .filter(enabled=True)
            .select_related("device", "pipeline", "user")[:limit]
        )
        for schedule in candidates:
            if not schedule.is_due(now):
                continue
            stats["due"] += 1

            # Claim: only the worker whose UPDATE matched the observed last_run_at
            # proceeds. Stamping the time up-front also means a schedule whose
            # launch explodes still waits a full interval before retrying.
            claimed = DevicePipelineSchedule.objects.filter(
                pk=schedule.pk, last_run_at=schedule.last_run_at,
            ).update(last_run_at=now)
            if not claimed:
                continue
            schedule.last_run_at = now

            try:
                stats["launched_runs"] += run_schedule(schedule)
            except Exception as exc:
                stats["errors"] += 1
                logger.exception("device schedule %s failed", schedule.pk)
                DevicePipelineSchedule.objects.filter(pk=schedule.pk).update(
                    last_error=str(exc)[:500], last_launched_count=0,
                )
    except Exception:  # a bad pass must never break the reconciler
        logger.exception("run_due_schedules failed")
    return stats
