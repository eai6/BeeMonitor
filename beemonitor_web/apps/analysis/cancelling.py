"""Cancelling GPU jobs so the GPU actually stops (memory/49).

SageMaker async inference has no API to abort a request. Marking a job
cancelled here used to be bookkeeping only: the clip still ran to the end on
the GPU and billed. Now each cancel also leaves a marker in S3 that the worker
watches for — a queued clip is answered in about a second, a running one stops
within about 20.
"""
from __future__ import annotations

import io
import logging

from django.utils import timezone

from .models import Job

logger = logging.getLogger(__name__)

CANCELLED_MESSAGE = "Cancelled by user."

# Must match beemonitor.cancellation.MARKER_PREFIX in the GPU image: the
# worker looks for "processed" bucket key cancel/<inference id>.
MARKER_PREFIX = "cancel/"

ACTIVE = [Job.Status.QUEUED, Job.Status.INGESTING,
          Job.Status.PROCESSING, Job.Status.POST_PROCESSING]


def _write_marker(job) -> None:
    """Best-effort: a job whose marker can't be written is still cancelled
    here; it just runs to the end on the GPU, as every cancel used to."""
    if not job.modal_job_id:
        return
    try:
        from config.storage import get_s3_client
        get_s3_client().upload_stream(
            "processed", MARKER_PREFIX + job.modal_job_id, io.BytesIO(b""),
            content_type="text/plain")
    except Exception:  # noqa: BLE001
        logger.exception("cancel: could not leave the marker for job %s", job.pk)


def cancel_jobs(jobs) -> int:
    """Cancel each still-active job: mark it, tell the GPU, fail its step."""
    from apps.pipelines import engine

    count = 0
    for job in jobs:
        if job.status not in ACTIVE:
            continue
        job.status = Job.Status.CANCELLED
        job.completed_at = timezone.now()
        job.error_message = CANCELLED_MESSAGE
        job.save(update_fields=["status", "completed_at", "error_message"])
        _write_marker(job)
        try:
            engine.on_job_finished(job)
        except Exception:  # noqa: BLE001
            logger.exception("cancel: pipeline hook failed for job %s", job.pk)
        count += 1
    return count
