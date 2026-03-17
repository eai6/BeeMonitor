import logging
import uuid

from celery import shared_task
from django.utils import timezone

from apps.analysis.models import Job, JobResult

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3, default_retry_delay=30)
def submit_analysis_job(self, job_id: int) -> None:
    """
    Submit an analysis job to the Modal processing pipeline.

    Workflow:
        1. Mark job as PROCESSING
        2. Call Modal's process_video function
        3. Store results in JobResult
        4. Mark job as COMPLETED (or FAILED on error)
    """
    try:
        job = Job.objects.select_related("video").get(pk=job_id)
    except Job.DoesNotExist:
        logger.error("Job %s does not exist; aborting.", job_id)
        return

    job.status = Job.Status.PROCESSING
    job.started_at = timezone.now()
    job.modal_job_id = f"modal_{uuid.uuid4().hex[:16]}"
    job.save(update_fields=["status", "started_at", "modal_job_id"])

    try:
        # TODO: Replace with actual Modal remote call, e.g.:
        #   from cloud.modal_app import process_video
        #   result_payload = process_video.remote(
        #       blob_path=job.video.azure_blob_path,
        #       config=job.config,
        #   )
        # For now, simulate a successful result.
        result_payload = {
            "events_csv_path": f"results/{job.modal_job_id}/events.csv",
            "tracking_csv_path": f"results/{job.modal_job_id}/tracking.csv",
            "annotated_video_path": f"results/{job.modal_job_id}/annotated.mp4",
            "total_events": 0,
            "entry_count": 0,
            "exit_count": 0,
            "unique_tracks": 0,
            "nest_count": 0,
            "summary_stats": {},
        }

        JobResult.objects.update_or_create(
            job=job,
            defaults=result_payload,
        )

        job.status = Job.Status.COMPLETED
        job.progress_pct = 100
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "progress_pct", "completed_at"])

        logger.info("Job %s completed successfully.", job_id)

    except Exception as exc:
        job.status = Job.Status.FAILED
        job.error_message = str(exc)
        job.save(update_fields=["status", "error_message"])
        logger.exception("Job %s failed.", job_id)
        raise self.retry(exc=exc)


@shared_task
def check_job_status(job_id: int) -> dict:
    """
    Poll the current status of a job.

    Returns a dict with key job fields so the caller (e.g. a periodic beat
    schedule or webhook dispatcher) can react accordingly.
    """
    try:
        job = Job.objects.get(pk=job_id)
    except Job.DoesNotExist:
        return {"error": f"Job {job_id} not found"}

    return {
        "job_id": job.id,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "modal_job_id": job.modal_job_id,
    }
