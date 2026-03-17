"""Batch processing orchestrator using Modal's Function.map().

Processes multiple videos in parallel, each on its own GPU container.
"""

import logging
from typing import Optional

import modal

from cloud.modal_app.app import app
from cloud.modal_app.secrets import azure_secret

logger = logging.getLogger(__name__)


@app.function(
    timeout=14400,  # 4 hours for orchestration
    secrets=[azure_secret],
)
def batch_process(
    jobs: list[dict],
    use_gpu: bool = True,
) -> list[dict]:
    """Process multiple videos in parallel.

    Args:
        jobs: List of job configs, each with:
            - job_id: str
            - user_id: str
            - video_blob_path: str
            - detection_mode: str (optional, default 'yolo')
            - confidence_threshold: float (optional, default 0.25)
            - ml_threshold: float (optional, default 0.6)
            - visualize: bool (optional, default True)
        use_gpu: Use GPU processing (default True).

    Returns:
        List of result dicts from process_video.
    """
    from cloud.modal_app.functions.process_video import process_video

    logger.info("Starting batch processing of %d videos", len(jobs))

    # Build argument lists for starmap
    results = []
    for job_result in process_video.starmap(
        [
            (
                job["job_id"],
                job["user_id"],
                job["video_blob_path"],
                job.get("detection_mode", "yolo"),
                job.get("confidence_threshold", 0.25),
                job.get("ml_threshold", 0.6),
                job.get("visualize", True),
            )
            for job in jobs
        ]
    ):
        results.append(job_result)

    succeeded = sum(1 for r in results if r.get("total_events", -1) >= 0)
    logger.info(
        "Batch complete: %d/%d succeeded", succeeded, len(jobs)
    )
    return results
