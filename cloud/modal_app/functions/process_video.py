"""GPU function for running the BeeMonitor analysis pipeline.

This is the core processing function. It downloads a video from Azure,
runs YOLO detection + tracking + event classification, and uploads results.
"""

import json
import logging
import os

import modal

from cloud.modal_app.app import app
from cloud.modal_app.image import beemonitor_image
from cloud.modal_app.volumes import model_volume, MODEL_VOLUME_MOUNT
from cloud.modal_app.secrets import azure_secret

logger = logging.getLogger(__name__)


@app.function(
    gpu="L4",
    timeout=7200,  # 2 hours max
    retries=1,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[azure_secret],
    memory=8192,  # 8 GiB RAM
    max_containers=50,
)
def process_video(
    job_id: str,
    user_id: str,
    video_blob_path: str,
    detection_mode: str = "yolo",
    confidence_threshold: float = 0.25,
    ml_threshold: float = 0.6,
    visualize: bool = True,
) -> dict:
    """Run BeeMonitor analysis on a video stored in Azure Blob.

    Args:
        job_id: Unique job identifier.
        user_id: Owner of the video.
        video_blob_path: Path within raw-videos container.
        detection_mode: 'yolo' or 'yolo_only'.
        confidence_threshold: YOLO confidence threshold.
        ml_threshold: Event classifier threshold.
        visualize: Generate annotated video.

    Returns:
        Dict with job results (PipelineResult.to_dict()).
    """
    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig
    from cloud.wrapper.model_manager import ModelManager
    from cloud.wrapper.pipeline import CloudPipeline

    logger.info("[%s] Starting GPU processing for user %s", job_id, user_id)

    config = StorageConfig()
    storage = AzureBlobClient(config)
    models = ModelManager(
        storage_client=storage,
        storage_config=config,
        local_cache_dir=MODEL_VOLUME_MOUNT,
    )
    pipeline = CloudPipeline(
        storage_client=storage,
        storage_config=config,
        model_manager=models,
    )

    result = pipeline.process(
        job_id=job_id,
        user_id=user_id,
        video_blob_path=video_blob_path,
        detection_mode=detection_mode,
        confidence_threshold=confidence_threshold,
        ml_threshold=ml_threshold,
        visualize=visualize,
    )

    # Clean up local working files
    pipeline.cleanup(job_id)
    model_volume.commit()

    logger.info("[%s] GPU processing complete — %d events", job_id, result.total_events)
    return result.to_dict()


@app.function(
    gpu="L4",
    timeout=7200,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[azure_secret],
    memory=8192,
    max_containers=50,
)
def process_video_cpu_fallback(
    job_id: str,
    user_id: str,
    video_blob_path: str,
    detection_mode: str = "yolo",
    confidence_threshold: float = 0.25,
    ml_threshold: float = 0.6,
    visualize: bool = False,
) -> dict:
    """CPU-only fallback — same as process_video but without GPU.

    Used when GPU is unavailable or for cost savings on shorter videos.
    Disables visualization by default for speed.
    """
    return process_video.local(
        job_id=job_id,
        user_id=user_id,
        video_blob_path=video_blob_path,
        detection_mode=detection_mode,
        confidence_threshold=confidence_threshold,
        ml_threshold=ml_threshold,
        visualize=visualize,
    )
