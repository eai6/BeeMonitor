"""Main Modal application definition for BeeMonitor Cloud.

Usage:
    modal serve cloud/modal_app/app.py    # Dev with live-reload
    modal deploy cloud/modal_app/app.py   # Production deployment
"""

import modal

from cloud.modal_app.image import beemonitor_image
from cloud.modal_app.volumes import model_volume, MODEL_VOLUME_MOUNT

app = modal.App("beemonitor-cloud", image=beemonitor_image)


# ── Health check ──────────────────────────────────────────────────────

@app.function(min_containers=1)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "service": "beemonitor-cloud"}


# ── GPU Processing (single video) ────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=7200,
    retries=1,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[modal.Secret.from_name("azure-storage")],
    memory=8192,
    min_containers=0,
    scaledown_window=60,
)
def process_video(
    job_id: str,
    user_id: str,
    video_blob_path: str,
    detection_mode: str = "yolo",
    confidence_threshold: float = 0.25,
    ml_threshold: float = 0.6,
    visualize: bool = True,
    two_mode_tracking: bool = True,
) -> dict:
    """Run BeeMonitor analysis on a video stored in Azure Blob."""
    import time
    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig
    from cloud.wrapper.model_manager import ModelManager
    from cloud.wrapper.pipeline import CloudPipeline

    start_time = time.time()

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
        two_mode_tracking=two_mode_tracking,
    )

    pipeline.cleanup(job_id)
    model_volume.commit()

    result_dict = result.to_dict()
    result_dict["execution_seconds"] = round(time.time() - start_time, 1)
    return result_dict


# ── Full Pipeline: S3 Transfer + GPU Processing ──────────────────────

@app.function(
    gpu="A10G",
    timeout=7200,
    retries=1,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[modal.Secret.from_name("azure-storage")],
    memory=8192,
    min_containers=0,
    scaledown_window=60,
)
def process_video_from_s3(
    job_id: str,
    user_id: str,
    s3_bucket: str,
    s3_key: str,
    s3_access_key_id: str,
    s3_secret_access_key: str,
    s3_region: str = "us-east-1",
    detection_mode: str = "yolo",
    confidence_threshold: float = 0.25,
    visualize: bool = True,
    two_mode_tracking: bool = True,
) -> dict:
    """Transfer video from S3 to Azure, then run BeeMonitor analysis."""
    import boto3
    import tempfile
    import time
    import logging
    from pathlib import Path

    start_time = time.time()
    logger = logging.getLogger(__name__)

    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig
    from cloud.wrapper.model_manager import ModelManager
    from cloud.wrapper.pipeline import CloudPipeline

    config = StorageConfig()
    storage = AzureBlobClient(config)

    # Step 1: Transfer S3 → Azure
    filename = s3_key.split("/")[-1]
    azure_blob_path = f"{user_id}/{job_id}/{filename}"

    s3 = boto3.client(
        "s3",
        aws_access_key_id=s3_access_key_id,
        aws_secret_access_key=s3_secret_access_key,
        region_name=s3_region,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        logger.info("[%s] Downloading s3://%s/%s", job_id, s3_bucket, s3_key)
        s3.download_file(s3_bucket, s3_key, tmp.name)
        file_size = Path(tmp.name).stat().st_size
        logger.info("[%s] Downloaded %d MB, uploading to Azure", job_id, file_size // (1024 * 1024))

        blob = storage._service.get_blob_client("raw-videos", azure_blob_path)
        with open(tmp.name, "rb") as fh:
            blob.upload_blob(fh, overwrite=True)

    logger.info("[%s] Transfer complete, starting analysis", job_id)

    # Step 2: Run BeeMonitor analysis
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
        video_blob_path=azure_blob_path,
        detection_mode=detection_mode,
        confidence_threshold=confidence_threshold,
        visualize=visualize,
        two_mode_tracking=two_mode_tracking,
    )

    pipeline.cleanup(job_id)
    model_volume.commit()

    result_dict = result.to_dict()
    result_dict["azure_blob_path"] = azure_blob_path
    result_dict["file_size"] = file_size
    result_dict["execution_seconds"] = round(time.time() - start_time, 1)
    return result_dict


# ── CPU Ingestion ─────────────────────────────────────────────────────

@app.function(
    timeout=3600,
    memory=4096,
    min_containers=1,
)
def ingest_video(
    user_id: str,
    source_type: str,
    source_config: dict,
    upload_id: str | None = None,
) -> dict:
    """Download a video from an external source and store in Azure Blob."""
    import uuid
    from cloud.modal_app.functions.ingest_video import _download_from_source
    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig
    from pathlib import Path
    import tempfile

    if upload_id is None:
        upload_id = str(uuid.uuid4())[:12]

    config = StorageConfig()
    storage = AzureBlobClient(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = _download_from_source(source_type, source_config, tmpdir)
        filename = Path(local_path).name
        file_size = Path(local_path).stat().st_size
        blob_path = f"{user_id}/{upload_id}/{filename}"
        storage.upload_file(config.raw_videos_container, blob_path, local_path)

    return {
        "upload_id": upload_id,
        "azure_blob_path": blob_path,
        "file_size": file_size,
    }


# ── Batch Processing (Modal-native parallelism) ─────────────────────

@app.function(timeout=43200, min_containers=0)  # 12 hour timeout for large batches
def batch_process(jobs: list[dict]) -> list[dict]:
    """Process multiple videos in parallel via starmap().

    Each video gets its own A10G GPU container. Modal handles queuing
    and scaling automatically. For S3 videos, use process_video_from_s3.
    """
    import logging
    logger = logging.getLogger(__name__)

    # Separate S3 vs Azure videos
    s3_jobs = [j for j in jobs if j.get("s3_bucket")]
    azure_jobs = [j for j in jobs if not j.get("s3_bucket")]

    results = []

    # Process Azure videos in parallel
    if azure_jobs:
        logger.info("Processing %d Azure videos in parallel", len(azure_jobs))
        for result in process_video.starmap(
            [
                (
                    j["job_id"], j["user_id"], j["video_blob_path"],
                    j.get("detection_mode", "yolo"),
                    j.get("confidence_threshold", 0.25),
                    j.get("ml_threshold", 0.6),
                    j.get("visualize", True),
                )
                for j in azure_jobs
            ]
        ):
            results.append(result)

    # Process S3 videos in parallel (includes transfer)
    if s3_jobs:
        logger.info("Processing %d S3 videos in parallel (with transfer)", len(s3_jobs))
        for result in process_video_from_s3.starmap(
            [
                (
                    j["job_id"], j["user_id"],
                    j["s3_bucket"], j["s3_key"],
                    j["s3_access_key_id"], j["s3_secret_access_key"],
                    j.get("s3_region", "us-east-1"),
                    j.get("detection_mode", "yolo"),
                    j.get("confidence_threshold", 0.25),
                    j.get("visualize", True),
                )
                for j in s3_jobs
            ]
        ):
            results.append(result)

    logger.info("Batch complete: %d/%d succeeded", len(results), len(jobs))
    return results
