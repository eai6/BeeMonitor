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

@app.function()
@modal.fastapi_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "beemonitor-cloud"}


# ── GPU Processing ────────────────────────────────────────────────────

@app.function(
    gpu="L4",
    timeout=7200,
    retries=1,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    # secrets=[modal.Secret.from_name("azure-storage")],  # Enable when Azure Storage is configured
    memory=8192,
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
    """Run BeeMonitor analysis on a video stored in Azure Blob."""
    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig
    from cloud.wrapper.model_manager import ModelManager
    from cloud.wrapper.pipeline import CloudPipeline

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

    pipeline.cleanup(job_id)
    model_volume.commit()
    return result.to_dict()


# ── CPU Ingestion ─────────────────────────────────────────────────────

@app.function(
    timeout=3600,
    memory=4096,
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


# ── Batch Processing ─────────────────────────────────────────────────

@app.function(timeout=14400)
def batch_process(jobs: list[dict]) -> list[dict]:
    """Process multiple videos in parallel via process_video.starmap()."""
    results = list(process_video.starmap(
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
    ))
    return results
