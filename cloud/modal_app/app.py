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


# ── Smart Pre-Annotation ─────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=3600,
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[modal.Secret.from_name("azure-storage")],
    memory=8192,
    min_containers=0,
)
def pre_annotate_video(
    video_blob_path: str,
    sample_interval: int = 30,  # Sample every N frames (default: 1 per second at 30fps)
    confidence_threshold: float = 0.25,
    max_frames: int = 200,
) -> dict:
    """Run YOLO detection on sampled frames with motion, return detected boxes.

    1. Opens video from Azure
    2. Runs motion detection (background subtraction) to find active frames
    3. Runs YOLO on active frames to detect bees/wasps
    4. Returns frame numbers + bounding boxes for pre-annotation

    Returns:
        {
            "frames": [
                {"frame_number": 150, "boxes": [{"x": 100, "y": 200, "w": 50, "h": 50, "class": "bee", "class_id": 0, "confidence": 0.85}]},
                ...
            ],
            "total_frames_checked": 1000,
            "frames_with_activity": 45,
            "total_detections": 120,
        }
    """
    import cv2
    import tempfile
    import logging
    from pathlib import Path

    logger = logging.getLogger(__name__)

    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig
    from cloud.wrapper.model_manager import ModelManager

    config = StorageConfig()
    storage = AzureBlobClient(config)
    models = ModelManager(
        storage_client=storage,
        storage_config=config,
        local_cache_dir=MODEL_VOLUME_MOUNT,
    )
    model_paths = models.ensure_models()
    model_volume.commit()

    # Download video
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        storage.download_file(config.raw_videos_container, video_blob_path, tmp.name)

        from ultralytics import YOLO
        yolo = YOLO(model_paths.bee_tracking)

        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return {"error": "Could not open video", "frames": []}

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Background subtractor for motion detection
        bg_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=False)

        # Warm up background model on first 30 frames
        for i in range(min(30, total_frames)):
            ret, frame = cap.read()
            if ret:
                bg_sub.apply(frame)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        annotated_frames = []
        frames_checked = 0
        frames_with_activity = 0
        total_detections = 0

        frame_num = 0
        while frame_num < total_frames and len(annotated_frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                frame_num += sample_interval
                continue

            frames_checked += 1

            # Check for motion
            fg_mask = bg_sub.apply(frame)
            motion_pixels = cv2.countNonZero(fg_mask)
            motion_ratio = motion_pixels / (width * height)

            if motion_ratio > 0.001:  # Some motion detected
                frames_with_activity += 1

                # Run YOLO detection
                results = yolo(frame, conf=confidence_threshold, verbose=False)

                boxes = []
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = r.names.get(cls_id, f"class_{cls_id}")

                        boxes.append({
                            "x": round(x1),
                            "y": round(y1),
                            "w": round(x2 - x1),
                            "h": round(y2 - y1),
                            "class": cls_name,
                            "class_id": cls_id,
                            "confidence": round(conf, 3),
                        })

                if boxes:
                    annotated_frames.append({
                        "frame_number": frame_num,
                        "boxes": boxes,
                    })
                    total_detections += len(boxes)

            frame_num += sample_interval

        cap.release()

    logger.info("Pre-annotation: %d frames checked, %d with activity, %d detections, %d frames annotated",
                frames_checked, frames_with_activity, total_detections, len(annotated_frames))

    return {
        "frames": annotated_frames,
        "total_frames_checked": frames_checked,
        "frames_with_activity": frames_with_activity,
        "total_detections": total_detections,
        "video_fps": fps,
        "video_width": width,
        "video_height": height,
    }


# ── YOLO Model Training ──────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=14400,  # 4 hours
    volumes={MODEL_VOLUME_MOUNT: model_volume},
    secrets=[modal.Secret.from_name("azure-storage")],
    memory=16384,
    min_containers=0,
)
def train_yolo_model(
    job_id: str,
    user_id: str,
    base_model: str,
    dataset_yaml_content: str,
    train_images: list[dict],  # [{"filename": "img.jpg", "label": "0 0.5 0.5 0.1 0.1\n..."}]
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
) -> dict:
    """Fine-tune a YOLO model on user annotations.

    Runs ultralytics training on a Modal GPU. Returns metrics and model path.
    """
    import time
    import logging
    import tempfile
    import yaml
    from pathlib import Path

    start_time = time.time()
    logger = logging.getLogger(__name__)

    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig

    config = StorageConfig()
    storage = AzureBlobClient(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dataset_dir = tmpdir / "dataset"

        # Create YOLO dataset structure
        for split in ["train", "val"]:
            (dataset_dir / split / "images").mkdir(parents=True)
            (dataset_dir / split / "labels").mkdir(parents=True)

        # Write images and labels (80/20 split)
        split_idx = int(len(train_images) * 0.8)
        for i, item in enumerate(train_images):
            split = "train" if i < split_idx else "val"
            img_name = item["filename"]
            label_name = img_name.rsplit(".", 1)[0] + ".txt"

            # Download image from Azure if path provided
            if item.get("azure_path"):
                storage.download_file(
                    "raw-videos", item["azure_path"],
                    str(dataset_dir / split / "images" / img_name),
                )
            elif item.get("image_data"):
                import base64
                img_bytes = base64.b64decode(item["image_data"])
                (dataset_dir / split / "images" / img_name).write_bytes(img_bytes)

            # Write label file
            (dataset_dir / split / "labels" / label_name).write_text(item["label"])

        # Write dataset.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_yaml.write_text(dataset_yaml_content)

        logger.info("[%s] Dataset prepared: %d train, %d val",
                    job_id, split_idx, len(train_images) - split_idx)

        # Train
        from ultralytics import YOLO

        model = YOLO(base_model)
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=str(tmpdir / "runs"),
            name="train",
            verbose=True,
        )

        # Extract metrics
        metrics = {}
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            metrics = {
                "mAP50": round(rd.get("metrics/mAP50(B)", 0), 4),
                "mAP50_95": round(rd.get("metrics/mAP50-95(B)", 0), 4),
                "precision": round(rd.get("metrics/precision(B)", 0), 4),
                "recall": round(rd.get("metrics/recall(B)", 0), 4),
            }

        # Find best weights
        best_weights = tmpdir / "runs" / "train" / "weights" / "best.pt"
        if not best_weights.exists():
            # Try alternative path
            for p in (tmpdir / "runs").rglob("best.pt"):
                best_weights = p
                break

        # Upload weights to Azure
        azure_model_path = f"custom/{user_id}/{job_id}/best.pt"
        if best_weights.exists():
            storage.upload_file(
                config.models_container, azure_model_path, str(best_weights),
            )
            logger.info("[%s] Model uploaded to %s", job_id, azure_model_path)
        else:
            logger.warning("[%s] No best.pt found", job_id)
            azure_model_path = ""

    execution_seconds = round(time.time() - start_time, 1)
    model_volume.commit()

    return {
        "job_id": job_id,
        "azure_model_path": azure_model_path,
        "metrics": metrics,
        "execution_seconds": execution_seconds,
        "epochs_completed": epochs,
    }
