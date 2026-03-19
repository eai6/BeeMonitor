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
    custom_nest_model_path: str = "",
    custom_bee_model_path: str = "",
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
        custom_nest_model_path=custom_nest_model_path,
        custom_bee_model_path=custom_bee_model_path,
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
    custom_nest_model_path: str = "",
    custom_bee_model_path: str = "",
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
        custom_nest_model_path=custom_nest_model_path,
        custom_bee_model_path=custom_bee_model_path,
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
    sample_interval: int = 10,  # Sample every N frames (~3 per second at 30fps)
    confidence_threshold: float = 0.15,  # Low threshold to catch more bees
    max_frames: int = 300,
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

        annotated_frames = []
        frames_checked = 0
        frames_with_detections = 0
        total_detections = 0

        frame_num = 0
        while frame_num < total_frames and len(annotated_frames) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                frame_num += sample_interval
                continue

            frames_checked += 1

            # Run YOLO on every sampled frame (no motion filter)
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
                frames_with_detections += 1

                # Save frame JPEG to Azure for fast serving later
                frame_blob_path = f"frames/{video_blob_path.replace('/', '_')}/f{frame_num:06d}.jpg"
                try:
                    _, jpg_buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    import tempfile as _tf
                    with _tf.NamedTemporaryFile(suffix=".jpg", delete=True) as jpg_tmp:
                        jpg_tmp.write(jpg_buf.tobytes())
                        jpg_tmp.flush()
                        storage.upload_file("processed", frame_blob_path, jpg_tmp.name)
                except Exception as upload_err:
                    logger.warning("Failed to upload frame %d: %s", frame_num, upload_err)
                    frame_blob_path = ""

                annotated_frames.append({
                    "frame_number": frame_num,
                    "boxes": boxes,
                    "frame_image_path": frame_blob_path,
                })
                total_detections += len(boxes)

            frame_num += sample_interval

        cap.release()

    logger.info("Pre-annotation: %d frames checked, %d with detections, %d total detections, %d frames saved",
                frames_checked, frames_with_detections, total_detections, len(annotated_frames))

    return {
        "frames": annotated_frames,
        "total_frames_checked": frames_checked,
        "frames_with_activity": frames_with_detections,
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
    train_images: list[dict] | None = None,
    video_annotations: list[dict] | None = None,
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 16,
) -> dict:
    """Fine-tune a YOLO model on user annotations.

    Runs ultralytics training on a Modal GPU. Returns metrics and model path.
    Supports two modes:
      - train_images: pre-extracted images (azure_path or base64 image_data)
      - video_annotations: extract frames from videos on the GPU side
    """
    import time
    import logging
    import tempfile
    import os
    from pathlib import Path

    start_time = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("train_yolo")
    logger.setLevel(logging.DEBUG)

    logger.info("=" * 70)
    logger.info("[%s] TRAINING JOB STARTED", job_id)
    logger.info("[%s] Parameters: user=%s base_model=%s epochs=%d imgsz=%d batch=%d",
                job_id, user_id, base_model, epochs, imgsz, batch_size)
    logger.info("[%s] video_annotations: %d video(s)", job_id, len(video_annotations) if video_annotations else 0)
    logger.info("[%s] train_images: %d image(s)", job_id, len(train_images) if train_images else 0)
    logger.info("=" * 70)

    # Log GPU info
    try:
        import torch
        logger.info("[%s] PyTorch version: %s", job_id, torch.__version__)
        logger.info("[%s] CUDA available: %s", job_id, torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("[%s] CUDA device: %s", job_id, torch.cuda.get_device_name(0))
            logger.info("[%s] CUDA memory: %.1f GB total, %.1f GB free",
                        job_id,
                        torch.cuda.get_device_properties(0).total_mem / 1e9,
                        torch.cuda.mem_get_info()[0] / 1e9)
    except Exception as e:
        logger.warning("[%s] Could not get GPU info: %s", job_id, e)

    logger.info("[%s] Initializing Azure storage client...", job_id)
    from cloud.storage.azure_client import AzureBlobClient
    from cloud.storage.config import StorageConfig

    config = StorageConfig()
    storage = AzureBlobClient(config)
    logger.info("[%s] Azure storage client ready — models_container=%s", job_id, config.models_container)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        dataset_dir = tmpdir / "dataset"
        logger.info("[%s] Working directory: %s", job_id, tmpdir)

        # Create YOLO dataset structure
        for split in ["train", "val"]:
            (dataset_dir / split / "images").mkdir(parents=True)
            (dataset_dir / split / "labels").mkdir(parents=True)
        logger.info("[%s] Dataset directory structure created at %s", job_id, dataset_dir)

        # Collect all (filename, label) pairs from both modes
        all_items = []

        # Mode 1: video_annotations — download videos and extract frames
        if video_annotations:
            import cv2
            logger.info("[%s] MODE: video_annotations — processing %d video(s)", job_id, len(video_annotations))

            for vi, va in enumerate(video_annotations):
                video_path = va["video_blob_path"]
                frames = va["frames"]
                logger.info("[%s] Video %d/%d: blob_path='%s', %d frame(s) to extract",
                            job_id, vi + 1, len(video_annotations), video_path, len(frames))

                if not frames:
                    logger.warning("[%s] Video %d — no frames, skipping", job_id, vi + 1)
                    continue

                # Download video to temp file
                tmp_video = tmpdir / f"video_{hash(video_path) & 0xFFFFFFFF}.mp4"
                logger.info("[%s] Downloading video to %s ...", job_id, tmp_video)
                dl_start = time.time()
                try:
                    storage.download_file("raw-videos", video_path, str(tmp_video))
                    dl_time = time.time() - dl_start
                    file_size = tmp_video.stat().st_size
                    logger.info("[%s] Video downloaded: %.1f MB in %.1fs (%.1f MB/s)",
                                job_id, file_size / 1e6, dl_time,
                                (file_size / 1e6) / max(dl_time, 0.01))
                except Exception as e:
                    logger.error("[%s] FAILED to download video '%s': %s: %s",
                                 job_id, video_path, type(e).__name__, e)
                    continue

                cap = cv2.VideoCapture(str(tmp_video))
                if not cap.isOpened():
                    logger.error("[%s] FAILED to open video with cv2: %s", job_id, tmp_video)
                    tmp_video.unlink(missing_ok=True)
                    continue

                total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                logger.info("[%s] Video opened: %d total frames, %.1f fps, %dx%d",
                            job_id, total_video_frames, video_fps, video_w, video_h)

                extracted = 0
                failed = 0
                try:
                    for fi, fr in enumerate(frames):
                        frame_num = fr["frame_number"]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame_img = cap.read()
                        if ret:
                            all_items.append({
                                "filename": fr["filename"],
                                "label": fr["label"],
                                "frame_img": frame_img,
                            })
                            extracted += 1
                            if (fi + 1) % 50 == 0 or fi == len(frames) - 1:
                                logger.info("[%s] Extracted %d/%d frames from video %d",
                                            job_id, extracted, len(frames), vi + 1)
                        else:
                            failed += 1
                            logger.warning("[%s] Failed to read frame %d (total_frames=%d) from video %d",
                                           job_id, frame_num, total_video_frames, vi + 1)
                finally:
                    cap.release()
                    tmp_video.unlink(missing_ok=True)
                    logger.info("[%s] Video %d done: %d extracted, %d failed, temp file cleaned up",
                                job_id, vi + 1, extracted, failed)

        # Mode 2: pre-extracted images
        if train_images:
            logger.info("[%s] MODE: train_images — %d pre-extracted image(s)", job_id, len(train_images))
            for item in train_images:
                all_items.append(item)
                source = "azure_path" if item.get("azure_path") else ("image_data" if item.get("image_data") else "unknown")
                logger.debug("[%s] Image: %s source=%s", job_id, item.get("filename", "?"), source)

        logger.info("[%s] Total training images collected: %d", job_id, len(all_items))

        if not all_items:
            logger.error("[%s] NO TRAINING IMAGES — aborting", job_id)
            return {
                "job_id": job_id,
                "azure_model_path": "",
                "metrics": {},
                "execution_seconds": round(time.time() - start_time, 1),
                "epochs_completed": 0,
                "error": "No training images found",
            }

        # Write images and labels (80/20 split)
        split_idx = int(len(all_items) * 0.8)
        logger.info("[%s] Splitting dataset: %d train, %d val (80/20)",
                    job_id, split_idx, len(all_items) - split_idx)

        write_start = time.time()
        for i, item in enumerate(all_items):
            split = "train" if i < split_idx else "val"
            img_name = item["filename"]
            label_name = img_name.rsplit(".", 1)[0] + ".txt"

            # Write image from one of the supported sources
            img_path = dataset_dir / split / "images" / img_name
            if item.get("frame_img") is not None:
                import cv2
                cv2.imwrite(str(img_path), item["frame_img"])
            elif item.get("azure_path"):
                storage.download_file("raw-videos", item["azure_path"], str(img_path))
            elif item.get("image_data"):
                import base64
                img_bytes = base64.b64decode(item["image_data"])
                img_path.write_bytes(img_bytes)

            # Write label file
            label_path = dataset_dir / split / "labels" / label_name
            label_content = item["label"]
            label_path.write_text(label_content)
            label_lines = len(label_content.strip().split("\n")) if label_content.strip() else 0

            if (i + 1) % 50 == 0 or i == len(all_items) - 1:
                logger.info("[%s] Written %d/%d images+labels to disk", job_id, i + 1, len(all_items))
            logger.debug("[%s] %s/%s — %s (%d bbox lines)", job_id, split, img_name, label_name, label_lines)

        write_time = time.time() - write_start
        logger.info("[%s] Dataset files written in %.1fs", job_id, write_time)

        # Write dataset.yaml — override path with absolute dataset_dir
        data_yaml = dataset_dir / "data.yaml"
        fixed_yaml = dataset_yaml_content.replace("path: .", f"path: {dataset_dir}")
        data_yaml.write_text(fixed_yaml)
        logger.info("[%s] dataset.yaml written:\n%s", job_id, fixed_yaml)

        # Verify dataset structure
        train_imgs = list((dataset_dir / "train" / "images").glob("*"))
        val_imgs = list((dataset_dir / "val" / "images").glob("*"))
        train_labels = list((dataset_dir / "train" / "labels").glob("*"))
        val_labels = list((dataset_dir / "val" / "labels").glob("*"))
        logger.info("[%s] Dataset verification: train=%d imgs/%d labels, val=%d imgs/%d labels",
                    job_id, len(train_imgs), len(train_labels), len(val_imgs), len(val_labels))

        if not train_imgs:
            logger.error("[%s] NO TRAINING IMAGES ON DISK — something went wrong during write", job_id)

        # Log disk usage
        total_size = sum(f.stat().st_size for f in dataset_dir.rglob("*") if f.is_file())
        logger.info("[%s] Total dataset size on disk: %.1f MB", job_id, total_size / 1e6)

        # Train
        logger.info("[%s] " + "=" * 50, job_id)
        logger.info("[%s] STARTING YOLO TRAINING", job_id)
        logger.info("[%s] base_model=%s epochs=%d imgsz=%d batch=%d",
                    job_id, base_model, epochs, imgsz, batch_size)
        logger.info("[%s] data=%s", job_id, data_yaml)
        logger.info("[%s] " + "=" * 50, job_id)

        train_start = time.time()
        from ultralytics import YOLO

        logger.info("[%s] Loading YOLO model '%s'...", job_id, base_model)
        model = YOLO(base_model)
        logger.info("[%s] YOLO model loaded — starting training...", job_id)

        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=str(tmpdir / "runs"),
            name="train",
            verbose=True,
        )

        train_time = time.time() - train_start
        logger.info("[%s] YOLO training finished in %.1fs (%.1f min)", job_id, train_time, train_time / 60)

        # Extract metrics
        metrics = {}
        if hasattr(results, "results_dict"):
            rd = results.results_dict
            logger.info("[%s] Raw results_dict keys: %s", job_id, list(rd.keys()))
            logger.info("[%s] Raw results_dict values: %s",
                        job_id, {k: round(v, 4) if isinstance(v, float) else v for k, v in rd.items()})
            metrics = {
                "mAP50": round(rd.get("metrics/mAP50(B)", 0), 4),
                "mAP50_95": round(rd.get("metrics/mAP50-95(B)", 0), 4),
                "precision": round(rd.get("metrics/precision(B)", 0), 4),
                "recall": round(rd.get("metrics/recall(B)", 0), 4),
            }
            logger.info("[%s] Extracted metrics: %s", job_id, metrics)
        else:
            logger.warning("[%s] No results_dict on training results — type=%s, dir=%s",
                           job_id, type(results).__name__,
                           [a for a in dir(results) if not a.startswith("_")][:20])

        # Find best weights
        best_weights = tmpdir / "runs" / "train" / "weights" / "best.pt"
        logger.info("[%s] Looking for best.pt at %s — exists=%s", job_id, best_weights, best_weights.exists())
        if not best_weights.exists():
            logger.warning("[%s] best.pt not at expected path, searching recursively...", job_id)
            found_pts = list((tmpdir / "runs").rglob("*.pt"))
            logger.info("[%s] Found .pt files: %s", job_id, [str(p) for p in found_pts])
            for p in (tmpdir / "runs").rglob("best.pt"):
                best_weights = p
                logger.info("[%s] Found best.pt at: %s", job_id, best_weights)
                break

        # Upload weights to Azure
        azure_model_path = f"custom/{user_id}/{job_id}/best.pt"
        if best_weights.exists():
            weight_size = best_weights.stat().st_size
            logger.info("[%s] Uploading best.pt (%.1f MB) to Azure: %s/%s",
                        job_id, weight_size / 1e6, config.models_container, azure_model_path)
            upload_start = time.time()
            storage.upload_file(
                config.models_container, azure_model_path, str(best_weights),
            )
            upload_time = time.time() - upload_start
            logger.info("[%s] Model uploaded in %.1fs (%.1f MB/s)",
                        job_id, upload_time, (weight_size / 1e6) / max(upload_time, 0.01))
        else:
            logger.error("[%s] NO best.pt FOUND — training may have failed silently", job_id)
            azure_model_path = ""

    execution_seconds = round(time.time() - start_time, 1)
    logger.info("[%s] Committing model volume...", job_id)
    model_volume.commit()

    logger.info("=" * 70)
    logger.info("[%s] TRAINING JOB COMPLETE", job_id)
    logger.info("[%s] Total time: %.1fs (%.1f min)", job_id, execution_seconds, execution_seconds / 60)
    logger.info("[%s] Metrics: %s", job_id, metrics)
    logger.info("[%s] Model path: %s", job_id, azure_model_path or "NONE")
    logger.info("=" * 70)

    return {
        "job_id": job_id,
        "azure_model_path": azure_model_path,
        "metrics": metrics,
        "execution_seconds": execution_seconds,
        "epochs_completed": epochs,
    }
