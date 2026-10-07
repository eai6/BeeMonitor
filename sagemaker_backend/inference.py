"""
SageMaker inference handler for the BeeMonitor GPU endpoint.

Implements the SageMaker Python inference contract:

    model_fn(model_dir)             -> load CloudPipeline once per container
    input_fn(request_body, ctype)   -> parse {video_storage_key, job_id, ...}
    predict_fn(payload, pipeline)   -> run CloudPipeline.process
    output_fn(prediction, accept)   -> serialize PipelineResult to JSON

Request body (application/json):
    {
        "job_id":   "<unique-id-for-the-analysis-job>",
        "user_id":  "<owner-id>",
        "video_blob_path": "users/7/devices/3/2026/05/.../uuid.mp4",
        "detection_mode": "yolo",            # optional
        "confidence_threshold": 0.25,        # optional
        "visualize": true,                   # optional
        "custom_nest_model_path": "...",     # optional
        "custom_bee_model_path": "...",      # optional
    }

Response body (application/json) is the dict form of ``PipelineResult``,
plus ``status``, ``execution_seconds``, and ``device``.
"""

import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("beemonitor.handler")

JSON_CONTENT_TYPE = "application/json"


# How many CPU threads one invocation may fan out to.
#
# OpenCV and torch both size their pools to the MACHINE by default, which is the
# wrong unit here: the container serves several invocations in one process, so
# each job's decode and resize can claim every core and they contend. On
# 2026-09-09 a batch put CPU at 360% of a 400% box and SageMaker gave up waiting
# for the container — nine of twelve clips died of it, not of anything wrong
# with the clips.
#
# Capacity was cut to one job per instance, which fixed the batch; a single job
# still peaked at 251% of 400%, because Phase 4 gave decode its own thread and
# nothing bounds the pools underneath it. Two per instance is 500% of 400% at
# that rate, so packing stays impossible until this is bounded.
#
# 2 leaves headroom on a 4-vCPU box for the reader thread and — the part that
# actually failed — for gunicorn to answer /ping while a job runs.
CPU_THREADS = int(os.environ.get("BEEMONITOR_CPU_THREADS", "2"))


def _limit_cpu_threads():
    """Bound the OpenCV and torch thread pools, once per container.

    Best-effort: a container that cannot set these should still serve. Neither
    library is required to be present for the module to import — the web image
    has no torch — so both are guarded.
    """
    if CPU_THREADS <= 0:
        logger.info("cpu threads: unbounded (BEEMONITOR_CPU_THREADS=%s)", CPU_THREADS)
        return
    try:
        import cv2
        cv2.setNumThreads(CPU_THREADS)
    except Exception:
        logger.warning("cpu threads: could not cap OpenCV", exc_info=True)
    try:
        import torch
        # Intra-op only. Inter-op governs how many operators run in parallel and
        # is not the thing oversubscribing here.
        torch.set_num_threads(CPU_THREADS)
    except Exception:
        logger.warning("cpu threads: could not cap torch", exc_info=True)
    logger.info("cpu threads: OpenCV and torch capped at %s", CPU_THREADS)


def model_fn(model_dir=None):
    """Build the CloudPipeline once per container.

    Loading is lazy: ``CloudPipeline.__init__`` doesn't load any model
    weights — those come down from the S3 ``models`` bucket on the first
    ``process()`` call via ``ModelManager.ensure_models()``. That means
    cold-start time is dominated by the network pull of the YOLO weights
    (~50 MB) on the very first invocation, not by container boot.
    """
    logger.info("model_fn: building CloudPipeline (model_dir=%s)", model_dir)
    _limit_cpu_threads()
    # Imports here (not at module top) so the SageMaker contract module is
    # importable on the CPU dev box for tests where torch+cuda aren't present.
    from cloud.wrapper.pipeline import CloudPipeline
    pipeline = CloudPipeline()
    logger.info("model_fn: pipeline ready")
    return pipeline


def input_fn(request_body, content_type):
    if content_type != JSON_CONTENT_TYPE:
        raise ValueError(
            f"unsupported content_type {content_type!r}; expected {JSON_CONTENT_TYPE}"
        )
    if isinstance(request_body, (bytes, bytearray)):
        request_body = request_body.decode("utf-8")
    payload = json.loads(request_body)

    # A sample_label batch carries many clips, so it has no single video.
    required = (("batch_id", "clips") if payload.get("task") == "sample_label"
                else ("job_id", "user_id", "video_blob_path"))
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise ValueError(f"missing required keys: {', '.join(missing)}")
    return payload


def predict_fn(payload, pipeline):
    """Run BeeMonitor analysis on one video. Returns a serializable dict.

    Two tasks share this endpoint: the default full analysis, and
    ``task="pre_annotate"`` (sampled-frame YOLO detection that seeds the
    annotation editor). Both reuse the pipeline's models + S3 client.
    """
    # Per-run stage accounting, reset BEFORE the task dispatch so pre-annotation
    # and annotation are measured too — they run the same detectors. The
    # container serves one invocation at a time (async inference), so resetting
    # here makes "this run" explicit rather than relying on a fresh process.
    profiler = _profiler()
    if profiler is not None:
        profiler.reset()

    tasks = {"sample_label": _sample_label_batch, "pre_annotate": _pre_annotate,
             "annotate_video": _annotate_video, "transcode": _transcode,
             "detect_photo": _detect_photo}
    task = tasks.get(payload.get("task"))
    if task is not None:
        started = time.time()
        out = {**task(payload, pipeline), **_timings(profiler)}
        # The instance was busy for the whole handler, and that is what the web
        # app shows as GPU time and bills. A task that doesn't time itself
        # (detect_photo didn't) reported none, so its runs showed 0s and cost 0.
        out.setdefault("execution_seconds", round(time.time() - started, 2))
        return out

    job_id = payload["job_id"]
    user_id = str(payload["user_id"])
    video_blob_path = payload["video_blob_path"]

    started = time.time()
    logger.info("predict_fn: job=%s video=%s", job_id, video_blob_path)

    try:
        result = pipeline.process(
            job_id=job_id,
            user_id=user_id,
            video_blob_path=video_blob_path,
            detection_mode=payload.get("detection_mode", "yolo"),
            confidence_threshold=float(payload.get("confidence_threshold", 0.25)),
            ml_threshold=float(payload.get("ml_threshold", 0.6)),
            visualize=bool(payload.get("visualize", True)),
            two_mode_tracking=bool(payload.get("two_mode_tracking", True)),
            custom_nest_model_path=payload.get("custom_nest_model_path", "") or "",
            custom_bee_model_path=payload.get("custom_bee_model_path", "") or "",
            # Device-supplied hotel ROI + nest tubes (normalized); when both are
            # present the run uses them and the nest model is the backup.
            hotel_roi=payload.get("hotel_roi"),
            # The ROI's traced outline, when the user drew a polygon: tracking is
            # masked to it, so background inside the bounding box is ignored.
            hotel_polygon=payload.get("hotel_polygon"),
            nest_layout=payload.get("nest_layout"),
            # False = nest/hotel-only fast path (skip tracking + events).
            run_tracking=bool(payload.get("run_tracking", True)),
            # Detector: "sam3" = text-prompt tracking (heavy), else YOLO.
            detector_kind=payload.get("detector_kind", "yolo") or "yolo",
            text_prompt=payload.get("text_prompt", "") or "",
            # ISO recording start (video.recorded_at) — replaces the filename
            # timestamp convention for event timestamps.
            recorded_at=payload.get("recorded_at", "") or "",
            # Chunked long videos: one frame range per invocation so no single
            # GPU call exceeds the async platform's 1h cap. Absent = whole video.
            start_frame=int(payload.get("start_frame", 0) or 0),
            end_frame=int(payload["end_frame"]) if payload.get("end_frame") else None,
            # Species / marker identity: voted over every crop of each track
            # after tracking. BeeMachine (fetched only when on) or BioCLIP,
            # constrained to the region's species when the platform sends them.
            identify_species=bool(payload.get("identify_species", False)),
            species_model_key=payload.get("species_model_key", "") or "",
            species_classifier=payload.get("species_classifier", "beemachine") or "beemachine",
            candidate_taxa=payload.get("candidate_taxa") or None,
            identify_markers=bool(payload.get("identify_markers", False)),
            marker_type=payload.get("marker_type", "auto") or "auto",
            # MOT algorithm + settings from the pipeline (memory/43).
            tracker=payload.get("tracker", "beetrack") or "beetrack",
            tracker_params=payload.get("tracker_params") or None,
        )
    except Exception as exc:
        logger.exception("predict_fn: pipeline failed for job %s", job_id)
        return {
            "status": "failed",
            "job_id": job_id,
            "user_id": user_id,
            "error_message": str(exc),
            "execution_seconds": round(time.time() - started, 2),
            **_timings(profiler),
        }

    out = result.to_dict()
    out["status"] = "completed"
    out["execution_seconds"] = round(time.time() - started, 2)
    out.update(_timings(profiler))
    return out


def _profiler():
    """The analysis library's stage profiler, or None if it isn't importable."""
    try:
        from beemonitor.core.profiling import PROFILER
        return PROFILER
    except ImportError:  # pragma: no cover - the image always has it
        return None


def _timings(profiler) -> dict:
    """``gpu_seconds`` + the per-stage breakdown, for cost and for diagnosis.

    ``gpu_seconds`` is wall time around synchronous detector calls, not kernel
    residency: Ultralytics copies results back to the host before returning, so
    the call already blocks on the GPU. It is the honest answer to "how long was
    the GPU step" and the number the web app prices — as opposed to
    ``execution_seconds``, which also covers S3 transfer, decode and encode.
    """
    # ``device`` rides along because it is what the web app prices on: the GPU
    # the container saw, not a tier anyone chose. Reported on every task, so a
    # SAM 3 run on the g5 is billed at the g5 rate rather than the default.
    timings = {"device": _detect_device()}
    if profiler is None:
        return timings
    timings["gpu_seconds"] = profiler.seconds("inference")
    timings["stage_seconds"] = profiler.snapshot()
    return timings


def output_fn(prediction, accept):
    accept = accept or JSON_CONTENT_TYPE
    if accept == "*/*":
        accept = JSON_CONTENT_TYPE
    if accept != JSON_CONTENT_TYPE:
        raise ValueError(f"unsupported accept {accept!r}; expected {JSON_CONTENT_TYPE}")
    return json.dumps(prediction), JSON_CONTENT_TYPE


# ---------------------------------------------------------------------------
# Pre-annotation (AI-assisted) — sampled-frame YOLO detection
# ---------------------------------------------------------------------------

def _transcode(payload, pipeline) -> dict:
    """An uploaded AVI (or other non-MP4) as an MP4 next to it (memory/44).

    H.264 / HEVC video is re-wrapped without re-encoding — lossless and quick.
    Anything else (MJPEG from trail cameras, MPEG-4 Part 2…) is re-encoded
    with libx264 at CRF 18, visually lossless. Audio is dropped: the analysis
    never uses it and AVI audio codecs often don't fit in MP4. The platform
    switches the clip to ``output_key`` once it exists.
    """
    import subprocess
    import tempfile

    src_key, out_key = payload["video_blob_path"], payload["output_key"]
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in" + os.path.splitext(src_key)[1])
        dst = os.path.join(tmp, "out.mp4")
        pipeline._storage.download_file("raw-videos", src_key, src)
        codec = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=codec_name", "-of", "csv=p=0", src],
            capture_output=True, text=True, check=False).stdout.strip().lower()
        if codec in ("h264", "hevc"):
            mode, video = "copy", ["-c:v", "copy"] + (["-tag:v", "hvc1"] if codec == "hevc" else [])
        else:
            mode, video = "encode", ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
                                     "-pix_fmt", "yuv420p"]
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0", *video,
               "-an", "-movflags", "+faststart", dst]
        done = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if done.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
            raise RuntimeError(f"ffmpeg failed ({codec or 'unknown codec'}): {done.stderr[-400:]}")
        pipeline._storage.upload_file("raw-videos", out_key, dst, content_type="video/mp4")
        size = os.path.getsize(dst)
    logger.info("transcode %s -> %s (%s, %s) in %.1fs", src_key, out_key, codec, mode,
                time.time() - started)
    return {"status": "completed", "job_id": payload["job_id"], "output_key": out_key,
            "codec": codec, "mode": mode, "size_bytes": size}


def _read_photo(path):
    """A photo as a BGR array: OpenCV for JPEG/PNG/TIFF, Pillow (+ HEIF) else."""
    import cv2
    import numpy as np

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _detect_photo(payload, pipeline) -> dict:
    """Detect, crop and (optionally) name every insect in one photo (memory/45).

    Large photos are tiled (beemonitor.detection.tiling) so small insects keep
    their pixels. Each detection gets a padded crop in the processed bucket;
    with ``identify_species`` each crop is classified once (BeeMachine or
    BioCLIP, the same classifiers tracks use). Also saves a 1600 px preview
    for the results page, which draws the boxes over it.
    """
    import tempfile
    import cv2
    from beemonitor.detection.tiling import detect_tiled
    from beemonitor.tracking.bee_tracking import padded_box

    started = time.time()
    job_id, user_id = payload["job_id"], str(payload["user_id"])
    classes = [c for c in (payload.get("classes") or ["bee"]) if c]
    class_index = {c.lower(): i for i, c in enumerate(classes)}
    conf = float(payload.get("confidence_threshold", 0.25))
    storage = pipeline._storage
    prefix = f"{user_id}/{job_id}"

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "photo" + os.path.splitext(payload["video_blob_path"])[1].lower())
        storage.download_file("raw-videos", payload["video_blob_path"], src)
        image = _read_photo(src)
        height, width = image.shape[:2]
        detect = _frame_detector(payload, pipeline, classes, class_index, conf)
        boxes, tiles = detect_tiled(image, detect)

        crops, detections = [], []
        for i, b in enumerate(sorted(boxes, key=lambda b: (b["y"], b["x"]))):
            x1, y1, x2, y2 = padded_box((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]), width, height)
            crop = image[y1:y2, x1:x2]
            key = f"{prefix}/photo_crops/{i + 1:04d}.jpg"
            path = os.path.join(tmp, f"crop{i}.jpg")
            cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            storage.upload_file("processed", key, path, content_type="image/jpeg")
            crops.append(crop)
            detections.append({"id": i + 1, "x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"],
                               "class": b.get("class"), "confidence": b.get("confidence"),
                               "crop_key": key})

        scale = min(1.0, 1600.0 / max(width, height))
        preview = cv2.resize(image, (int(width * scale), int(height * scale)),
                             interpolation=cv2.INTER_AREA) if scale < 1 else image
        preview_path, preview_key = os.path.join(tmp, "preview.jpg"), f"{prefix}/photo_preview.jpg"
        cv2.imwrite(preview_path, preview, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        storage.upload_file("processed", preview_key, preview_path, content_type="image/jpeg")

    species_status = None
    if payload.get("identify_species") and crops:
        clf, species_status = pipeline._species_classifier(
            job_id, payload.get("species_classifier", "beemachine"),
            payload.get("species_model_key", ""), payload.get("candidate_taxa"))
        if clf is not None:
            for det, reading in zip(detections, clf.classify_images(crops)):
                if reading:
                    det["species"], det["species_confidence"] = reading[0], round(float(reading[1]), 4)
            species_status["identified"] = sum(1 for d in detections if d.get("species"))
    elif payload.get("identify_species"):
        species_status = {"model": payload.get("species_classifier", "beemachine"),
                          "loaded": False, "error": "no insects detected"}

    logger.info("detect_photo %s: %dx%d, %d tiles, %d detections in %.1fs", job_id, width,
                height, tiles, len(detections), time.time() - started)
    return {"status": "completed", "job_id": job_id,
            "photo": {"width": width, "height": height, "tiles": tiles, "preview_key": preview_key,
                      "detections": detections, "species_status": species_status}}


def _frame_detector(payload, pipeline, classes, class_index, conf):
    """``detect(frame) -> [{x, y, w, h, class, class_id, confidence}]`` for one
    BGR frame: SAM 3 prompted with ``classes`` or YOLO (a custom model when the
    payload names one, else the built-in bee model), keeping only boxes of
    ``classes``. Shared by sampled detection and photo detection.
    """
    # Detector: SAM 3 text-prompt (default for pre-annotation) or YOLO. Both
    # yield the same per-frame box list via ``_detect`` so the sampling/save
    # loop below is detector-agnostic.
    detector_kind = (payload.get("detector_kind") or "yolo").strip().lower()
    if detector_kind == "sam3":
        from beemonitor.detection.sam3_detector import Sam3Detector

        # Prompt SAM 3 with the project's classes (nest tubes stay manual).
        prompt_classes = [c for c in classes if c.lower() != "nest"]
        nms_iou = float(payload.get("nms_iou", 0.5))
        _sam3 = Sam3Detector(
            prompt=",".join(prompt_classes) or "bee",
            conf_threshold=conf,
            iou_threshold=nms_iou if nms_iou > 0 else 0.0,
            max_detections=int(payload.get("max_detections", 100)),
        )

        def _detect(frame):
            out = []
            for det in _sam3.detect(frame):
                name = (det.label or "").lower()
                if name not in class_index:
                    continue
                x1, y1, x2, y2 = det.bbox
                out.append({
                    "x": round(x1), "y": round(y1),
                    "w": round(x2 - x1), "h": round(y2 - y1),
                    "class": classes[class_index[name]],
                    "class_id": class_index[name],
                    "confidence": round(float(det.confidence), 3)
                    if det.confidence is not None else None,
                })
            return out
    else:
        # Custom (fine-tuned) detector when requested; else the built-in bee model.
        custom_model_key = payload.get("custom_bee_model_path") or ""
        if custom_model_key:
            model_path = pipeline._models.ensure_custom_model(custom_model_key)
        else:
            model_path = pipeline._models.ensure_models().bee_tracking
        from ultralytics import YOLO
        yolo = YOLO(model_path)

        def _detect(frame):
            out = []
            for r in yolo(frame, conf=conf, verbose=False):
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = (r.names.get(cls_id, f"class_{cls_id}") or "").lower()
                    if name not in class_index:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    out.append({
                        "x": round(x1), "y": round(y1),
                        "w": round(x2 - x1), "h": round(y2 - y1),
                        "class": classes[class_index[name]],
                        "class_id": class_index[name],
                        "confidence": round(float(box.conf[0]), 3),
                    })
            return out

    return _detect


def _pre_annotate(payload, pipeline) -> dict:
    """Run the bee detector on sampled frames; return boxes to seed annotations.

    Mirrors the legacy Modal ``pre_annotate_video``: sample every Nth frame, run
    the bee/wasp detector, keep only boxes whose class is in the project's
    classes (mapping to the project's class_id), save each hit frame's JPEG to
    the processed bucket, and return the frame list. Nest boxes stay manual.
    """
    import tempfile
    import cv2

    started = time.time()
    video_blob_path = payload["video_blob_path"]
    classes = payload.get("classes") or ["bee", "wasp", "nest"]
    sample_interval = max(1, int(payload.get("sample_interval", 10)))
    max_frames = int(payload.get("max_frames", 300))
    conf = float(payload.get("confidence_threshold", 0.15))
    class_index = {c.lower(): i for i, c in enumerate(classes)}

    storage = pipeline._storage
    _detect = _frame_detector(payload, pipeline, classes, class_index, conf)

    frames_out = []
    checked = total_detections = 0
    width = height = 0
    fps = 30.0
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        storage.download_file("raw-videos", video_blob_path, tmp.name)
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return {"status": "completed", "frames": [], "error": "could not open video",
                    "execution_seconds": round(time.time() - started, 2)}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # An explicit frame_numbers list (editor per-frame pre-annotate) overrides
        # the every-Nth sampling.
        explicit = payload.get("frame_numbers")
        if explicit:
            target_frames = [int(f) for f in explicit if int(f) >= 0][:max_frames]
        else:
            target_frames = list(range(0, total, sample_interval))
        for frame_num in target_frames:
            if len(frames_out) >= max_frames:
                break
            if frame_num >= total:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            checked += 1
            boxes = _detect(frame)
            if boxes:
                frame_blob = f"frames/{video_blob_path.replace('/', '_')}/f{frame_num:06d}.jpg"
                try:
                    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as jt:
                        jt.write(buf.tobytes())
                        jt.flush()
                        storage.upload_file("processed", frame_blob, jt.name)
                except Exception as e:  # noqa: BLE001
                    logger.warning("pre_annotate: frame %d upload failed: %s", frame_num, e)
                    frame_blob = ""
                frames_out.append({
                    "frame_number": frame_num, "boxes": boxes,
                    "frame_image_path": frame_blob,
                })
                total_detections += len(boxes)
        cap.release()

    logger.info("pre_annotate: %s -> %d frames, %d detections (%d checked)",
                video_blob_path, len(frames_out), total_detections, checked)
    return {
        "status": "completed",
        "frames": frames_out,
        "total_frames_checked": checked,
        "frames_with_activity": len(frames_out),
        "total_detections": total_detections,
        "video_fps": fps,
        "video_width": width,
        "video_height": height,
        "execution_seconds": round(time.time() - started, 2),
    }


def _sample_label_batch(payload, pipeline) -> dict:
    """Sample and pre-label a batch of clips in one invocation (memory/38 §4.1).

    Per clip: decode once, motion proposes candidate frames, SAM 3 confirms
    insects for the requested classes, and the top ``max_frames`` frames whose
    detections *moved* are written as JPEGs with their boxes. Clips are
    prepared on a small thread pool (one decode thread each, leaving a core for
    /ping) while the GPU works through whichever clip is ready.

    Payload::

        {"task": "sample_label", "batch_id", "result_bucket", "result_key",
         "classes": [...], "confidence": 0.3, "candidates": 15,
         "min_gap_s": 0.5,
         "clips": [{"task_id", "video_blob_path", "max_frames", "roi", "polygon"}]}

    One clip failing never fails the batch: it comes back with ``error``. The
    whole result is also written to ``result_bucket/result_key`` so the web app
    can find it even if it lost the async output location.
    """
    import io
    import tempfile
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import cv2

    from beemonitor.detection.sam3_detector import Sam3Detector
    from beemonitor.processing import sample_label as sl

    started = time.time()
    storage = pipeline._storage
    classes = [c for c in (payload.get("classes") or ["bee"]) if c]
    candidates = int(payload.get("candidates", 15))
    min_gap_s = float(payload.get("min_gap_s", 0.5))
    detector = Sam3Detector(prompt=",".join(classes),
                            conf_threshold=float(payload.get("confidence", 0.3)),
                            iou_threshold=0.5, max_detections=50)
    wanted = {c.lower(): c for c in classes}
    gpu_lock = threading.Lock()

    def detect_fn(frames):
        with gpu_lock:                       # one model on one GPU
            per_frame = detector.detect_many(frames)
        out = []
        for dets in per_frame:
            boxes = []
            for d in dets:
                name = wanted.get((d.label or "").lower())
                if not name:
                    continue
                x1, y1, x2, y2 = d.bbox
                boxes.append({"x": round(x1), "y": round(y1), "w": round(x2 - x1),
                              "h": round(y2 - y1), "class": name,
                              "confidence": round(float(d.confidence), 3)
                              if d.confidence is not None else None})
            out.append(boxes)
        return out

    def one_clip(clip):
        t0 = time.time()
        blob = clip["video_blob_path"]
        res = {"task_id": clip.get("task_id"), "frames": []}
        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                storage.download_file("raw-videos", blob, tmp.name)
                t_dl = time.time()
                out = sl.sample_label_clip(
                    tmp.name, detect_fn, max_frames=int(clip.get("max_frames", 20)),
                    candidates=candidates, min_gap_s=min_gap_s,
                    roi=clip.get("roi"), polygon=clip.get("polygon"))
            t_scan = time.time()
            prefix = f"frames/{blob.replace('/', '_')}"
            for pick in out["picks"]:
                ok, buf = cv2.imencode(".jpg", pick["frame"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    continue
                key = f"{prefix}/f{pick['n']:06d}.jpg"
                storage.upload_stream("processed", key, io.BytesIO(buf.tobytes()),
                                      content_type="image/jpeg")
                h, w = pick["frame"].shape[:2]
                res["frames"].append({"n": pick["n"], "key": key, "w": w, "h": h,
                                      "boxes": pick["boxes"]})
            res.update(motion=out["motion"], candidates=out["candidates"],
                       total_frames=out["frames"], fps=out["fps"],
                       seconds={"download": round(t_dl - t0, 2),
                                "scan_and_detect": round(t_scan - t_dl, 2),
                                "upload": round(time.time() - t_scan, 2)})
            logger.info("sample_label: %s -> %d frames (%d candidates, %d decoded) in %.1fs",
                        blob, len(res["frames"]), out["candidates"], out["frames"],
                        time.time() - t0)
        except Exception as exc:  # one clip never fails the batch
            logger.exception("sample_label: %s failed", blob)
            res["error"] = f"{type(exc).__name__}: {exc}"[:500]
        return res

    # Three clips at a time: each is one decode thread plus its share of the GPU.
    workers = max(1, min(3, int(os.environ.get("BEEMONITOR_SAMPLE_WORKERS", "3"))))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="sample-label") as pool:
        results = list(pool.map(one_clip, payload.get("clips") or []))

    body = {"status": "completed", "batch_id": payload.get("batch_id"),
            "clips": results, "execution_seconds": round(time.time() - started, 2)}
    bucket, key = payload.get("result_bucket"), payload.get("result_key")
    if bucket and key:
        try:
            import boto3
            boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1")).put_object(
                Bucket=bucket, Key=key, Body=json.dumps(body).encode("utf-8"),
                ContentType="application/json")
        except Exception:
            logger.exception("sample_label: could not write %s/%s", bucket, key)
    return body


def _annotate_video(payload, pipeline) -> dict:
    """Render an annotated video AFTER analysis, streaming from the saved
    tracking CSV — never holds more than one frame in memory, so it can't OOM
    the way inline visualization did, and it runs as its own invocation with
    its own time budget.

    Payload: job_id, user_id, video_blob_path, tracking_csv_path,
    nest_bboxes (optional {id: [x1,y1,x2,y2]}), hotel_bbox (optional).
    """
    import csv
    import tempfile
    import cv2

    started = time.time()
    job_id = payload["job_id"]
    user_id = str(payload["user_id"])
    video_blob_path = payload["video_blob_path"]
    tracking_csv_path = payload.get("tracking_csv_path", "")
    storage = pipeline._storage

    if not tracking_csv_path:
        return {"status": "failed", "job_id": job_id,
                "error_message": "annotate_video needs tracking_csv_path"}

    # Tracks grouped by frame -> [(x1,y1,x2,y2,track_id,taxon), ...]
    import io
    buf = io.BytesIO()
    storage.download_to_stream("processed", tracking_csv_path, buf)
    reader = csv.DictReader(io.StringIO(buf.getvalue().decode("utf-8", "replace")))
    by_frame = {}
    for row in reader:
        try:
            fn = int(float(row["frame"]))
            box = (int(float(row["x1"])), int(float(row["y1"])),
                   int(float(row["x2"])), int(float(row["y2"])))
        except (KeyError, ValueError, TypeError):
            continue
        by_frame.setdefault(fn, []).append(
            (box, str(row.get("track_id", "")), row.get("taxon", "") or ""))

    nests = payload.get("nest_bboxes") or {}
    hotel = payload.get("hotel_bbox")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tin:
        storage.download_file("raw-videos", video_blob_path, tin.name)
        cap = cv2.VideoCapture(tin.name)
        if not cap.isOpened():
            return {"status": "failed", "job_id": job_id,
                    "error_message": "could not open source video"}
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_local = tin.name.replace(".mp4", "_annotated.mp4")
        writer = cv2.VideoWriter(out_local, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
        fn = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if hotel and len(hotel) == 4:
                x1, y1, x2, y2 = [int(v) for v in hotel]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 160, 255), 2)
            for nid, bb in nests.items():
                if bb and len(bb) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bb]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 1)
            for box, tid, taxon in by_frame.get(fn, ()):  # one frame's tracks only
                x1, y1, x2, y2 = box
                color = colors[(hash(tid) if tid else 0) % len(colors)]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{taxon}:{tid}" if taxon else f"T:{tid}"
                cv2.putText(frame, label, (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            writer.write(frame)
            fn += 1
        cap.release()
        writer.release()

        blob = f"{user_id}/{job_id}/annotated.mp4"
        storage.upload_file("processed", blob, out_local, content_type="video/mp4")
        import os
        try:
            os.unlink(out_local)
        except OSError:
            pass

    logger.info("annotate_video: %d frames -> %s", fn, blob)
    return {
        "status": "completed",
        "job_id": job_id,
        "annotated_video_path": blob,
        "frames": fn,
        "execution_seconds": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Report which device the pipeline ran on (for response telemetry)."""
    try:
        import torch
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.get_device_name(0)}"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"
