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


def model_fn(model_dir=None):
    """Build the CloudPipeline once per container.

    Loading is lazy: ``CloudPipeline.__init__`` doesn't load any model
    weights — those come down from the S3 ``models`` bucket on the first
    ``process()`` call via ``ModelManager.ensure_models()``. That means
    cold-start time is dominated by the network pull of the YOLO weights
    (~50 MB) on the very first invocation, not by container boot.
    """
    logger.info("model_fn: building CloudPipeline (model_dir=%s)", model_dir)
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

    required = ("job_id", "user_id", "video_blob_path")
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
    if payload.get("task") == "pre_annotate":
        return _pre_annotate(payload, pipeline)
    if payload.get("task") == "annotate_video":
        return _annotate_video(payload, pipeline)

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
            nest_layout=payload.get("nest_layout"),
            # False = nest/hotel-only fast path (skip tracking + events).
            run_tracking=bool(payload.get("run_tracking", True)),
            # Detector: "sam3" = text-prompt tracking (heavy), else YOLO.
            detector_kind=payload.get("detector_kind", "yolo") or "yolo",
            text_prompt=payload.get("text_prompt", "") or "",
            # ISO recording start (video.recorded_at) — replaces the filename
            # timestamp convention for event timestamps.
            recorded_at=payload.get("recorded_at", "") or "",
        )
    except Exception as exc:
        logger.exception("predict_fn: pipeline failed for job %s", job_id)
        return {
            "status": "failed",
            "job_id": job_id,
            "user_id": user_id,
            "error_message": str(exc),
            "execution_seconds": round(time.time() - started, 2),
        }

    out = result.to_dict()
    out["status"] = "completed"
    out["execution_seconds"] = round(time.time() - started, 2)
    out["device"] = _detect_device()
    return out


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
    # Custom (fine-tuned) detector when requested; else the built-in bee model.
    custom_model_key = payload.get("custom_bee_model_path") or ""
    if custom_model_key:
        model_path = pipeline._models.ensure_custom_model(custom_model_key)
    else:
        model_path = pipeline._models.ensure_models().bee_tracking
    from ultralytics import YOLO
    yolo = YOLO(model_path)

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
            boxes = []
            for r in yolo(frame, conf=conf, verbose=False):
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = (r.names.get(cls_id, f"class_{cls_id}") or "").lower()
                    if name not in class_index:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    boxes.append({
                        "x": round(x1), "y": round(y1),
                        "w": round(x2 - x1), "h": round(y2 - y1),
                        "class": classes[class_index[name]],
                        "class_id": class_index[name],
                        "confidence": round(float(box.conf[0]), 3),
                    })
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
