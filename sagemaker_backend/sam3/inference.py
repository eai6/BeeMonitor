"""SAM 3 auto-labeler — SageMaker BYOC handler.

Loads facebook/sam3 (Promptable Concept Segmentation) and labels frames by text
prompt (e.g. "bee") to bootstrap YOLO training data for new-domain footage the
current detector misses. Two request modes (POST /invocations, application/json):

1. pre_annotate (the labeler) — drop-in compatible with the tracking endpoint's
   task="pre_annotate", so the web pre-annotation flow only swaps the endpoint:
     {"task": "pre_annotate", "video_blob_path": "...",
      "classes": ["bee","wasp","nest"], "sample_interval": 10,
      "max_frames": 300, "confidence_threshold": 0.3}
   Samples every Nth frame, runs SAM 3 with each non-"nest" class as the text
   prompt, uploads hit frames' JPEGs to the processed bucket, and returns the same
   {frames:[{frame_number, boxes:[{x,y,w,h,class,class_id,confidence}],
   frame_image_path}], video_width, video_height, ...} shape. Nest tubes stay manual.

2. images (ad-hoc) — {"images": ["<b64>",...], "prompt": "bee", "threshold": 0.3}
   -> {"prompt": ..., "results": [[{"bbox":[x1,y1,x2,y2], "score":...}], ...]}
"""

import base64
import io
import json
import logging
import os
import time

from PIL import Image

logger = logging.getLogger("beemonitor.sam3")

JSON = "application/json"
_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")


def model_fn(model_dir=None):
    import torch
    from transformers import Sam3Model, Sam3Processor

    device = os.environ.get("SAM3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    token = os.environ.get("HF_TOKEN") or None  # gated model; None when baked/offline
    processor = Sam3Processor.from_pretrained(_MODEL_ID, token=token)
    model = Sam3Model.from_pretrained(_MODEL_ID, token=token).to(device).eval()
    logger.info("SAM 3 (%s) loaded on %s", _MODEL_ID, device)
    return {"model": model, "processor": processor, "device": device}


def _segment(ctx, pil_image, prompt, threshold, max_detections=100):
    """Run SAM 3 on one PIL RGB image → [(x1, y1, x2, y2, score), ...] in pixels.

    Mirrors EcoMorph's verified HuggingFace SAM 3 call.
    """
    import torch

    model, processor, device = ctx["model"], ctx["processor"], ctx["device"]
    proc = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**proc)
    post = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=0.5,
        target_sizes=proc.get("original_sizes").tolist(),
    )[0]

    boxes = post.get("boxes")
    scores = post.get("scores")
    out = []
    if boxes is not None:
        box_list = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
        score_list = (scores.tolist() if scores is not None and hasattr(scores, "tolist")
                      else [None] * len(box_list))
        for b, s in list(zip(box_list, score_list))[:max_detections]:
            x1, y1, x2, y2 = [float(v) for v in b]
            out.append((x1, y1, x2, y2, float(s) if s is not None else None))
    return out


def input_fn(body, content_type=JSON):
    return json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)


def predict_fn(payload, ctx):
    if payload.get("task") == "pre_annotate":
        return _pre_annotate(payload, ctx)
    return _segment_images(payload, ctx)


def _segment_images(payload, ctx):
    images_b64 = payload.get("images")
    if not images_b64 and payload.get("image"):
        images_b64 = [payload["image"]]
    if not images_b64:
        raise ValueError("no images provided (expected 'images': [b64,...] or 'image': b64)")
    prompt = payload.get("prompt", "bee")
    thr = float(payload.get("threshold", 0.3))
    maxd = int(payload.get("max_detections", 100))
    results = []
    for b in images_b64:
        img = Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB")
        dets = [{"bbox": [x1, y1, x2, y2], "score": sc}
                for (x1, y1, x2, y2, sc) in _segment(ctx, img, prompt, thr, maxd)]
        results.append(dets)
    return {"prompt": prompt, "results": results}


def _pre_annotate(payload, ctx):
    """Sample frames + SAM 3-label them; mirrors the YOLO worker's task=pre_annotate."""
    import tempfile
    from urllib.parse import urlparse

    import boto3
    import cv2

    started = time.time()
    region = os.environ.get("AWS_REGION")
    raw_bucket = os.environ["AWS_S3_BUCKET_RAW_VIDEOS"]
    proc_bucket = os.environ["AWS_S3_BUCKET_PROCESSED"]
    s3 = boto3.client("s3", region_name=region)

    video_blob_path = payload["video_blob_path"]
    classes = payload.get("classes") or ["bee", "wasp", "nest"]
    sample_interval = max(1, int(payload.get("sample_interval", 10)))
    max_frames = int(payload.get("max_frames", 300))
    conf = float(payload.get("confidence_threshold", 0.3))
    # SAM 3 prompts = the project's classes minus "nest" (nest tubes stay manual).
    prompt_classes = [(i, c) for i, c in enumerate(classes) if c.lower() != "nest"]

    frames_out = []
    checked = total_detections = 0
    width = height = 0
    fps = 30.0
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        if video_blob_path.startswith("s3://"):
            u = urlparse(video_blob_path)
            s3.download_file(u.netloc, u.path.lstrip("/"), tmp.name)
        else:
            s3.download_file(raw_bucket, video_blob_path, tmp.name)
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return {"status": "completed", "frames": [], "error": "could not open video",
                    "execution_seconds": round(time.time() - started, 2)}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_num = 0
        while frame_num < total and len(frames_out) < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                frame_num += sample_interval
                continue
            checked += 1
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = []
            for cls_id, cls_name in prompt_classes:
                for (x1, y1, x2, y2, sc) in _segment(ctx, pil, cls_name, conf):
                    boxes.append({
                        "x": round(x1), "y": round(y1),
                        "w": round(x2 - x1), "h": round(y2 - y1),
                        "class": classes[cls_id], "class_id": cls_id,
                        "confidence": round(sc, 3) if sc is not None else None,
                    })
            if boxes:
                frame_blob = f"frames/{video_blob_path.replace('/', '_')}/f{frame_num:06d}.jpg"
                try:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as jt:
                        jt.write(buf.tobytes())
                        jt.flush()
                        s3.upload_file(jt.name, proc_bucket, frame_blob)
                except Exception as e:  # noqa: BLE001
                    logger.warning("sam3 pre_annotate: frame %d upload failed: %s", frame_num, e)
                    frame_blob = ""
                frames_out.append({"frame_number": frame_num, "boxes": boxes,
                                   "frame_image_path": frame_blob})
                total_detections += len(boxes)
            frame_num += sample_interval
        cap.release()

    logger.info("sam3 pre_annotate: %s -> %d frames, %d detections (%d checked)",
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


def output_fn(prediction, accept=JSON):
    return json.dumps(prediction), JSON
