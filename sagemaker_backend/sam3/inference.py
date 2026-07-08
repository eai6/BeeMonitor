"""SAM 3 auto-labeler (+ DINOv3 active selection) — SageMaker BYOC handler.

Loads facebook/sam3 (Promptable Concept Segmentation) to label frames by text
prompt ("bee"), and facebook/dinov3 to embed frames for active selection — so the
labeler spends its effort on a diverse, representative subset instead of every Nth
(near-duplicate) frame. Both bootstrap YOLO training data for new-domain footage
the current detector misses.

Request modes (POST /invocations, application/json):

1. pre_annotate (the labeler) — drop-in compatible with the tracking endpoint's
   task="pre_annotate":
     {"task": "pre_annotate", "video_blob_path": "...",
      "classes": ["bee","wasp","nest"], "sample_interval": 10, "max_frames": 300,
      "confidence_threshold": 0.3,
      "selection": "uniform" | "diverse", "candidate_cap": 400}
   selection="diverse" (DINOv3): sample up to candidate_cap frames, embed them, pick
   max_frames by farthest-point sampling, then SAM 3-label only those. Returns the
   same {frames:[{frame_number, boxes:[{x,y,w,h,class,class_id,confidence}],
   frame_image_path}], video_width/height, ...} shape. Nest tubes stay manual.

2. embed (DINOv3) — {"task": "embed", "images": ["<b64>",...]} (or a video ref +
   sampling) -> {"dim": D, "embeddings": [[...], ...]} (L2-normalised CLS tokens).
   Used by the drift trigger / active-learning selection outside the container.

3. images (ad-hoc SAM 3) — {"images": ["<b64>",...], "prompt": "bee", "threshold": 0.3}
   -> {"prompt": ..., "results": [[{"bbox":[x1,y1,x2,y2], "score":...}], ...]}
"""

import base64
import io
import json
import logging
import os
import time

import numpy as np
from PIL import Image

logger = logging.getLogger("beemonitor.sam3")

JSON = "application/json"
_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
# DINOv2-base: ungated, instant. (DINOv3 is manual-review gated — swap the id back
# here + rebuild once Meta grants access; the AutoModel/AutoImageProcessor API is identical.)
_DINO_ID = os.environ.get("DINO_MODEL_ID", "facebook/dinov2-base")
_EMBED_BATCH = int(os.environ.get("DINOV3_BATCH", "16"))


def model_fn(model_dir=None):
    import torch
    from transformers import AutoImageProcessor, AutoModel, Sam3Model, Sam3Processor

    device = os.environ.get("SAM3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    token = os.environ.get("HF_TOKEN") or None  # gated models; None when baked/offline

    processor = Sam3Processor.from_pretrained(_MODEL_ID, token=token)
    model = Sam3Model.from_pretrained(_MODEL_ID, token=token).to(device).eval()
    dino_processor = AutoImageProcessor.from_pretrained(_DINO_ID, token=token)
    dino_model = AutoModel.from_pretrained(_DINO_ID, token=token).to(device).eval()
    logger.info("SAM 3 (%s) + DINO (%s) loaded on %s", _MODEL_ID, _DINO_ID, device)
    return {
        "model": model, "processor": processor,
        "dino_model": dino_model, "dino_processor": dino_processor,
        "device": device,
    }


# --- SAM 3 ------------------------------------------------------------------

def _segment(ctx, pil_image, prompt, threshold, max_detections=100):
    """Run SAM 3 on one PIL RGB image → [(x1, y1, x2, y2, score), ...] in pixels."""
    import torch

    model, processor, device = ctx["model"], ctx["processor"], ctx["device"]
    proc = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**proc)
    post = processor.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
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


# --- DINOv3 -----------------------------------------------------------------

def _embed(ctx, pil_images):
    """L2-normalised DINOv3 CLS-token embeddings for a batch of PIL images → (N, D)."""
    import torch

    proc = ctx["dino_processor"](images=pil_images, return_tensors="pt").to(ctx["device"])
    with torch.no_grad():
        out = ctx["dino_model"](**proc)
    pooled = getattr(out, "pooler_output", None)
    emb = pooled if pooled is not None else out.last_hidden_state[:, 0]
    emb = emb.float().cpu().numpy()
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return emb / norms


def _iou(a, b):
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, iou_thr):
    """Greedy non-max suppression across ALL boxes (class-agnostic) — collapses the
    duplicate bee/wasp boxes the per-prompt passes produce on the same object; keeps
    the higher-confidence label."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].get("confidence") or 0.0,
                   reverse=True)
    keep = []
    for i in order:
        if all(_iou(boxes[i], boxes[j]) <= iou_thr for j in keep):
            keep.append(i)
    return [boxes[i] for i in keep]


def _farthest_point_select(emb, k):
    """Greedy farthest-point sampling → k indices maximising min pairwise distance.

    Deterministic: seeds from the point nearest the centroid.
    """
    n = emb.shape[0]
    if k >= n:
        return list(range(n))
    centroid = emb.mean(axis=0, keepdims=True)
    start = int(np.argmin(((emb - centroid) ** 2).sum(axis=1)))
    selected = [start]
    dist = ((emb - emb[start]) ** 2).sum(axis=1)
    for _ in range(k - 1):
        idx = int(np.argmax(dist))
        selected.append(idx)
        dist = np.minimum(dist, ((emb - emb[idx]) ** 2).sum(axis=1))
    return selected


# --- I/O contract -----------------------------------------------------------

def input_fn(body, content_type=JSON):
    return json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)


def predict_fn(payload, ctx):
    task = payload.get("task")
    if task == "pre_annotate":
        return _pre_annotate(payload, ctx)
    if task == "embed":
        if payload.get("video_blob_path"):
            return _embed_video(payload, ctx)
        return _embed_images(payload, ctx)
    return _segment_images(payload, ctx)


def _decode_images(payload):
    images_b64 = payload.get("images")
    if not images_b64 and payload.get("image"):
        images_b64 = [payload["image"]]
    if not images_b64:
        raise ValueError("no images provided (expected 'images': [b64,...] or 'image': b64)")
    return [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in images_b64]


def _segment_images(payload, ctx):
    images = _decode_images(payload)
    prompt = payload.get("prompt", "bee")
    thr = float(payload.get("threshold", 0.3))
    maxd = int(payload.get("max_detections", 100))
    results = [[{"bbox": [x1, y1, x2, y2], "score": sc}
                for (x1, y1, x2, y2, sc) in _segment(ctx, img, prompt, thr, maxd)]
               for img in images]
    return {"prompt": prompt, "results": results}


def _embed_images(payload, ctx):
    images = _decode_images(payload)
    vecs = []
    for i in range(0, len(images), _EMBED_BATCH):
        vecs.append(_embed(ctx, images[i:i + _EMBED_BATCH]))
    emb = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, 0))
    return {"dim": int(emb.shape[1]) if emb.size else 0, "embeddings": emb.tolist()}


def _embed_video(payload, ctx):
    """Sample a video's frames and DINOv3-embed them → the drift-trigger primitive.

    {"task": "embed", "video_blob_path": "...", "sample_interval": 20, "max_frames": 200}
    -> {"dim": D, "n_frames": N, "embeddings": [[...], ...], "centroid": [...]}
    (centroid = L2-normalised mean, for cheap reference/compare on the web side.)
    """
    import tempfile
    from urllib.parse import urlparse

    import boto3
    import cv2

    region = os.environ.get("AWS_REGION")
    raw_bucket = os.environ["AWS_S3_BUCKET_RAW_VIDEOS"]
    s3 = boto3.client("s3", region_name=region)

    video_blob_path = payload["video_blob_path"]
    sample_interval = max(1, int(payload.get("sample_interval", 20)))
    max_frames = int(payload.get("max_frames", 200))

    embs = []
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        if video_blob_path.startswith("s3://"):
            u = urlparse(video_blob_path)
            s3.download_file(u.netloc, u.path.lstrip("/"), tmp.name)
        else:
            s3.download_file(raw_bucket, video_blob_path, tmp.name)
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            return {"dim": 0, "n_frames": 0, "embeddings": [], "centroid": []}
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        batch = []
        for frame_num in range(0, total, sample_interval):
            if len(embs) * _EMBED_BATCH + len(batch) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            batch.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            if len(batch) >= _EMBED_BATCH:
                embs.append(_embed(ctx, batch))
                batch = []
        if batch:
            embs.append(_embed(ctx, batch))
        cap.release()

    if not embs:
        return {"dim": 0, "n_frames": 0, "embeddings": [], "centroid": []}
    emb = np.concatenate(embs, axis=0)[:max_frames]
    centroid = emb.mean(axis=0)
    n = np.linalg.norm(centroid)
    centroid = (centroid / n if n else centroid)
    return {"dim": int(emb.shape[1]), "n_frames": int(emb.shape[0]),
            "embeddings": emb.tolist(), "centroid": centroid.tolist()}


def _pre_annotate(payload, ctx):
    """Sample frames + SAM 3-label them; mirrors the YOLO worker's task=pre_annotate.

    selection="diverse" uses DINOv3 to pick a representative, non-redundant subset.
    """
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
    maxd = int(payload.get("max_detections", 100))
    nms_iou = float(payload.get("nms_iou", 0.5))  # <=0 disables dedupe
    selection = payload.get("selection", "uniform")
    candidate_cap = int(payload.get("candidate_cap", max(max_frames * 4, 400)))
    # SAM 3 prompts = the project's classes minus "nest" (nest tubes stay manual).
    prompt_classes = [(i, c) for i, c in enumerate(classes) if c.lower() != "nest"]

    frames_out = []
    total_detections = 0
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

        # Decide which frame numbers to label. An explicit frame_numbers list
        # (e.g. the editor's per-frame pre-annotate) overrides sampling.
        explicit = payload.get("frame_numbers")
        if explicit:
            target_frames = [int(f) for f in explicit if 0 <= int(f) < max(total, 1)][:max_frames]
        elif selection == "diverse" and total > 0:
            target_frames = _diverse_targets(ctx, cap, total, max_frames, candidate_cap)
        else:
            target_frames = list(range(0, max(total, 1), sample_interval))[:max_frames]

        for frame_num in target_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if not ret:
                continue
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = []
            for cls_id, cls_name in prompt_classes:
                for (x1, y1, x2, y2, sc) in _segment(ctx, pil, cls_name, conf, maxd):
                    boxes.append({
                        "x": round(x1), "y": round(y1),
                        "w": round(x2 - x1), "h": round(y2 - y1),
                        "class": classes[cls_id], "class_id": cls_id,
                        "confidence": round(sc, 3) if sc is not None else None,
                    })
            # Dedupe overlapping boxes across the per-class prompt passes.
            if nms_iou > 0 and len(boxes) > 1:
                boxes = _nms(boxes, nms_iou)
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
        cap.release()

    logger.info("sam3 pre_annotate (%s): %s -> %d frames, %d detections",
                selection, video_blob_path, len(frames_out), total_detections)
    return {
        "status": "completed",
        "frames": frames_out,
        "selection": selection,
        "frames_with_activity": len(frames_out),
        "total_detections": total_detections,
        "video_fps": fps,
        "video_width": width,
        "video_height": height,
        "execution_seconds": round(time.time() - started, 2),
    }


def _diverse_targets(ctx, cap, total, max_frames, candidate_cap):
    """Sample up to candidate_cap frames, embed with DINOv3, farthest-point select
    max_frames of them → a sorted list of frame numbers to label."""
    import cv2

    step = max(1, total // candidate_cap)
    candidate_nums = list(range(0, total, step))[:candidate_cap]

    embs = []
    kept_nums = []
    batch_imgs = []
    batch_nums = []

    def _flush():
        if batch_imgs:
            embs.append(_embed(ctx, batch_imgs))
            kept_nums.extend(batch_nums)
            batch_imgs.clear()
            batch_nums.clear()

    for fn in candidate_nums:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            continue
        batch_imgs.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        batch_nums.append(fn)
        if len(batch_imgs) >= _EMBED_BATCH:
            _flush()
    _flush()

    if not kept_nums:
        return list(range(0, total, max(1, total // max_frames)))[:max_frames]
    emb = np.concatenate(embs, axis=0)
    idx = _farthest_point_select(emb, min(max_frames, len(kept_nums)))
    return sorted(kept_nums[i] for i in idx)


def output_fn(prediction, accept=JSON):
    return json.dumps(prediction), JSON
