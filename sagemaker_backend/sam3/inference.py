"""SAM 3 auto-labeler — SageMaker BYOC handler.

Loads facebook/sam3 (Promptable Concept Segmentation) and, given base64 image(s) +
a text prompt (e.g. "bee"), returns per-image detections (boxes + scores) to
bootstrap YOLO training labels for new-domain footage the current detector misses.

Mirrors EcoMorph's verified HuggingFace SAM 3 integration:
    inputs  = processor(images=img, text=prompt, return_tensors="pt")
    outputs = model(**inputs)
    result  = processor.post_process_instance_segmentation(
                  outputs, threshold=..., mask_threshold=0.5,
                  target_sizes=inputs.get("original_sizes").tolist())[0]

Request (POST /invocations, application/json):
    {"images": ["<b64 jpeg>", ...], "prompt": "bee",
     "threshold": 0.3, "max_detections": 100}
  (or a single "image": "<b64>")
Response:
    {"prompt": "bee", "results": [[{"bbox": [x1,y1,x2,y2], "score": 0.9}, ...], ...]}
  results[i] are the detections for images[i], boxes in pixel coords.
"""

import base64
import io
import json
import logging
import os

from PIL import Image

logger = logging.getLogger("beemonitor.sam3")

JSON = "application/json"
_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")


def model_fn(model_dir=None):
    import torch
    from transformers import Sam3Model, Sam3Processor

    device = os.environ.get("SAM3_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    token = os.environ.get("HF_TOKEN") or None  # gated model; None if baked/offline
    processor = Sam3Processor.from_pretrained(_MODEL_ID, token=token)
    model = Sam3Model.from_pretrained(_MODEL_ID, token=token).to(device).eval()
    logger.info("SAM 3 (%s) loaded on %s", _MODEL_ID, device)
    return {"model": model, "processor": processor, "device": device}


def input_fn(body, content_type=JSON):
    data = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    images_b64 = data.get("images")
    if not images_b64 and data.get("image"):
        images_b64 = [data["image"]]
    if not images_b64:
        raise ValueError("no images provided (expected 'images': [b64,...] or 'image': b64)")
    images = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in images_b64]
    return {
        "images": images,
        "prompt": data.get("prompt", "bee"),
        "threshold": float(data.get("threshold", 0.3)),
        "max_detections": int(data.get("max_detections", 100)),
    }


def predict_fn(parsed, ctx):
    import torch

    model, processor, device = ctx["model"], ctx["processor"], ctx["device"]
    prompt = parsed["prompt"]
    thr = parsed["threshold"]
    maxd = parsed["max_detections"]

    results = []
    for img in parsed["images"]:
        proc = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**proc)
        post = processor.post_process_instance_segmentation(
            outputs,
            threshold=thr,
            mask_threshold=0.5,
            target_sizes=proc.get("original_sizes").tolist(),
        )[0]

        boxes = post.get("boxes")
        scores = post.get("scores")
        dets = []
        if boxes is not None:
            box_list = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
            score_list = (scores.tolist() if scores is not None and hasattr(scores, "tolist")
                          else [None] * len(box_list))
            for b, s in list(zip(box_list, score_list))[:maxd]:
                x1, y1, x2, y2 = [float(v) for v in b]
                dets.append({"bbox": [x1, y1, x2, y2],
                             "score": float(s) if s is not None else None})
        results.append(dets)

    return {"prompt": prompt, "results": results}


def output_fn(prediction, accept=JSON):
    return json.dumps(prediction), JSON
