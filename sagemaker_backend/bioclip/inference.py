"""SageMaker inference handlers for the BioCLIP insect-ID endpoint (CPU).

Zero-shot Tree-of-Life classification of one mover crop. Input is JPEG bytes
(``image/jpeg``) or ``{"image_b64": "..."}`` JSON; output matches what
``beemonitor_web/apps/monitor/pipeline.py`` expects:

    {"predictions": [
        {"score": 0.82, "common_name": "common eastern bumble bee",
         "ranks": {"kingdom": "Animalia", ..., "species": "Bombus impatiens"}},
        ...
    ]}

Runs on a SageMaker **Serverless** endpoint (scale-to-zero), so there's no idle
GPU to keep alive. The BioCLIP weights are baked into the image at build time
(see Dockerfile.bioclip) so cold starts don't download them.
"""

import base64
import io
import json
import logging
import os

logger = logging.getLogger("beemonitor.bioclip")

JSON = "application/json"
TOPK = int(os.environ.get("BIOCLIP_TOPK", "5"))

# pybioclip emits these taxonomic keys; they map 1:1 to our Taxon.Rank values.
_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def model_fn(model_dir=None):
    """Load the Tree-of-Life classifier once (at worker import)."""
    from bioclip import TreeOfLifeClassifier
    return TreeOfLifeClassifier()


def input_fn(body, content_type=JSON):
    """Decode the request into a PIL RGB image."""
    from PIL import Image
    if content_type and content_type.startswith("image/"):
        return Image.open(io.BytesIO(body)).convert("RGB")
    if content_type == JSON or not content_type:
        data = json.loads(body)
        if "image_b64" not in data:
            raise ValueError("JSON body must contain 'image_b64'")
        return Image.open(io.BytesIO(base64.b64decode(data["image_b64"]))).convert("RGB")
    raise ValueError(f"unsupported content-type: {content_type}")


def predict_fn(image, classifier):
    """Run BioCLIP zero-shot; return a normalized ranked list of taxa."""
    from bioclip import Rank
    raw = classifier.predict([image], rank=Rank.SPECIES, k=TOPK)
    # pybioclip returns a list of dicts with the taxonomic-rank keys above plus
    # 'common_name' and 'score'. Normalize into our contract.
    out = []
    for p in raw:
        ranks = {r: p[r] for r in _RANKS if p.get(r)}
        try:
            score = float(p.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        out.append({
            "score": score,
            "common_name": p.get("common_name", "") or "",
            "ranks": ranks,
        })
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def output_fn(prediction, accept=JSON):
    return json.dumps({"predictions": prediction}), JSON
