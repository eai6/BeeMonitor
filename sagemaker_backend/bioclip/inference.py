"""SageMaker inference handlers for the BioCLIP insect-ID endpoint (CPU).

Zero-shot classification of one mover crop. Two modes:

  * **Unconstrained** (Phase 1) — `TreeOfLifeClassifier` over the whole Tree of
    Life. Used when the request carries no candidate list.
  * **Location-constrained** (Phase 2) — when the request carries
    ``candidate_taxa`` (the region's insect species, from the Django side), a
    `CustomLabelsClassifier` restricted to those labels. The classifier is
    LRU-cached by label-set so its text embeddings are computed once per region
    and reused across that device's requests.

Request (``application/json``): ``{"image_b64": ..., "candidate_taxa": [...]?,
"rank": "species"?}``; a bare ``image/jpeg`` body is also accepted (unconstrained).
Response matches what ``beemonitor_web/apps/monitor/pipeline.py`` expects:

  {"predictions": [{"score", "common_name", "ranks": {kingdom..species}}, ...]}

For custom labels (binomial species names) the full lineage isn't known, so
``ranks`` is filled with species + the genus parsed from the binomial.
"""

import base64
import io
import json
import logging
import os
from functools import lru_cache

logger = logging.getLogger("beemonitor.bioclip")

JSON = "application/json"
TOPK = int(os.environ.get("BIOCLIP_TOPK", "5"))

# pybioclip's taxonomic-rank keys; map 1:1 to our Taxon.Rank values.
_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]


def model_fn(model_dir=None):
    """Load the unconstrained Tree-of-Life classifier once (at worker import)."""
    from bioclip import TreeOfLifeClassifier
    return TreeOfLifeClassifier()


def input_fn(body, content_type=JSON):
    """Decode the request into {image, candidate_taxa, rank}."""
    from PIL import Image
    if content_type and content_type.startswith("image/"):
        return {"image": Image.open(io.BytesIO(body)).convert("RGB"),
                "candidate_taxa": None, "rank": "species"}
    if content_type == JSON or not content_type:
        data = json.loads(body)
        if "image_b64" not in data:
            raise ValueError("JSON body must contain 'image_b64'")
        img = Image.open(io.BytesIO(base64.b64decode(data["image_b64"]))).convert("RGB")
        cands = data.get("candidate_taxa") or None
        return {"image": img, "candidate_taxa": cands, "rank": data.get("rank", "species")}
    raise ValueError(f"unsupported content-type: {content_type}")


@lru_cache(maxsize=16)
def _custom_classifier(labels_key: tuple):
    """A CustomLabelsClassifier for a label-set, cached so the (expensive) text
    embeddings are built once per region. ``labels_key`` is a tuple (hashable)."""
    from bioclip import CustomLabelsClassifier
    return CustomLabelsClassifier(list(labels_key))


def predict_fn(payload, classifier):
    """Run BioCLIP; constrained to candidate_taxa when present, else full ToL."""
    image = payload["image"]
    cands = payload.get("candidate_taxa")
    if cands:
        clf = _custom_classifier(tuple(cands))
        raw = clf.predict([image], k=min(TOPK, len(cands)))
        return _normalize(raw, constrained=True)
    from bioclip import Rank
    raw = classifier.predict([image], rank=Rank.SPECIES, k=TOPK)
    return _normalize(raw, constrained=False)


def _normalize(raw, constrained: bool):
    """Coerce pybioclip output into our {score, common_name, ranks} contract.

    ToL predictions carry the full rank breakdown; custom-label predictions carry
    only the label (a species name) + score, so we derive {genus, species} from
    the binomial.
    """
    out = []
    for p in raw:
        try:
            score = float(p.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if constrained:
            name = (p.get("classification") or p.get("species") or "").strip()
            ranks = {}
            if name:
                ranks["species"] = name
                parts = name.split()
                if len(parts) >= 2:
                    ranks["genus"] = parts[0]
        else:
            ranks = {r: p[r] for r in _RANKS if p.get(r)}
        out.append({"score": score,
                    "common_name": (p.get("common_name") or ""),
                    "ranks": ranks})
    out.sort(key=lambda d: d["score"], reverse=True)
    return out


def output_fn(prediction, accept=JSON):
    return json.dumps({"predictions": prediction}), JSON
