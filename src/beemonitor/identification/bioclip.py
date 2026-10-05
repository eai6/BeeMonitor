"""BioCLIP zero-shot species classification of track crops (GPU worker).

Same contract as ``SpeciesIdentifier.classify_images``: a list of BGR crops in,
``(species, confidence)`` per crop out. Constrained to ``candidates`` — the
species recorded near the device (``monitor/priors.py``) — when there are any,
otherwise the whole Tree of Life. Constraining is the biggest accuracy lever
for zero-shot BioCLIP on small, imperfect crops.

The classifier (and, for candidates, the text embeddings of the label set) is
expensive to build, so instances are cached per label set for the life of the
worker process.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import cv2

logger = logging.getLogger(__name__)


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@lru_cache(maxsize=8)
def _classifier(labels: tuple):
    if labels:
        from bioclip import CustomLabelsClassifier
        return CustomLabelsClassifier(list(labels), device=_device())
    from bioclip import TreeOfLifeClassifier
    return TreeOfLifeClassifier(device=_device())


class BioClipIdentifier:
    method = "bioclip"

    def __init__(self, candidates=None, min_crop_side: int = 24):
        # Order-insensitive cache key; duplicates dropped.
        self.labels = tuple(sorted({str(c).strip() for c in (candidates or []) if str(c).strip()}))
        self.min_crop_side = int(min_crop_side)

    @property
    def constrained(self) -> bool:
        return bool(self.labels)

    def _usable(self, image) -> bool:
        return (image is not None and getattr(image, "ndim", 0) == 3
                and min(image.shape[:2]) >= self.min_crop_side)

    def classify_images(self, images, batch: int = 32):
        from PIL import Image

        results = [None] * len(images)
        usable = [(i, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                  for i, img in enumerate(images) if self._usable(img)]
        clf = _classifier(self.labels)
        for start in range(0, len(usable), batch):
            chunk = usable[start:start + batch]
            pil = [im for _i, im in chunk]
            if self.labels:
                preds = clf.predict(pil, k=1, batch_size=len(pil))
            else:
                from bioclip import Rank
                preds = clf.predict(pil, rank=Rank.SPECIES, k=1, batch_size=len(pil))
            # k=1: one prediction per image, in input order.
            if len(preds) != len(chunk):
                logger.warning("BioCLIP returned %d predictions for %d crops; skipping batch",
                               len(preds), len(chunk))
                continue
            for (i, _im), p in zip(chunk, preds):
                name = (p.get("classification") or p.get("species") or "").strip()
                if name:
                    results[i] = (name, float(p.get("score", 0.0) or 0.0))
        return results
