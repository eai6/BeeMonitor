"""SAM 3 promptable detector — a text-prompt drop-in for YOLO in tracking.

Implements the same `BaseDetector.detect(frame) -> List[Detection]` contract the
tracker consumes, so `BeeTracking` can be pointed at SAM 3 instead of YOLO with
no other change. Users type a prompt ("bee", "hoverfly", "beetle") and SAM 3
grounds it per frame; the prompt string becomes each detection's label (→ the
tracking CSV `taxon`).

Runs the SAM 3 model IN-PROCESS on the local GPU (same as the SAM 3 endpoint
container) — never a per-frame network call. It's a heavy transformer, so it is
far slower than YOLO; the tracker's two-mode motion gating keeps it off idle
frames, and callers should prefer short clips. Model + weights load lazily on
the first `detect()`.
"""

from __future__ import annotations

import logging
import os
from typing import List

import numpy as np

from beemonitor.detection.base_detector import BaseDetector, Detection

logger = logging.getLogger(__name__)

_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")


class Sam3Detector(BaseDetector):
    def __init__(self, prompt: str = "bee", conf_threshold: float = 0.4,
                 iou_threshold: float = 0.5, max_detections: int = 100,
                 device: str | None = None):
        # One or more comma-separated prompts ("bee, wasp"). Each is grounded
        # separately and the union is deduped class-agnostically.
        self.prompts = [p.strip() for p in (prompt or "bee").split(",") if p.strip()] or ["bee"]
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self._device = device
        self._model = None
        self._processor = None

    # ── lazy model load ──────────────────────────────────────────────────────
    def _ensure_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        token = os.environ.get("HF_TOKEN") or None  # None when baked/offline
        logger.info("Loading SAM 3 (%s) on %s for tracking prompts=%s",
                    _MODEL_ID, self._device, self.prompts)
        self._processor = Sam3Processor.from_pretrained(_MODEL_ID, token=token)
        self._model = Sam3Model.from_pretrained(_MODEL_ID, token=token).to(self._device).eval()

    # ── one prompt on one PIL image → [(x1,y1,x2,y2,score), ...] ──────────────
    def _segment(self, pil_image, prompt: str):
        import torch

        proc = self._processor(images=pil_image, text=prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**proc)
        post = self._processor.post_process_instance_segmentation(
            outputs, threshold=self.conf_threshold, mask_threshold=0.5,
            target_sizes=proc.get("original_sizes").tolist(),
        )[0]
        boxes, scores = post.get("boxes"), post.get("scores")
        out = []
        if boxes is not None:
            box_list = boxes.tolist() if hasattr(boxes, "tolist") else list(boxes)
            score_list = (scores.tolist() if scores is not None and hasattr(scores, "tolist")
                          else [1.0] * len(box_list))
            for b, s in list(zip(box_list, score_list))[:self.max_detections]:
                x1, y1, x2, y2 = [float(v) for v in b]
                out.append((x1, y1, x2, y2, float(s) if s is not None else 1.0))
        return out

    # ── BaseDetector API ─────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray, **kwargs) -> List[Detection]:
        import cv2
        from PIL import Image

        self._ensure_model()
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dets: List[Detection] = []
        for prompt in self.prompts:
            for x1, y1, x2, y2, score in self._segment(pil, prompt):
                dets.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    centroid=((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    confidence=score,
                    label=prompt,       # → tracking CSV taxon
                    source="sam3",
                ))
        # Dedupe overlapping boxes from the per-prompt passes.
        return self.nms(dets, self.iou_threshold)

    def configure(self, **kwargs) -> None:
        if "prompt" in kwargs and kwargs["prompt"]:
            self.prompts = [p.strip() for p in str(kwargs["prompt"]).split(",") if p.strip()] or self.prompts
        if "conf_threshold" in kwargs:
            self.conf_threshold = float(kwargs["conf_threshold"])
        if "iou_threshold" in kwargs:
            self.iou_threshold = float(kwargs["iou_threshold"])

    def reset(self) -> None:
        pass

    def get_source_name(self) -> str:
        return "sam3"
