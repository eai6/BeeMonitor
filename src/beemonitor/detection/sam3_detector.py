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
import threading
from typing import List

import numpy as np

from beemonitor.core.profiling import PROFILER
from beemonitor.detection.base_detector import BaseDetector, Detection

logger = logging.getLogger(__name__)

_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")

# One model per PROCESS, not per detector. The lock serialises both the lazy
# ``transformers`` import (which is not thread-safe on first access) and the
# load itself, so concurrent invocations cannot race or duplicate a multi-GB
# model on the GPU.
_LOAD_LOCK = threading.Lock()
_SHARED_MODEL = None
_SHARED_PROCESSOR = None
_SHARED_DEVICE = None


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
        """Load SAM 3 once per process, serialised across threads.

        Two things made this the batch's second failure mode:

        **The import races.** ``transformers`` resolves submodules lazily, so a
        first ``from transformers import Sam3Model`` from two threads at once
        can have one of them observe a half-registered module and raise
        ``ImportError: cannot import name 'Sam3Model'``. The container serves
        several invocations on gunicorn gthread workers, each constructing its
        own detector, so the first batch to run SAM 3 concurrently hit it —
        intermittently, which is why a single retry looked fine.

        **The model was loaded per detector.** Every concurrent invocation put
        another multi-GB copy on the same GPU. Sharing one across the process
        is what makes more than one job per instance feasible at all; the model
        is read-only in eval mode and the prompt is per-call, so there is no
        per-detector state to keep separate.
        """
        if self._model is not None:
            return

        global _SHARED_MODEL, _SHARED_PROCESSOR, _SHARED_DEVICE
        with _LOAD_LOCK:
            if _SHARED_MODEL is None:
                import torch
                from transformers import Sam3Model, Sam3Processor

                device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
                token = os.environ.get("HF_TOKEN") or None  # None when baked/offline
                logger.info("Loading SAM 3 (%s) on %s (once per process)",
                            _MODEL_ID, device)
                _SHARED_PROCESSOR = Sam3Processor.from_pretrained(_MODEL_ID, token=token)
                _SHARED_MODEL = (Sam3Model.from_pretrained(_MODEL_ID, token=token)
                                 .to(device).eval())
                _SHARED_DEVICE = device

            self._processor = _SHARED_PROCESSOR
            self._model = _SHARED_MODEL
            self._device = _SHARED_DEVICE
        logger.info("SAM 3 ready on %s for prompts=%s", self._device, self.prompts)

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
        # Recorded as "inference", same stage name YOLODetector uses, with a
        # count of one per FRAME rather than per prompt — so calls means frames
        # whichever detector ran, and the seconds cover every prompt pass for
        # that frame. Without this a SAM 3 run reported gpu_seconds = 0, which
        # reads as "the GPU was idle" on the very detector where it is busiest.
        with PROFILER.stage("inference"):
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
