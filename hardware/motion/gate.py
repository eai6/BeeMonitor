"""Motion gate: slim MOG2 blob detector operating on the lores Y-plane."""

from __future__ import annotations

import cv2
import numpy as np

from motion.config import (
    MOG2_HISTORY, MOG2_VAR_THRESHOLD, MORPH_KERNEL, MORPH_ITERS,
    MIN_BLOB_AREA, MAX_BLOB_AREA, MIN_MOTION_BLOBS, DETECT_SHADOWS,
    SHADOW_THRESHOLD,
)


class MotionGate:
    """MOG2 background subtraction + contour-area filtering -> motion?

    Faithful to ``BlobDetector`` but self-contained (cv2 + numpy only) and
    operating on a single-channel grayscale frame, so it pulls in none of the
    YOLO/torch deps that ``beemonitor.detection`` would.

    The three tunable thresholds live here as instance attributes so the
    dry-run visualiser can slide them live:
        * ``var_threshold`` — MOG2 per-pixel foreground sensitivity
        * ``min_area`` / ``max_area`` — per-blob size window (lores pixels)
        * ``min_blobs`` — how many qualifying blobs == "motion"
    After each ``update`` the last mask + kept blobs are stashed on the
    instance (``last_mask``, ``last_blobs``) for visualisation.
    """

    def __init__(self, roi=None, history=MOG2_HISTORY,
                 var_threshold=MOG2_VAR_THRESHOLD, morph_kernel=MORPH_KERNEL,
                 morph_iters=MORPH_ITERS, min_area=MIN_BLOB_AREA,
                 max_area=MAX_BLOB_AREA, min_blobs=MIN_MOTION_BLOBS,
                 detect_shadows=DETECT_SHADOWS, shadow_threshold=SHADOW_THRESHOLD):
        self.var_threshold = var_threshold
        self.history = history
        self.detect_shadows = detect_shadows
        self.shadow_threshold = shadow_threshold
        self.bg = self._make_bg()
        self._morph_kernel = morph_kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.morph_iters = morph_iters
        self.min_area = min_area
        self.max_area = max_area
        self.min_blobs = min_blobs
        self.roi = roi  # (x1, y1, x2, y2) in lores coords, or None
        self.last_mask = None
        self.last_blobs = []   # list of (x, y, w, h, area) for kept blobs

    def _make_bg(self):
        """Build a MOG2 subtractor with the current shadow/threshold settings."""
        bg = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows)
        if self.detect_shadows:
            bg.setShadowThreshold(self.shadow_threshold)
        return bg

    def set_var_threshold(self, value: float) -> None:
        self.var_threshold = value
        self.bg.setVarThreshold(value)

    def set_morph_kernel(self, size: int) -> None:
        size = max(1, size | 1)  # keep odd & >= 1
        self._morph_kernel = size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _crop(self, gray):
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            return gray[y1:y2, x1:x2]
        return gray

    def update(self, gray: np.ndarray):
        """Feed one grayscale frame. Returns (motion: bool, n_blobs, motion_area)."""
        gray = self._crop(gray)

        fg = self.bg.apply(gray)
        if self.detect_shadows:
            # MOG2 marks shadow pixels as 127 (vs 255 for hard foreground); drop
            # them so moving shadows / illumination changes don't count as motion.
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=self.morph_iters)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n_blobs = 0
        motion_area = 0.0
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            n_blobs += 1
            motion_area += area
            blobs.append((*cv2.boundingRect(c), area))

        self.last_mask = fg
        self.last_blobs = blobs
        return (n_blobs >= self.min_blobs), n_blobs, motion_area

    def warm(self, gray: np.ndarray) -> None:
        """Update the background model without evaluating motion (warmup)."""
        self.bg.apply(self._crop(gray))

    def reset(self) -> None:
        """Discard and rebuild the background model from scratch.

        Keeps the current var_threshold (calibration may have tuned it) and the
        shadow setting. The caller should re-warm for a few seconds before
        trusting motion again, since the fresh model treats the first frames as
        all-foreground.
        """
        self.bg = self._make_bg()
