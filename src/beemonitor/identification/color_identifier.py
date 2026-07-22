"""Read a paint-mark colour off a bee.

Solitary-bee studies mark individuals with a dot of enamel or queen-marking
paint. That is a small, *saturated* patch on an animal that is otherwise brown,
black or dull amber — which is what makes this tractable without any model: find
the one bright, saturated blob in the crop and ask which palette colour its hue
is closest to.

The default palette is the international queen-marking set (white, yellow, red,
green, blue), because that is what the paint pens come in. White is handled
separately: it is the one mark that is *un*saturated, so a hue test says nothing
about it and it is found by brightness instead.

Deliberately dependency-free beyond OpenCV/numpy, which the worker image already
carries — no new wheels in the GPU image to add this.
"""

import logging
from typing import Dict, Optional, Sequence

import cv2
import numpy as np

from .base import BaseIdentifier, crop_bbox

logger = logging.getLogger(__name__)

# Hue centres in OpenCV's 0-179 space. Red straddles the wrap point, which the
# circular distance below handles.
QUEEN_MARKING_PALETTE: Dict[str, float] = {
    "red": 0.0,
    "yellow": 27.0,
    "green": 60.0,
    "blue": 110.0,
}

# Extra hues for studies using more than the standard five.
EXTENDED_PALETTE: Dict[str, float] = {
    **QUEEN_MARKING_PALETTE,
    "orange": 12.0,
    "cyan": 90.0,
    "violet": 130.0,
    "pink": 165.0,
}


def hue_distance(a: float, b: float) -> float:
    """Circular distance between two OpenCV hues (0-179)."""
    diff = abs(float(a) - float(b)) % 180.0
    return min(diff, 180.0 - diff)


class ColorIdentifier(BaseIdentifier):
    """Classify a paint mark into a named palette colour.

    Args:
        palette: name -> hue centre (OpenCV 0-179). Defaults to the queen-marking
            colours. ``white`` is added implicitly unless ``detect_white=False``.
        sat_min / val_min: a pixel counts as "paint" above these. Bee cuticle is
            comparatively desaturated, so ``sat_min`` is the main discriminator.
        white_sat_max / white_val_min: white paint is the inverse — bright and
            unsaturated.
        min_area_frac: the blob must cover at least this fraction of the crop.
            Guards against a single speckle of specular highlight reading as a
            mark.
        max_hue_distance: beyond this the blob is closer to nothing in the
            palette than to something, so report nothing rather than guess.
        min_confidence: readings below this are dropped.
    """

    method = "color"

    def __init__(
        self,
        palette: Optional[Dict[str, float]] = None,
        sat_min: int = 120,
        val_min: int = 80,
        detect_white: bool = True,
        white_sat_max: int = 40,
        white_val_min: int = 200,
        min_area_frac: float = 0.01,
        max_hue_distance: float = 20.0,
        min_confidence: float = 0.35,
    ):
        self.palette = dict(palette or QUEEN_MARKING_PALETTE)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.detect_white = bool(detect_white)
        self.white_sat_max = int(white_sat_max)
        self.white_val_min = int(white_val_min)
        self.min_area_frac = float(min_area_frac)
        self.max_hue_distance = float(max_hue_distance)
        self.min_confidence = float(min_confidence)

    # ── main entry point ─────────────────────────────────────────────────────
    def identify(self, frame, bbox=None):
        try:
            region = crop_bbox(frame, bbox)
            if region is None or region.size == 0:
                return None
            if region.ndim != 3 or region.shape[2] != 3:
                return None  # greyscale/alpha — nothing to classify

            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            total = float(region.shape[0] * region.shape[1])
            if total <= 0:
                return None

            best = self._best_chromatic(hsv, total)
            if self.detect_white:
                white = self._white(hsv, total)
                if white and (best is None or white[2] > best[2]):
                    best = white

            if best is None or best[2] < self.min_confidence:
                return None
            return best
        except Exception:  # called inside the tracking loop — never propagate
            logger.debug("ColorIdentifier failed on a region", exc_info=True)
            return None

    # ── internals ────────────────────────────────────────────────────────────
    def _largest_blob(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Clean the mask and return a boolean mask of its largest component."""
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return None
        # Label 0 is the background.
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    def _best_chromatic(self, hsv: np.ndarray, total: float):
        """Nearest palette colour for the biggest saturated blob."""
        if not self.palette:
            return None
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        mask = ((sat >= self.sat_min) & (val >= self.val_min)).astype(np.uint8) * 255
        blob = self._largest_blob(mask)
        if blob is None:
            return None

        area_frac = float(blob.sum()) / total
        if area_frac < self.min_area_frac:
            return None

        hues = hsv[:, :, 0][blob].astype(np.float64)
        if hues.size == 0:
            return None
        centre = _circular_mean_hue(hues)
        spread = _circular_spread_hue(hues, centre)

        name, distance = min(
            ((n, hue_distance(centre, h)) for n, h in self.palette.items()),
            key=lambda pair: pair[1],
        )
        if distance > self.max_hue_distance:
            return None

        confidence = self._confidence(area_frac, distance, spread,
                                      float(hsv[:, :, 1][blob].mean()))
        return (name, self.method, confidence)

    def _white(self, hsv: np.ndarray, total: float):
        """White paint: bright and unsaturated, so hue is meaningless."""
        sat, val = hsv[:, :, 1], hsv[:, :, 2]
        mask = ((sat <= self.white_sat_max) &
                (val >= self.white_val_min)).astype(np.uint8) * 255
        blob = self._largest_blob(mask)
        if blob is None:
            return None
        area_frac = float(blob.sum()) / total
        if area_frac < self.min_area_frac:
            return None
        # No hue to agree on, so confidence rests on size and how cleanly
        # unsaturated the blob is.
        purity = 1.0 - min(1.0, float(sat[blob].mean()) / max(1, self.white_sat_max))
        confidence = round(min(1.0, 0.45 + 0.35 * _area_score(area_frac) + 0.2 * purity), 3)
        return ("white", self.method, confidence)

    def _confidence(self, area_frac, distance, spread, mean_sat):
        """Blend the independent signals into 0..1.

        A confident reading is a reasonably large blob, whose hue sits close to a
        palette centre, is internally consistent, and is strongly saturated.
        """
        area = _area_score(area_frac)
        closeness = 1.0 - min(1.0, distance / max(1e-6, self.max_hue_distance))
        tightness = 1.0 - min(1.0, spread / 30.0)
        saturation = min(1.0, mean_sat / 255.0)
        score = 0.30 * area + 0.30 * closeness + 0.25 * tightness + 0.15 * saturation
        return round(min(1.0, max(0.0, score)), 3)


def _area_score(area_frac: float) -> float:
    """Saturating score for blob size — a mark covering ~15% of the crop is as
    good as it needs to get; bigger usually means the mask caught the background."""
    return min(1.0, float(area_frac) / 0.15)


def _circular_mean_hue(hues: np.ndarray) -> float:
    """Mean hue on the circle, so reds either side of the 0/179 wrap agree."""
    radians = hues * (np.pi / 90.0)  # 0..179 -> 0..2pi
    angle = np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())
    return float((angle * (90.0 / np.pi)) % 180.0)


def _circular_spread_hue(hues: np.ndarray, centre: float) -> float:
    """Mean absolute circular deviation from ``centre``."""
    if hues.size == 0:
        return 0.0
    return float(np.mean([hue_distance(h, centre) for h in hues]))


def build_identifier(marker_type: str = "auto",
                     palette: Optional[Dict[str, float]] = None,
                     **kwargs) -> Optional[BaseIdentifier]:
    """Build the identifier for a pipeline node's ``marker_type``.

    ``auto`` fans out over every decoder we have. Today that is colour only —
    tag decoding (ArUco/BEEtag) registers here when it lands, and nothing else
    has to change.
    """
    from .base import BeeIdentifierManager

    marker_type = (marker_type or "auto").strip().lower()
    color = ColorIdentifier(palette=palette, **kwargs)
    if marker_type == "color":
        return color
    if marker_type == "auto":
        return BeeIdentifierManager([color])
    return None


def available_marker_types() -> Sequence[str]:
    """Marker types with a decoder behind them, for the UI to offer."""
    return ("auto", "color")
