"""Individual-marker identification — the contract.

A bee carries a physical mark (a paint dot today; a printed tag later) that says
*which* individual it is, as opposed to the taxon label the detector produces.
An identifier turns an image region into that individual's ID.

The signature is fixed by the tracker's existing hook
(``BeeTracking.process_frame``)::

    result = self.identifier.identify(frame, track_obj.last_bbox)
    if result:
        bee_id, method, confidence = result

so anything implementing :class:`BaseIdentifier` can be dropped in there
unchanged. The same objects are used by the web-side pass over the per-track
crops already stored in S3 — that path calls ``identify(crop)`` with no bbox,
because the crop *is* the region.
"""

from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple

import numpy as np

# (bee_id, method, confidence) — method is one of 'color' | 'number' | 'qrcode'.
Identification = Tuple[str, str, float]


def crop_bbox(frame: np.ndarray, bbox: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Return the bbox region of ``frame``, clamped to the frame bounds.

    ``bbox`` is ``(x1, y1, x2, y2)`` in full-frame pixels and is **not**
    guaranteed integral or in-bounds — the tracker stores it straight off the
    detector. ``None`` returns the whole frame (the crop-pass case). Returns
    ``None`` when the region is empty after clamping.
    """
    if frame is None or frame.size == 0:
        return None
    if bbox is None:
        return frame

    height, width = frame.shape[:2]
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    except (TypeError, ValueError):
        return None

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    xi1 = max(0, min(width, int(round(x1))))
    xi2 = max(0, min(width, int(round(x2))))
    yi1 = max(0, min(height, int(round(y1))))
    yi2 = max(0, min(height, int(round(y2))))
    if xi2 <= xi1 or yi2 <= yi1:
        return None
    return frame[yi1:yi2, xi1:xi2]


class BaseIdentifier(ABC):
    """Read one individual's ID from an image region."""

    #: Value written to the tracking CSV's ``bee_id_method`` column.
    method: str = "unknown"

    @abstractmethod
    def identify(self, frame: np.ndarray,
                 bbox: Optional[Sequence[float]] = None) -> Optional[Identification]:
        """Return ``(bee_id, method, confidence)``, or ``None`` if unreadable.

        Must never raise on odd input (empty crop, out-of-bounds bbox, greyscale
        frame) — it is called inside the per-frame tracking loop, where an
        exception would abort the whole video.
        """
        raise NotImplementedError


class BeeIdentifierManager(BaseIdentifier):
    """Fan out to several identifiers and keep the most confident reading.

    This is the ``marker_type: auto`` case: try every decoder we have and let
    the best-supported answer win. Constructed with a single decoder it behaves
    exactly like that decoder, so the tracker hook doesn't care which it gets.
    """

    method = "auto"

    def __init__(self, identifiers: Sequence[BaseIdentifier],
                 min_confidence: float = 0.0):
        self.identifiers = list(identifiers)
        self.min_confidence = float(min_confidence)

    def identify(self, frame, bbox=None):
        best = None
        for identifier in self.identifiers:
            try:
                result = identifier.identify(frame, bbox)
            except Exception:  # one bad decoder must not sink the rest
                continue
            if not result:
                continue
            if result[2] < self.min_confidence:
                continue
            if best is None or result[2] > best[2]:
                best = result
        return best
