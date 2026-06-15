"""Small, dependency-free frame & geometry helpers shared across the package."""

from __future__ import annotations

import cv2


def _main_array_to_bgr(frame):
    """Normalise a picamera2 main-stream array to BGR regardless of pixel format."""
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
    # YUV420 (I420) packed: shape (H*3/2, W)
    return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)


def _scale_roi(roi, src_wh, dst_wh):
    """Scale an (x1,y1,x2,y2) box from src resolution to dst, clamped to dst."""
    x1, y1, x2, y2 = roi
    sw, sh = src_wh
    dw, dh = dst_wh
    sx, sy = dw / sw, dh / sh
    return (
        max(0, int(x1 * sx)), max(0, int(y1 * sy)),
        min(dw, int(x2 * sx)), min(dh, int(y2 * sy)),
    )
