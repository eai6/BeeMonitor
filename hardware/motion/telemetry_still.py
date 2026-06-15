"""On-demand telemetry still capture (picture / live view / ROI editor).

A CLEAN frame — no overlays burned in. The dashboard's ROI editor draws the hotel
ROI + nest tubes as editable boxes over this image from the device's stored layout
(roi_override / nest_layout), so the user can drag/edit them. Burned-in boxes can't
be edited, so the device no longer draws anything. See devices/roi_editor.html.
"""

from __future__ import annotations

from datetime import datetime

import cv2

from motion.config import log, TELEMETRY_QUEUE, TELEMETRY_IMAGE_HEIGHT
from motion.frames import _main_array_to_bgr


def _save_telemetry_still(cam) -> None:
    """Capture one downscaled, CLEAN JPEG into the telemetry queue (best-effort).

    Grabs the main (recorded) stream so the still reflects the real framing, then
    downscales to keep it small. No ROI/nest overlay is burned in — the dashboard
    overlays those interactively. Never let a capture error stop recording.
    """
    try:
        bgr = _main_array_to_bgr(cam.capture_array("main"))
        h, w = bgr.shape[:2]
        if h > TELEMETRY_IMAGE_HEIGHT:
            scale = TELEMETRY_IMAGE_HEIGHT / h
            bgr = cv2.resize(
                bgr, (int(w * scale), TELEMETRY_IMAGE_HEIGHT),
                interpolation=cv2.INTER_AREA)

        TELEMETRY_QUEUE.mkdir(parents=True, exist_ok=True)
        out = TELEMETRY_QUEUE / (datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".jpg")
        cv2.imwrite(str(out), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

        # Keep only the few most recent so a stalled uploader can't fill disk.
        queued = sorted(TELEMETRY_QUEUE.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
        for old in queued[:-3]:
            try:
                old.unlink()
            except OSError:
                pass
        log.info("telemetry still -> %s", out.name)
    except Exception as e:  # pragma: no cover - must never crash recording
        log.warning("telemetry still capture failed: %s", e)
