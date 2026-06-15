"""On-demand telemetry still capture (picture / live view / ROI debug overlay)."""

from __future__ import annotations

from datetime import datetime

import cv2

from motion.config import (
    log, TELEMETRY_QUEUE, TELEMETRY_IMAGE_HEIGHT, LORES_W, LORES_H,
)
from motion.frames import _main_array_to_bgr, _scale_roi
from motion.overrides import load_nest_layout


def _save_telemetry_still(cam, roi=None, draw_roi=False) -> None:
    """Capture one downscaled JPEG into the telemetry queue (best-effort).

    The telemetry service ships the latest queued image over cellular. We grab
    the main (recorded) stream so the still reflects the real framing, then
    downscale to keep it small. Never let a capture error stop recording.

    ``draw_roi`` overlays the active motion ROI (the hotel region, in lores
    coords) — a debugging aid so the dashboard can show exactly where motion is
    gated. If ``roi`` is None the gate runs on the whole frame (more sensitive),
    which we mark explicitly.
    """
    try:
        bgr = _main_array_to_bgr(cam.capture_array("main"))

        h, w = bgr.shape[:2]
        if draw_roi:
            th = max(2, w // 400)
            # Nest boxes: a dashboard-edited layout wins (normalized -> main
            # coords); otherwise fresh detections, ordered top→bottom/left→right.
            layout = load_nest_layout()
            if layout:
                nest_boxes = [(nid, (int(b[0] * w), int(b[1] * h),
                                     int(b[2] * w), int(b[3] * h)))
                              for nid, b in layout]
            else:
                # No on-device detection — just take the picture and overlay the
                # device's configured nest tubes (the dashboard layout). If none is
                # set yet, draw none. (Nest detection only runs in the cloud.)
                nest_boxes = []
            for nid, (nx1, ny1, nx2, ny2) in nest_boxes:
                cv2.rectangle(bgr, (nx1, ny1), (nx2, ny2), (0, 0, 255), max(1, th // 2))
                ty = ny1 - 6 if ny1 > 20 else ny2 + 22
                cv2.putText(bgr, str(nid), (nx1, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), max(1, th // 2), cv2.LINE_AA)
            # Active motion ROI (green hotel box) or full-frame (orange).
            if roi is not None:
                rx = _scale_roi(roi, (LORES_W, LORES_H), (w, h))
                cv2.rectangle(bgr, (rx[0], rx[1]), (rx[2], rx[3]), (0, 255, 0), th)
                _label = "motion ROI · %d nests" % len(nest_boxes)
                _color = (0, 255, 0)
            else:
                cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), (0, 165, 255), th)
                _label = "ROI: full frame · %d nests" % len(nest_boxes)
                _color = (0, 165, 255)
            cv2.putText(bgr, _label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 0), th + 2, cv2.LINE_AA)
            cv2.putText(bgr, _label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, _color, th, cv2.LINE_AA)
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
