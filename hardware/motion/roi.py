"""Hotel / nest detection and recording-ROI resolution.

Mirrors cloud BeeMonitor: detect the hotel (nest_detection.pt, class 0 = hotel,
class 1 = nest hole) and confine downstream motion detection to it. ultralytics
is imported lazily inside the functions so importing this module off-device (or
on the recording hot path, which never calls these) stays cheap.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from motion.config import (
    log, ROI, NEST_MODEL, NEST_CONF, HOTEL_ROI_DETECT,
    HOTEL_PAD_X_BASE, HOTEL_PAD_Y_BASE, HOTEL_SETTLE_SECONDS,
    MAIN_W, MAIN_H, LORES_W, LORES_H,
)
from motion.frames import _main_array_to_bgr, _scale_roi
from motion.overrides import load_roi_override_lores


def _parse_roi():
    if not ROI:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in ROI.split(","))
        return (x1, y1, x2, y2)
    except ValueError:
        log.warning("ignoring malformed BEEMONITOR_ROI=%r (want x1,y1,x2,y2)", ROI)
        return None


def detect_hotel_roi(frame_bgr):
    """Cloud-faithful hotel ROI from one frame, in that frame's pixel coords.

    Runs nest_detection.pt (class 0 = hotel, class 1 = nest hole) like cloud
    BeeMonitor. Prefers the highest-confidence hotel box; else the bounding box
    of all detected nest holes. Pads (100/50 px @ 1920x1080, scaled to the
    capture res) and clamps. Returns None on any failure -> caller uses the
    whole frame.
    """
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - heavy, lazy
    except ImportError:
        log.warning("ultralytics not installed — cannot detect hotel, using full frame")
        return None
    if not Path(NEST_MODEL).exists():
        log.warning("nest model not found at %s — using full frame", NEST_MODEL)
        return None
    try:
        model = YOLO(NEST_MODEL)
        results = model.predict(frame_bgr, conf=NEST_CONF, verbose=False)
    except Exception as e:  # pragma: no cover - model/runtime issues mustn't crash startup
        log.warning("nest detection failed (%s) — using full frame", e)
        return None

    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        log.warning("no hotel/nest detections — using full frame")
        return None

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    h, w = frame_bgr.shape[:2]
    HOTEL_CLASS, NEST_CLASS = 0, 1

    hotel_mask = cls == HOTEL_CLASS
    if hotel_mask.any():
        # Highest-confidence hotel box.
        idx = np.where(hotel_mask)[0]
        best = idx[conf[idx].argmax()]
        x1, y1, x2, y2 = xyxy[best]
        source = f"hotel box (conf {conf[best]:.2f})"
    else:
        nests = xyxy[cls == NEST_CLASS]
        if len(nests) == 0:
            nests = xyxy  # no class-1 either; bound whatever was found
        x1, y1 = nests[:, 0].min(), nests[:, 1].min()
        x2, y2 = nests[:, 2].max(), nests[:, 3].max()
        source = f"{len(nests)} nest holes"

    pad_x = HOTEL_PAD_X_BASE * (w / 1920.0)
    pad_y = HOTEL_PAD_Y_BASE * (h / 1080.0)
    roi = (
        int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y)),
        int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y)),
    )
    if roi[2] - roi[0] < 10 or roi[3] - roi[1] < 10:
        log.warning("detected hotel ROI degenerate (%s) — using full frame", roi)
        return None
    log.info("hotel ROI from %s: %s in %dx%d frame", source, roi, w, h)
    return roi


def detect_nest_boxes(frame_bgr):
    """Nest-hole detections (class 1 of nest_detection.pt) in the frame's pixel
    coords — for the on-demand debug overlay. Returns a list of (x1,y1,x2,y2);
    empty on any failure. Runs the model fresh (only used on debug captures)."""
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - heavy, lazy
    except ImportError:
        return []
    if not Path(NEST_MODEL).exists():
        return []
    try:
        model = YOLO(NEST_MODEL)
        results = model.predict(frame_bgr, conf=NEST_CONF, verbose=False)
    except Exception as e:  # pragma: no cover
        log.warning("nest detection (overlay) failed: %s", e)
        return []
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    NEST_CLASS = 1
    return [tuple(int(v) for v in b) for b, c in zip(xyxy, cls) if c == NEST_CLASS]


def _order_nest_boxes(boxes):
    """Order nest boxes top→bottom, left→right and assign IDs the same way as the
    beemonitor nest_detector: cluster into rows (by y), sort each row by x, and
    label id = (col+1) + row*10 (row 1 → 1-10, row 2 → 11-20, …). Returns a list
    of (id:int, box) in that order."""
    if not boxes:
        return []
    items = [((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0, b) for b in boxes]  # (cy, cx, box)
    heights = sorted(b[3] - b[1] for b in boxes)
    med_h = heights[len(heights) // 2] if heights else 20
    row_gap = max(8.0, med_h * 0.6)
    items.sort(key=lambda t: t[0])  # by cy
    rows, cur, cur_sum = [], [items[0]], items[0][0]
    for it in items[1:]:
        row_mean = cur_sum / len(cur)
        if it[0] - row_mean > row_gap:
            rows.append(cur); cur, cur_sum = [it], it[0]
        else:
            cur.append(it); cur_sum += it[0]
    rows.append(cur)
    out = []
    for ri, row in enumerate(rows):
        row.sort(key=lambda t: t[1])  # by cx
        for ci, (_cy, _cx, box) in enumerate(row):
            out.append(((ci + 1) + ri * 10, box))
    return out


def _resolve_record_roi(cam):
    """Decide the lores-coord detection ROI for recording.

    Priority: explicit BEEMONITOR_ROI (lores coords) > hotel auto-detection >
    whole frame. Mirrors cloud BeeMonitor, which detects the hotel first and
    confines downstream detection to it.
    """
    override = load_roi_override_lores()
    if override is not None:
        log.info("using dashboard ROI override (lores coords): %s", override)
        return override
    env_roi = _parse_roi()
    if env_roi is not None:
        log.info("using explicit BEEMONITOR_ROI=%s (lores coords)", env_roi)
        return env_roi
    if not HOTEL_ROI_DETECT:
        log.info("hotel auto-detection disabled — recording on full frame")
        return None
    # Let auto-exposure settle, then grab a clean main-stream frame for detection.
    if HOTEL_SETTLE_SECONDS > 0:
        time.sleep(HOTEL_SETTLE_SECONDS)
    try:
        bgr = _main_array_to_bgr(cam.capture_array("main"))
    except Exception as e:  # pragma: no cover
        log.warning("could not grab a frame for hotel detection (%s) — full frame", e)
        return None
    roi_main = detect_hotel_roi(bgr)
    if roi_main is None:
        log.info("hotel detection unsuccessful — recording on full frame")
        return None
    roi_lores = _scale_roi(roi_main, (MAIN_W, MAIN_H), (LORES_W, LORES_H))
    log.info("hotel ROI scaled to lores: %s", roi_lores)
    return roi_lores
