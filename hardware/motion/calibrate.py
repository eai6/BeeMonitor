"""Calibration (on-Pi, scheduled): learn the bee-sized blob window from YOLO.

YOLO is too slow for the Pi 4 hot path, but running it on a few hundred frames
*once* is fine. We measure the MOG2 blob area of every YOLO-confirmed bee and
freeze the 5th/95th-percentile window into calibration.json. The recorder then
runs MOG2 only. This is the on-device version of cloud BeeMonitor's
``BlobDetector.learn_geometric_thresholds_from_video``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from motion.config import (
    log, CALIB_FILE, CALIB_MAX_AGE_DAYS, LORES_W, LORES_H, FPS,
    YOLO_CONF, CALIB_TARGET_SAMPLES, CALIB_MIN_SAMPLES, CALIB_YOLO_EVERY,
    MOG2_VAR_THRESHOLD, MIN_MOTION_BLOBS, RECORD_DIR, MAX_BLOB_AREA,
    CALIB_MIN_PAD, CALIB_MAX_PAD, CALIB_AREA_FLOOR, CALIB_MIN_SPAN,
)
from motion.gate import MotionGate
from motion.roi import _parse_roi
from motion.overrides import load_roi_override_lores, load_roi_polygon_lores


def _lores_from_bgr(frame_bgr):
    """Mimic the Pi's lores stream from a full-res BGR frame: downscale + gray."""
    small = cv2.resize(frame_bgr, (LORES_W, LORES_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _bbox_overlap(a, b) -> bool:
    """Do (x1,y1,x2,y2) boxes a and b intersect at all?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _iter_video_frames(path):
    """Yield BGR frames from an mp4, or nothing if it can't be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log.warning("cannot open snippet for calibration: %s", path)
        return
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _find_snippets(limit):
    """Newest recorded .mp4 snippets first (these contain the activity/bees)."""
    mp4s = list(RECORD_DIR.rglob("*.mp4"))
    mp4s.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[:limit]


def _calibration_fresh() -> bool:
    """True if calibration.json exists and is younger than CALIB_MAX_AGE_DAYS."""
    if not CALIB_FILE.exists():
        return False
    age_days = (time.time() - CALIB_FILE.stat().st_mtime) / 86400.0
    return age_days < CALIB_MAX_AGE_DAYS


def calibrate(video_paths, model_path: str, force: bool = False) -> int:
    """Learn the bee blob-area window from saved snippets (no live bee needed).

    Runs YOLO over recorded clips, measures the MOG2 blob area of every
    YOLO-confirmed bee, and writes the 5th/95th-percentile window to
    calibration.json. Accumulates across clips until it has enough samples.
    """
    if not force and _calibration_fresh():
        log.info("calibration.json is < %.0f days old — skipping (use --force "
                 "to recalibrate)", CALIB_MAX_AGE_DAYS)
        return 0
    if not video_paths:
        log.warning("no snippets to calibrate from yet — recorder needs to "
                    "capture some activity first; will retry on next schedule")
        return 1
    try:
        from ultralytics import YOLO  # noqa: PLC0415 - heavy, lazy, calibration-only
    except ImportError:
        log.error("calibration needs ultralytics (pip install ultralytics)")
        return 2

    # Gate the calibration clips through the SAME region the recorder gates on,
    # dashboard ROI first (that is what live blobs are measured in). With a polygon
    # ROI this matters: a blob learned unmasked would be larger than the one the
    # live gate ever sees, and the learned area window would sit too high.
    roi = load_roi_override_lores()
    roi_polygon = load_roi_polygon_lores() if roi is not None else None
    if roi is None:
        roi = _parse_roi()
    if roi_polygon:
        log.info("calibration: masking to the dashboard ROI polygon (%d corners)",
                 len(roi_polygon))
    log.info("calibration: loading YOLO %s (slow on a Pi 4)...", model_path)
    model = YOLO(model_path)

    bee_areas: list[float] = []
    bee_frames = 0
    clips_used = 0

    for vp in video_paths:
        if len(bee_areas) >= CALIB_TARGET_SAMPLES:
            break
        clips_used += 1
        # Fresh background per clip; warm on its first second of frames.
        gate = MotionGate(roi=roi, polygon=roi_polygon,
                          min_area=4.0, max_area=1e9, min_blobs=1)
        log.info("calibrating from %s (%d/%d samples so far)",
                 vp.name, len(bee_areas), CALIB_TARGET_SAMPLES)

        for fi, frame in enumerate(_iter_video_frames(vp)):
            gray = _lores_from_bgr(frame)
            if fi < FPS:                 # warmup window
                gate.warm(gray)
                continue
            gate.update(gray)            # populates gate.last_blobs (lores coords)
            if fi % CALIB_YOLO_EVERY:    # only YOLO 1 of every K frames
                continue

            res = model(frame, conf=YOLO_CONF, verbose=False)
            sx, sy = LORES_W / frame.shape[1], LORES_H / frame.shape[0]
            yolo_boxes = []
            for r in res:
                if r.boxes is None:
                    continue
                for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
                    yolo_boxes.append((x1 * sx, y1 * sy, x2 * sx, y2 * sy))

            if yolo_boxes:
                bee_frames += 1
            for (x, y, w, h, area) in gate.last_blobs:
                box = (x, y, x + w, y + h)
                if any(_bbox_overlap(box, yb) for yb in yolo_boxes):
                    bee_areas.append(area)
            if len(bee_areas) >= CALIB_TARGET_SAMPLES:
                break

    log.info("calibration scan: %d clips, %d bee-frames, %d blob samples",
             clips_used, bee_frames, len(bee_areas))

    if len(bee_areas) < CALIB_MIN_SAMPLES:
        log.error("only %d bee-blob samples (need >= %d) — keeping existing/"
                  "default thresholds, will retry next schedule",
                  len(bee_areas), CALIB_MIN_SAMPLES)
        return 1

    # Raw percentile window, then pad + clamp it (see config.py). MOG2 fragments a
    # bee into small blobs, so p95 sits near fragment size; widen both ends and
    # clamp to sane bounds so real bees (incl. close/merged blobs) aren't rejected.
    p5 = float(np.percentile(bee_areas, 5))
    p95 = float(np.percentile(bee_areas, 95))
    min_area = max(CALIB_AREA_FLOOR, p5 * CALIB_MIN_PAD)
    max_area = min(MAX_BLOB_AREA, p95 * CALIB_MAX_PAD)
    # Never let the window collapse: keep at least MIN_SPAN× headroom above min.
    max_area = max(max_area, min_area * CALIB_MIN_SPAN)
    calib = {
        "min_area": round(min_area, 1),
        "max_area": round(max_area, 1),
        "var_threshold": MOG2_VAR_THRESHOLD,
        "min_blobs": MIN_MOTION_BLOBS,
        # Provenance for the dashboard: raw percentiles before pad/clamp.
        "raw_p5": round(p5, 1),
        "raw_p95": round(p95, 1),
        "n_samples": len(bee_areas),
        "n_clips": clips_used,
        "lores": [LORES_W, LORES_H],
        "model": model_path,
    }
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-rename so a recorder reading concurrently never sees a partial file.
    tmp = CALIB_FILE.with_suffix(".json.part")
    tmp.write_text(json.dumps(calib, indent=2))
    tmp.replace(CALIB_FILE)
    log.info("calibration saved -> %s  area=[%.0f, %.0f] (raw p5/p95 [%.0f, %.0f]) "
             "from %d samples", CALIB_FILE, min_area, max_area, p5, p95, len(bee_areas))
    return 0
