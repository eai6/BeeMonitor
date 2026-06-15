"""Activity-frame sampling: crop the mover for taxonomic ID (queued for cellular).

Per recorded activity (motion clip) we sample a few crops of the strongest-motion
blob and queue them for the telemetry service; BioCLIP identifies the insect in
the cloud. Crops only (tiny bytes) — the WiFi-gated full video is unaffected.
See memory/15_monitoring_agent_design.md.
"""

from __future__ import annotations

import json

import cv2

from motion.config import (
    log, ACTIVITY_FRAMES_QUEUE, FRAMES_PER_ACTIVITY,
    FRAME_CROP_PAD, FRAME_MAX_SIDE, LORES_W, LORES_H,
)
from motion.frames import _scale_roi


def _largest_blob(blobs):
    """The (x, y, w, h, area) blob with the greatest area, or None."""
    return max(blobs, key=lambda b: b[4]) if blobs else None


def _mover_crop(main_bgr, blob, roi):
    """Crop the mover out of the main BGR frame.

    ``blob`` is (x, y, w, h, area) in *lores* pixels relative to the gate ROI;
    we add the ROI origin, scale lores->main, pad for context, and crop. Returns
    ``(jpg_bytes, bbox_norm, (w, h))`` or None — bbox_norm is the padded crop box
    in main-frame normalized (0..1) coords.
    """
    h, w = main_bgr.shape[:2]
    bx, by, bw, bh, _area = blob
    ox, oy = (roi[0], roi[1]) if roi is not None else (0, 0)
    x1, y1, x2, y2 = _scale_roi(
        (ox + bx, oy + by, ox + bx + bw, oy + by + bh),
        (LORES_W, LORES_H), (w, h))
    pad_x = int((x2 - x1) * FRAME_CROP_PAD)
    pad_y = int((y2 - y1) * FRAME_CROP_PAD)
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = main_bgr[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    longest = max(ch, cw)
    if longest > FRAME_MAX_SIDE:
        s = FRAME_MAX_SIDE / longest
        crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))),
                          interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        return None
    bbox_norm = [round(x1 / w, 5), round(y1 / h, 5), round(x2 / w, 5), round(y2 / h, 5)]
    return buf.tobytes(), bbox_norm, (crop.shape[1], crop.shape[0])


def _flush_activity_frames(uid, started_epoch, candidates):
    """Write the top FRAMES_PER_ACTIVITY candidate crops + sidecars to the queue.

    Each frame is a ``<uid>_<i>.jpg`` plus a ``<uid>_<i>.json`` sidecar; the JSON
    is written last so the telemetry drainer only sees a complete pair. Telemetry
    enforces the daily cellular cap; we just queue the best of this activity.
    """
    if not candidates:
        return
    top = sorted(candidates, key=lambda c: c["area"], reverse=True)[:FRAMES_PER_ACTIVITY]
    peak = top[0]["area"]
    try:
        ACTIVITY_FRAMES_QUEUE.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("activity frames: cannot create queue dir: %s", e)
        return
    written = 0
    for i, c in enumerate(top):
        base = ACTIVITY_FRAMES_QUEUE / f"{uid}_{i}"
        meta = {
            "activity_uid": uid,
            "started_at": started_epoch,
            "captured_at": c["captured_at"],
            "bbox": c["bbox"],
            "motion_score": c["area"],
            "peak_motion": peak,
            "kind": "crop",
            "width": c["wh"][0],
            "height": c["wh"][1],
        }
        try:
            base.with_suffix(".jpg").write_bytes(c["jpg"])
            base.with_suffix(".json").write_text(json.dumps(meta))
            written += 1
        except OSError as e:
            log.warning("activity frames: write failed for %s: %s", uid, e)
    if written:
        log.info("activity frames: queued %d crop(s) for %s", written, uid)
