"""Full-resolution stills between clips (memory/40).

The recorder writes 1080p video: the Pi 4's H.264 encoder tops out there, and
the OV64A40 reads its whole 9152x6944 sensor at only ~2.6 fps. A still is not
limited by the encoder, so every N minutes — and only while no clip is open —
the recorder pauses video (~2 s), switches the camera to its full sensor size,
takes one frame, and switches back. Encoding the JPEG happens afterwards on a
thread, while video runs again.

Each still is written as ``<stamp>.jpg`` + ``<stamp>.thumb.jpg`` (1280 px) and
then ``<stamp>.json``, last, as the "complete" marker the uploader waits for.
The uploader sends them like videos (WiFi only) and deletes them after.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from motion.config import (
    log, STILLS_DIR, STILL_REQUEST_FILE, STILLS_MAX_BYTES,
    STILL_JPEG_QUALITY, STILL_THUMB_SIDE,
)

MIN_INTERVAL_MIN = 15
# The 16 MP binned mode, for when a 64 MP buffer cannot be allocated.
FALLBACK_SIZE = (4624, 3472)


def interval_seconds(minutes) -> float:
    """0 = off; anything else floored at 15 minutes."""
    try:
        m = int(minutes or 0)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if m <= 0 else max(m, MIN_INTERVAL_MIN) * 60.0


def due(now: float, next_at: float, interval_s: float, *, encoding: bool,
        in_window: bool, recording_on: bool, requested: bool) -> bool:
    """Take a still now? Never mid-clip. A scheduled one also needs recording
    on and the hour window; a requested one ("take one now") does not."""
    if encoding:
        return False
    if requested:
        return True
    return bool(interval_s) and recording_on and in_window and now >= next_at


def requested() -> bool:
    return STILL_REQUEST_FILE.exists()


def clear_request() -> None:
    try:
        STILL_REQUEST_FILE.unlink()
    except OSError:
        pass


def take(cam, encoder, transform, lens, source: str = "schedule") -> bool:
    """Pause video, capture the full sensor, resume. True if a still was taken.

    The encoder is stopped for the mode switch and restarted after, the same
    way the recorder starts it; switch_mode_and_capture_array puts the video
    configuration back. The lens is held where the recorder focused it.
    """
    from libcamera import controls

    taken_at = datetime.now(timezone.utc)
    t0 = time.monotonic()
    arr, mode = None, ""
    cam.stop_encoder()
    try:
        for size, label in ((tuple(cam.sensor_resolution), "64mp"), (FALLBACK_SIZE, "16mp")):
            try:
                cfg = cam.create_still_configuration(
                    main={"size": size, "format": "RGB888"}, buffer_count=1,
                    transform=transform)
                if lens is not None:
                    cfg["controls"]["AfMode"] = controls.AfModeEnum.Manual
                    cfg["controls"]["LensPosition"] = float(lens)
                arr = cam.switch_mode_and_capture_array(cfg, "main")
                mode = label
                break
            except Exception as e:  # most likely CMA: fall back to 16 MP
                log.warning("still: %s capture failed (%s)", label, e)
    finally:
        cam.start_encoder(encoder)
        if lens is not None:
            try:
                cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                  "LensPosition": float(lens)})
            except Exception as e:
                log.warning("still: could not restore focus: %s", e)
    pause = time.monotonic() - t0
    if arr is None:
        log.warning("still: none taken; video paused %.1fs", pause)
        return False
    log.info("still: %dx%d (%s) taken, video paused %.1fs",
             arr.shape[1], arr.shape[0], mode, pause)
    threading.Thread(target=_write, args=(arr, taken_at, mode, lens, source),
                     daemon=True).start()
    return True


def _write(arr, taken_at: datetime, mode: str, lens, source: str) -> None:
    """JPEG + 1280 px preview + the marker, then keep the folder under its cap.
    "RGB888" arrays are BGR-ordered, which is what cv2 writes."""
    try:
        STILLS_DIR.mkdir(parents=True, exist_ok=True)
        stem = STILLS_DIR / taken_at.strftime("%Y-%m-%d_%H_%M_%S")
        h, w = arr.shape[:2]
        ok, full = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, STILL_JPEG_QUALITY])
        scale = STILL_THUMB_SIDE / float(max(w, h))
        small = cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok2, thumb = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        del arr, small
        if not (ok and ok2):
            log.warning("still: JPEG encode failed")
            return
        Path(str(stem) + ".jpg").write_bytes(full.tobytes())
        Path(str(stem) + ".thumb.jpg").write_bytes(thumb.tobytes())
        meta = {"taken_at": taken_at.isoformat(), "width": w, "height": h,
                "sensor_mode": mode, "lens_position": lens, "source": source}
        tmp = Path(str(stem) + ".json.tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, Path(str(stem) + ".json"))   # the marker comes last
        log.info("still: saved %s.jpg (%.1f MB)", stem.name, len(full) / 1e6)
        enforce_cap()
    except Exception as e:  # never take the recorder down over a still
        log.warning("still: write failed: %s", e)


def enforce_cap(max_bytes: int = STILLS_MAX_BYTES) -> int:
    """Delete the oldest stills until the folder fits. Returns how many went."""
    groups = sorted(STILLS_DIR.glob("*.json"), key=lambda p: p.name)
    sizes = {}
    for meta in groups:
        stem = str(meta)[:-5]
        sizes[meta] = sum(p.stat().st_size for p in
                          (Path(stem + ".jpg"), Path(stem + ".thumb.jpg"), meta) if p.exists())
    total, dropped = sum(sizes.values()), 0
    for meta in groups:
        if total <= max_bytes:
            break
        stem = str(meta)[:-5]
        for p in (Path(stem + ".jpg"), Path(stem + ".thumb.jpg"), meta):
            try:
                p.unlink()
            except OSError:
                pass
        total -= sizes[meta]
        dropped += 1
    if dropped:
        log.warning("still: dropped %d oldest still(s) to stay under %d MB",
                    dropped, max_bytes // (1024 * 1024))
    return dropped
