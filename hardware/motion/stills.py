"""Full-resolution stills between clips (memory/40).

The recorder writes 1080p video: the Pi 4's H.264 encoder tops out there, and
the OV64A40 reads its whole 9152x6944 sensor at only ~2.6 fps. A still is not
limited by the encoder, so every N minutes — and only while no clip is open —
the recorder pauses video (~2 s), switches the camera to its full sensor size,
takes one frame, and switches back. Encoding the JPEG happens afterwards on a
thread, while video runs again.

A motion burst (``take_burst``) is the same thing on a trigger: one switch,
N consecutive frames, switch back — then the recorder opens the clip.

Each still is written as ``<stamp>.jpg`` + ``<stamp>.thumb.jpg`` (1280 px) and
then ``<stamp>.json``, last, as the "complete" marker the uploader waits for.
The uploader sends them like videos (WiFi only) and, like videos, leaves them
on the card (``<stamp>.json.uploaded`` records the cloud id) until someone
clears them on the dashboard — telemetry's cleanup pass then deletes them.

Times follow the clips' convention: the Pi's wall clock labelled UTC, so a
burst and the clip after it can be matched on the server.
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
    log, STILLS_DIR, STILL_REQUEST_FILE, STILL_JPEG_QUALITY, STILL_THUMB_SIDE,
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


def _now() -> datetime:
    """The clips' convention: wall clock, labelled UTC (remux._snippet_paths)."""
    return datetime.now().replace(tzinfo=timezone.utc)


def _still_config(cam, size, transform, lens):
    cfg = cam.create_still_configuration(
        main={"size": size, "format": "RGB888"}, buffer_count=1, transform=transform)
    if lens is not None:   # a fixed-focus module has none to hold
        from libcamera import controls
        cfg["controls"]["AfMode"] = controls.AfModeEnum.Manual
        cfg["controls"]["LensPosition"] = float(lens)
    return cfg


def _restore_focus(cam, lens) -> None:
    if lens is None:
        return
    from libcamera import controls
    try:
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(lens)})
    except Exception as e:
        log.warning("still: could not restore focus: %s", e)


def take_burst(cam, encoder, video_config, transform, lens, count: int) -> int:
    """On a motion trigger: ``count`` full-sensor frames in a row, then back to
    video. One mode switch each way (not one per frame). Returns frames taken.

    The encoder is stopped for the switch, which discards the pre-roll — by
    design, a burst keeps no video from before the trigger. The caller opens
    the clip straight after.
    """
    taken_at = _now()
    burst_id = taken_at.strftime("%Y%m%d%H%M%S") + "-" + os.urandom(2).hex()
    t0 = time.monotonic()
    frames, mode = [], ""
    cam.stop_encoder()
    try:
        for size, label in ((tuple(cam.sensor_resolution), "64mp"), (FALLBACK_SIZE, "16mp")):
            try:
                cam.switch_mode(_still_config(cam, size, transform, lens))
                mode = label
                break
            except Exception as e:  # most likely CMA: fall back to 16 MP
                log.warning("burst: %s mode failed (%s)", label, e)
        if mode:
            for _ in range(count):
                try:
                    frames.append((_now(), cam.capture_array("main")))
                except Exception as e:
                    log.warning("burst: frame %d failed: %s", len(frames) + 1, e)
                    break
    finally:
        try:
            cam.switch_mode(video_config)
        finally:
            cam.start_encoder(encoder)
            _restore_focus(cam, lens)
    taken = len(frames)   # counted now: the writer thread empties the list
    log.info("burst: %d x %s in %.1fs (video paused)", taken, mode or "none",
             time.monotonic() - t0)
    if frames:
        # One writer thread, frames in order: each raw 64 MP frame is ~190 MB,
        # so they are encoded and released one at a time.
        def _write_all(items=frames):
            index = 0
            while items:
                at, arr = items.pop(0)   # release each frame once written
                _write(arr, at, mode, lens, "burst", burst_id=burst_id, burst_index=index)
                index += 1
        threading.Thread(target=_write_all, daemon=True).start()
    return taken


def take(cam, encoder, transform, lens, source: str = "schedule") -> bool:
    """Pause video, capture the full sensor, resume. True if a still was taken.

    The encoder is stopped for the mode switch and restarted after, the same
    way the recorder starts it; switch_mode_and_capture_array puts the video
    configuration back. The lens is held where the recorder focused it.
    """
    taken_at = _now()
    t0 = time.monotonic()
    arr, mode = None, ""
    cam.stop_encoder()
    try:
        for size, label in ((tuple(cam.sensor_resolution), "64mp"), (FALLBACK_SIZE, "16mp")):
            try:
                arr = cam.switch_mode_and_capture_array(
                    _still_config(cam, size, transform, lens), "main")
                mode = label
                break
            except Exception as e:  # most likely CMA: fall back to 16 MP
                log.warning("still: %s capture failed (%s)", label, e)
    finally:
        cam.start_encoder(encoder)
        _restore_focus(cam, lens)
    pause = time.monotonic() - t0
    if arr is None:
        log.warning("still: none taken; video paused %.1fs", pause)
        return False
    log.info("still: %dx%d (%s) taken, video paused %.1fs",
             arr.shape[1], arr.shape[0], mode, pause)
    threading.Thread(target=_write, args=(arr, taken_at, mode, lens, source),
                     daemon=True).start()
    return True


def _write(arr, taken_at: datetime, mode: str, lens, source: str,
           burst_id: str = "", burst_index: int | None = None) -> None:
    """JPEG + 1280 px preview + the marker (written last).
    "RGB888" arrays are BGR-ordered, which is what cv2 writes."""
    try:
        STILLS_DIR.mkdir(parents=True, exist_ok=True)
        name = taken_at.strftime("%Y-%m-%d_%H_%M_%S")
        if burst_id:
            name += f"_b{burst_index}"   # a burst's frames share their second
        stem = STILLS_DIR / name
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
        if burst_id:
            meta.update(burst_id=burst_id, burst_index=burst_index)
        tmp = Path(str(stem) + ".json.tmp")
        tmp.write_text(json.dumps(meta))
        os.replace(tmp, Path(str(stem) + ".json"))   # the marker comes last
        log.info("still: saved %s.jpg (%.1f MB)", stem.name, len(full) / 1e6)
    except Exception as e:  # never take the recorder down over a still
        log.warning("still: write failed: %s", e)
