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

On a Luxonis OAK (``take_oak`` / ``take_burst_oak``) none of the pausing
applies: the OAK JPEG-encodes a full-size frame of the running video on
request (motion/oak.py), so a photo is the video's own resolution — 12 MP on
the OAK-1-AF — the video never stops, and the pre-roll survives a burst. The
OAK's JPEG is saved as it arrives rather than decoded and re-encoded.
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

class CameraStuck(RuntimeError):
    """The camera could not be put back into video mode after a still.

    Fatal on purpose: the recorder must NOT carry on, because its next read of
    the motion stream would wait forever on a camera that is stopped in still
    mode — which is how a failed 64 MP capture once left a unit silently not
    recording (and deaf to SIGTERM) for an hour. The recorder exits on this and
    systemd restarts it with a fresh camera.
    """


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


def _resume_video(cam, encoder, video_config, output, lens) -> None:
    """Back to video after a still, whatever state the still left the camera in.

    Always switches to `video_config` explicitly. picamera2's switch-back
    helpers return to whatever mode the camera was in when they were called —
    after a failed 64 MP attempt that is the 64 MP still mode itself, and the
    16 MP fallback then "restores" into it and fails. If switching fails, a full
    stop / configure / start is the second try. Raises CameraStuck if neither
    brings video back.
    """
    last = None
    for attempt in ("switch", "restart"):
        try:
            if attempt == "switch":
                cam.switch_mode(video_config)
            else:
                cam.stop()
                cam.configure(video_config)
                cam.start()
            if output is not None:
                encoder.output = output
            cam.start_encoder(encoder)
            break
        except Exception as e:
            last = e
            log.error("still: back to video by %s failed: %s", attempt, e)
    else:
        raise CameraStuck(f"could not restore video after a still: {last}")
    _restore_focus(cam, lens)


def _restore_focus(cam, lens) -> None:
    if lens is None:
        return
    from libcamera import controls
    try:
        cam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(lens)})
    except Exception as e:
        log.warning("still: could not restore focus: %s", e)


class Burst:
    """A motion burst's captured frames, saved once the clip after it is named.

    ``save(clip)`` writes them on a thread (in order; each raw 64 MP frame is
    ~190 MB, so they are encoded and released one at a time), each marker
    naming the clip so the server shows the stills on that clip's page.
    """

    def __init__(self, frames, mode, lens, burst_id):
        self.frames, self.mode, self.lens, self.burst_id = frames, mode, lens, burst_id
        self.count = len(frames)

    def save(self, clip: str = "") -> None:
        if not self.frames:
            return
        frames, self.frames = self.frames, []

        def _write_all(items=frames):
            index = 0
            while items:
                at, arr, *jpeg = items.pop(0)   # release each frame once written
                _write(arr, at, self.mode, self.lens, "burst", burst_id=self.burst_id,
                       burst_index=index, clip=clip, jpeg=jpeg[0] if jpeg else None)
                index += 1
        threading.Thread(target=_write_all, daemon=True).start()


def take_burst(cam, encoder, video_config, transform, lens, count: int,
               output=None) -> "Burst":
    """On a motion trigger: ``count`` full-sensor frames in a row, then back to
    video. One mode switch each way (not one per frame). Returns the Burst;
    the caller opens the clip, then calls ``burst.save(clip_name)``.

    A burst keeps no video from before the trigger. Stopping the encoder does
    NOT empty the pre-roll buffer, so the caller passes a fresh ``output``
    (an empty CircularOutput) to restart on — otherwise the clip opened next
    begins with the 2 s from before the burst and jumps past the stills.
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
        _resume_video(cam, encoder, video_config, output, lens)
    log.info("burst: %d x %s in %.1fs (video paused)", len(frames), mode or "none",
             time.monotonic() - t0)
    return Burst(frames, mode, lens, burst_id)


def take(cam, encoder, video_config, transform, lens, source: str = "schedule",
         output=None) -> bool:
    """Pause video, capture the full sensor, resume. True if a still was taken.

    The encoder is stopped for the mode switch and restarted after, the same
    way the recorder starts it, and video_config is put back explicitly
    (_resume_video) — raising CameraStuck if it cannot be. The lens is held
    where the recorder focused it.
    """
    taken_at = _now()
    t0 = time.monotonic()
    arr, mode = None, ""
    cam.stop_encoder()
    try:
        for size, label in ((tuple(cam.sensor_resolution), "64mp"), (FALLBACK_SIZE, "16mp")):
            try:
                cam.switch_mode(_still_config(cam, size, transform, lens))
                arr = cam.capture_array("main")
                mode = label
                break
            except Exception as e:  # most likely CMA: fall back to 16 MP
                log.warning("still: %s capture failed (%s)", label, e)
    finally:
        # output: see take_burst — no stale pre-roll after a pause.
        _resume_video(cam, encoder, video_config, output, lens)
    pause = time.monotonic() - t0
    if arr is None:
        log.warning("still: none taken; video paused %.1fs", pause)
        return False
    log.info("still: %dx%d (%s) taken, video paused %.1fs",
             arr.shape[1], arr.shape[0], mode, pause)
    threading.Thread(target=_write, args=(arr, taken_at, mode, lens, source),
                     daemon=True).start()
    return True


def take_oak(cam, lens, source: str = "schedule") -> bool:
    """A full-size photo from an OAK, video running throughout."""
    try:
        (taken_at, jpeg), = cam.capture_jpegs(1)
    except Exception as e:
        log.warning("still: OAK capture failed: %s", e)
        return False
    mode = _oak_mode(cam)
    log.info("still: %s taken (OAK, video not paused)", mode)
    threading.Thread(target=_write, args=(None, taken_at, mode, lens, source),
                     kwargs={"jpeg": jpeg}, daemon=True).start()
    return True


def take_burst_oak(cam, lens, count: int) -> "Burst":
    """``count`` consecutive full-size frames from an OAK, video running."""
    taken_at = _now()
    burst_id = taken_at.strftime("%Y%m%d%H%M%S") + "-" + os.urandom(2).hex()
    t0 = time.monotonic()
    try:
        frames = [(at, None, jpeg) for at, jpeg in cam.capture_jpegs(count)]
    except Exception as e:
        log.warning("burst: OAK capture failed: %s", e)
        frames = []
    mode = _oak_mode(cam)
    log.info("burst: %d x %s in %.2fs (OAK, video not paused)", len(frames), mode,
             time.monotonic() - t0)
    return Burst(frames, mode, lens, burst_id)


def _oak_mode(cam) -> str:
    w, h = cam.main_wh
    return f"{round(w * h / 1e6)}mp"


def _write(arr, taken_at: datetime, mode: str, lens, source: str,
           burst_id: str = "", burst_index: int | None = None, clip: str = "",
           jpeg: bytes | None = None) -> None:
    """JPEG + 1280 px preview + the marker (written last).
    "RGB888" arrays are BGR-ordered, which is what cv2 writes. With `jpeg`
    (an OAK photo) those bytes are the full-size file; arr may be None."""
    try:
        STILLS_DIR.mkdir(parents=True, exist_ok=True)
        if jpeg is not None and arr is None:
            import numpy as np
            arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                log.warning("still: undecodable JPEG from the camera")
                return
        name = taken_at.strftime("%Y-%m-%d_%H_%M_%S")
        if burst_id:
            name += f"_b{burst_index}"   # a burst's frames share their second
        stem = STILLS_DIR / name
        h, w = arr.shape[:2]
        if jpeg is not None:
            ok, full = True, jpeg
        else:
            ok, full = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, STILL_JPEG_QUALITY])
        scale = STILL_THUMB_SIDE / float(max(w, h))
        small = cv2.resize(arr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        ok2, thumb = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        del arr, small
        if not (ok and ok2):
            log.warning("still: JPEG encode failed")
            return
        Path(str(stem) + ".jpg").write_bytes(bytes(full))
        Path(str(stem) + ".thumb.jpg").write_bytes(thumb.tobytes())
        meta = {"taken_at": taken_at.isoformat(), "width": w, "height": h,
                "sensor_mode": mode, "lens_position": lens, "source": source}
        if burst_id:
            meta.update(burst_id=burst_id, burst_index=burst_index, clip=clip)
        tmp = Path(str(stem) + ".json.tmp")
        with open(tmp, "w") as fh:   # fsync, or a power cut can leave a 0-byte marker
            fh.write(json.dumps(meta))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, Path(str(stem) + ".json"))   # the marker comes last
        log.info("still: saved %s.jpg (%.1f MB)", stem.name, len(full) / 1e6)
    except Exception as e:  # never take the recorder down over a still
        log.warning("still: write failed: %s", e)
