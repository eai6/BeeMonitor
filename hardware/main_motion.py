"""Motion-gated recorder for BeeMonitor (cellular-friendly).

Drop-in alternative to ``main.py`` that records **only activity snippets**
instead of fixed 10-minute chunks, so a cellular uplink doesn't have to push
hours of empty footage.che

How it works
------------
``picamera2`` runs two streams at once (same trick ``main.py`` already uses):

    * ``main``  — full-res H.264, fed into a ``CircularOutput`` ring buffer that
      is *always* encoding but only flushed to disk when there's activity. The
      ring buffer is what gives us **pre-roll** (the seconds *before* motion).
    * ``lores`` — a small grayscale stream we run MOG2 background subtraction on
      to decide "is something moving?". This is the cheap half of BeeMonitor's
      detection stack (ported from ``src/beemonitor/detection/blob_detector.py``);
      YOLO/tracking stay in the cloud, run later on the uploaded snippets.

Segment lifecycle:
    motion starts  -> open a new .h264 segment (ring buffer flushes pre-roll)
    motion ongoing -> keep writing
    motion stops   -> after POST_ROLL seconds, close + remux to .mp4
    (safety)       -> a segment is force-rotated after MAX_SEGMENT seconds

Snippets are written as ``RECORD_DIR/YYYY-MM-DD/YYYY-MM-DD_HH_MM_SS.mp4`` — the
exact convention ``uploader.py`` already scans for, so the existing uploader
service ships them unchanged.

IMPORTANT — on-device gating means a missed detection is permanent data loss.
Defaults here favour *over*-triggering. Validate with the logged stats
(triggers/hour, snippet length) before trusting it in the field, then tighten
thresholds.

Calibrating the thresholds
--------------------------
The gate's bee-sized blob window is set the same way cloud BeeMonitor sets it —
*from detected bees* — but on the Pi itself, once. ``--calibrate`` runs the YOLO
model on a few hundred frames (slow on a Pi 4, but it's a one-time pass),
measures the MOG2 blob area of every YOLO-confirmed bee, and writes the
5th/95th-percentile area window to ``calibration.json``. The recorder loads that
file and then runs **MOG2 only** — YOLO never runs in the recording hot path.

    # One-time, on the Pi, with bees active in frame (takes a few minutes):
    python3 main_motion.py --calibrate --model bee_yolo.pt

    # Production: record activity snippets (loads calibration.json if present):
    python3 main_motion.py

Run the recorder directly (``python3 main_motion.py``) or point a systemd unit at
it the same way ``driver.py`` launches ``main.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - Pi image ships opencv-python
    print("main_motion requires opencv-python (cv2)", file=sys.stderr)
    raise

# picamera2 only exists on the Pi. Keep the import soft so the module can be
# imported off-device (e.g. for linting); record()/calibrate() require it.
try:
    import libcamera
    from picamera2 import MappedArray, Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import CircularOutput
    HAVE_PICAMERA2 = True
except ImportError:  # pragma: no cover - not on a Pi
    HAVE_PICAMERA2 = False


# ---------------------------------------------------------------------------
# Config (env-overridable, mirrors uploader.py's BEEMONITOR_* convention)
# ---------------------------------------------------------------------------

def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))

def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# Where finished .mp4 snippets land — keep == uploader's BEEMONITOR_RECORD_DIR.
RECORD_DIR = Path(os.environ.get(
    "BEEMONITOR_RECORD_DIR", "/home/beemonitor/Desktop/cameraOutput/beeHotel"))
# Scratch dir for the raw .h264 segments before remux (kept off the watched tree).
WORK_DIR = Path(os.environ.get("BEEMONITOR_WORK_DIR", "/home/beemonitor/Desktop/cameraOutput/_work"))

# Capture geometry.
MAIN_W = _env_int("BEEMONITOR_MAIN_W", 1920)
MAIN_H = _env_int("BEEMONITOR_MAIN_H", 1080)
# lores is the *detection* stream. 640x480 gives bee-sized blobs room to be
# seen; 320x240 (what main.py used) is borderline. Keep width a multiple of 32
# so the Y-plane slice below has no row padding.
LORES_W = _env_int("BEEMONITOR_LORES_W", 640)
LORES_H = _env_int("BEEMONITOR_LORES_H", 480)
FPS = _env_int("BEEMONITOR_FPS", 25)

# Clip timing (seconds).
PRE_ROLL = _env_float("BEEMONITOR_PRE_ROLL", 3.0)
POST_ROLL = _env_float("BEEMONITOR_POST_ROLL", 4.0)
MAX_SEGMENT = _env_float("BEEMONITOR_MAX_SEGMENT", 120.0)   # force-rotate cap
WARMUP_SECONDS = _env_float("BEEMONITOR_WARMUP", 5.0)        # let MOG2 learn bg

# Telemetry stills queue. Periodic capture is OFF by default now — telemetry is
# JSON-only over cellular; stills are captured ON DEMAND (picture / live view).
# Set BEEMONITOR_TELEMETRY_IMAGE_INTERVAL > 0 to re-enable periodic stills.
TELEMETRY_QUEUE = Path(os.environ.get(
    "BEEMONITOR_TELEMETRY_QUEUE", str(RECORD_DIR.parent / "telemetry")))
TELEMETRY_IMAGE_INTERVAL = _env_float("BEEMONITOR_TELEMETRY_IMAGE_INTERVAL", 0.0)
TELEMETRY_IMAGE_HEIGHT = _env_int("BEEMONITOR_TELEMETRY_IMAGE_HEIGHT", 720)

# Detection cost knob: run MOG2 on 1 of every N lores frames (timing stays
# wall-clock based, so this only trades latency for CPU).
DETECT_EVERY_N = max(1, _env_int("BEEMONITOR_DETECT_EVERY_N", 2))

# MOG2 / blob params — ported from BlobDetector defaults, scaled for lores.
MOG2_HISTORY = _env_int("BEEMONITOR_MOG2_HISTORY", 500)
MOG2_VAR_THRESHOLD = _env_int("BEEMONITOR_MOG2_VAR", 16)
# Treat cast shadows and soft illumination changes as background, not motion.
# With this on, MOG2 classifies shadow pixels (same chromaticity as background
# but darker — moving shadows, the hotel's own shadow drifting with the sun,
# passing clouds) separately, and we threshold them out before counting blobs.
# This is the main lever against shadow / background-light false triggers.
DETECT_SHADOWS = _env_bool("BEEMONITOR_DETECT_SHADOWS", True)
# MOG2 shadow tau (0..1): a pixel is shadow if its intensity is within
# [tau*bg, bg]. Lower = more aggressive shadow rejection. 0.5 is OpenCV default.
SHADOW_THRESHOLD = _env_float("BEEMONITOR_SHADOW_THRESHOLD", 0.5)
# Periodically rebuild the background model from scratch so the gate tracks
# slow scene changes (sun/shadow drift, a moved leaf) in real time instead of
# letting them bleed in through MOG2's long history. The model is also fresh at
# recorder start (the gate is constructed then). 0 disables periodic rebuilds.
BG_RESET_INTERVAL = _env_float("BEEMONITOR_BG_RESET_INTERVAL", 600.0)  # 10 min
MORPH_KERNEL = _env_int("BEEMONITOR_MORPH_KERNEL", 5)
MORPH_ITERS = _env_int("BEEMONITOR_MORPH_ITERS", 2)
# Blob area filters in *lores* pixels. Defaults are deliberately permissive.
MIN_BLOB_AREA = _env_float("BEEMONITOR_MIN_BLOB_AREA", 20.0)
MAX_BLOB_AREA = _env_float("BEEMONITOR_MAX_BLOB_AREA", 5000.0)
# How many qualifying blobs constitute "motion".
MIN_MOTION_BLOBS = _env_int("BEEMONITOR_MIN_MOTION_BLOBS", 1)

# Optional detection ROI in lores coords: "x1,y1,x2,y2" (e.g. the hotel face).
# Empty (the default) => auto-detect the hotel with nest_detection.pt at startup,
# exactly like cloud BeeMonitor; falls back to the whole frame if detection fails.
ROI = os.environ.get("BEEMONITOR_ROI", "").strip()

# ML models. Default to the repo's models/ committed next to hardware/ so the Pi
# uses the SAME weights as cloud BeeMonitor with no env to set — we are NOT using
# the stock yolo11n any more. Override the dir or individual paths via env.
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR = Path(os.environ.get("BEEMONITOR_MODELS_DIR", str(_DEFAULT_MODELS_DIR)))

# nest_detection.pt — hotel/nest detector. Defines the recording ROI (the hotel)
# the same way cloud BeeMonitor does: class 0 = hotel, class 1 = nest hole.
NEST_MODEL = os.environ.get("BEEMONITOR_NEST_MODEL", str(MODELS_DIR / "nest_detection.pt"))
NEST_CONF = _env_float("BEEMONITOR_NEST_CONF", 0.25)
# Run hotel detection before recording to set the ROI. Off => whole frame.
HOTEL_ROI_DETECT = _env_bool("BEEMONITOR_HOTEL_ROI_DETECT", True)
# Padding around the detected hotel, base px @ 1920x1080, scaled to capture res
# (mirrors NestConfig.hotel_padding_x/y_base in cloud config.py).
HOTEL_PAD_X_BASE = _env_float("BEEMONITOR_HOTEL_PAD_X", 100.0)
HOTEL_PAD_Y_BASE = _env_float("BEEMONITOR_HOTEL_PAD_Y", 50.0)
# Seconds to let auto-exposure settle before grabbing the hotel-detection frame.
HOTEL_SETTLE_SECONDS = _env_float("BEEMONITOR_HOTEL_SETTLE", 2.0)

# Calibration: where the blob-area window is stored, and the bee detector used by
# the (scheduled, offline) calibration pass over saved snippets. bee_tracking.pt
# is BeeMonitor's own bee/wasp detector — every box is a bee, so calibration no
# longer needs a COCO class filter.
CALIB_FILE = Path(os.environ.get(
    "BEEMONITOR_CALIB_FILE", str(RECORD_DIR.parent / "calibration.json")))
YOLO_MODEL = os.environ.get("BEEMONITOR_YOLO_MODEL", str(MODELS_DIR / "bee_tracking.pt"))
YOLO_CONF = _env_float("BEEMONITOR_YOLO_CONF", 0.25)
# Stop once we've measured this many confirmed-bee blobs across snippets.
# MIN_SAMPLES is the floor below which we refuse to overwrite a calibration.
CALIB_TARGET_SAMPLES = _env_int("BEEMONITOR_CALIB_SAMPLES", 40)
CALIB_MIN_SAMPLES = _env_int("BEEMONITOR_CALIB_MIN_SAMPLES", 12)
# Auto mode scans the newest N snippets; run YOLO on 1 of every K frames.
CALIB_MAX_CLIPS = _env_int("BEEMONITOR_CALIB_MAX_CLIPS", 20)
CALIB_YOLO_EVERY = max(1, _env_int("BEEMONITOR_CALIB_YOLO_EVERY", 3))
# Skip re-calibrating if calibration.json is younger than this (days). --force overrides.
CALIB_MAX_AGE_DAYS = _env_float("BEEMONITOR_CALIB_MAX_AGE_DAYS", 7.0)
# Recorder re-reads calibration.json this often (s) so a scheduled calibration
# is picked up without a restart.
CALIB_RELOAD_SECONDS = _env_float("BEEMONITOR_CALIB_RELOAD_SECONDS", 300.0)

# Burn a timestamp into the recorded (main) frames like main.py does.
TIMESTAMP_OVERLAY = _env_bool("BEEMONITOR_TIMESTAMP", True)


logging.basicConfig(
    format="%(asctime)s %(levelname)s recorder %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("recorder")


# ---------------------------------------------------------------------------
# Motion gate (slim MOG2 blob detector, lores Y-plane only)
# ---------------------------------------------------------------------------

class MotionGate:
    """MOG2 background subtraction + contour-area filtering -> motion?

    Faithful to ``BlobDetector`` but self-contained (cv2 + numpy only) and
    operating on a single-channel grayscale frame, so it pulls in none of the
    YOLO/torch deps that ``beemonitor.detection`` would.

    The three tunable thresholds live here as instance attributes so the
    dry-run visualiser can slide them live:
        * ``var_threshold`` — MOG2 per-pixel foreground sensitivity
        * ``min_area`` / ``max_area`` — per-blob size window (lores pixels)
        * ``min_blobs`` — how many qualifying blobs == "motion"
    After each ``update`` the last mask + kept blobs are stashed on the
    instance (``last_mask``, ``last_blobs``) for visualisation.
    """

    def __init__(self, roi=None, history=MOG2_HISTORY,
                 var_threshold=MOG2_VAR_THRESHOLD, morph_kernel=MORPH_KERNEL,
                 morph_iters=MORPH_ITERS, min_area=MIN_BLOB_AREA,
                 max_area=MAX_BLOB_AREA, min_blobs=MIN_MOTION_BLOBS,
                 detect_shadows=DETECT_SHADOWS, shadow_threshold=SHADOW_THRESHOLD):
        self.var_threshold = var_threshold
        self.history = history
        self.detect_shadows = detect_shadows
        self.shadow_threshold = shadow_threshold
        self.bg = self._make_bg()
        self._morph_kernel = morph_kernel
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        self.morph_iters = morph_iters
        self.min_area = min_area
        self.max_area = max_area
        self.min_blobs = min_blobs
        self.roi = roi  # (x1, y1, x2, y2) in lores coords, or None
        self.last_mask = None
        self.last_blobs = []   # list of (x, y, w, h, area) for kept blobs

    def _make_bg(self):
        """Build a MOG2 subtractor with the current shadow/threshold settings."""
        bg = cv2.createBackgroundSubtractorMOG2(
            history=self.history, varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows)
        if self.detect_shadows:
            bg.setShadowThreshold(self.shadow_threshold)
        return bg

    def set_var_threshold(self, value: float) -> None:
        self.var_threshold = value
        self.bg.setVarThreshold(value)

    def set_morph_kernel(self, size: int) -> None:
        size = max(1, size | 1)  # keep odd & >= 1
        self._morph_kernel = size
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    def _crop(self, gray):
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            return gray[y1:y2, x1:x2]
        return gray

    def update(self, gray: np.ndarray):
        """Feed one grayscale frame. Returns (motion: bool, n_blobs, motion_area)."""
        gray = self._crop(gray)

        fg = self.bg.apply(gray)
        if self.detect_shadows:
            # MOG2 marks shadow pixels as 127 (vs 255 for hard foreground); drop
            # them so moving shadows / illumination changes don't count as motion.
            _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=self.morph_iters)

        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n_blobs = 0
        motion_area = 0.0
        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue
            n_blobs += 1
            motion_area += area
            blobs.append((*cv2.boundingRect(c), area))

        self.last_mask = fg
        self.last_blobs = blobs
        return (n_blobs >= self.min_blobs), n_blobs, motion_area

    def warm(self, gray: np.ndarray) -> None:
        """Update the background model without evaluating motion (warmup)."""
        self.bg.apply(self._crop(gray))

    def reset(self) -> None:
        """Discard and rebuild the background model from scratch.

        Keeps the current var_threshold (calibration may have tuned it) and the
        shadow setting. The caller should re-warm for a few seconds before
        trusting motion again, since the fresh model treats the first frames as
        all-foreground.
        """
        self.bg = self._make_bg()


# ---------------------------------------------------------------------------
# Snippet remux (.h264 elementary stream -> .mp4 via stream copy, no re-encode)
# ---------------------------------------------------------------------------

def _remux(h264_path: Path, mp4_path: Path) -> None:
    """ffmpeg stream-copy .h264 -> .mp4, then delete the .h264. Cheap (no re-encode)."""
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first so the uploader never sees a half-muxed .mp4.
    tmp = mp4_path.with_suffix(".part.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-r", str(FPS), "-i", str(h264_path),
        "-c", "copy", "-f", "mp4", str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(mp4_path)          # atomic within the same filesystem
        log.info("snippet ready: %s (%.1f MB)",
                 mp4_path.name, mp4_path.stat().st_size / (1024 * 1024))
    except subprocess.CalledProcessError as e:
        log.error("remux failed for %s: %s", h264_path.name, e)
        tmp.unlink(missing_ok=True)
    finally:
        h264_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

_running = True


def _handle_signal(signum, frame):  # noqa: ARG001
    global _running
    log.info("signal %s received, finalising current segment then exiting", signum)
    _running = False


def _parse_roi():
    if not ROI:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in ROI.split(","))
        return (x1, y1, x2, y2)
    except ValueError:
        log.warning("ignoring malformed BEEMONITOR_ROI=%r (want x1,y1,x2,y2)", ROI)
        return None


def _snippet_paths(now: datetime):
    """Return (work .h264 path, final .mp4 path) for a clip starting at `now`."""
    stamp = now.strftime("%Y-%m-%d_%H_%M_%S")
    day = now.strftime("%Y-%m-%d")
    h264 = WORK_DIR / f"{stamp}.h264"
    mp4 = RECORD_DIR / day / f"{stamp}.mp4"   # matches uploader.py's regex
    return h264, mp4


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
            # Nest-hole detections (red boxes + IDs), in main-frame coords. IDs
            # follow the nest_detector scheme: top→bottom, left→right, row*10+col.
            nest_boxes = _order_nest_boxes(detect_nest_boxes(bgr))
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


def record() -> None:
    """Main capture loop. Blocks until SIGTERM/SIGINT."""
    if not HAVE_PICAMERA2:
        raise RuntimeError("record() needs picamera2 — run this on the Pi")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    cam = Picamera2()
    config = cam.create_video_configuration(
        main={"size": (MAIN_W, MAIN_H)},
        lores={"size": (LORES_W, LORES_H), "format": "YUV420"},
        controls={"FrameRate": FPS},
        transform=libcamera.Transform(vflip=1, hflip=1),
    )
    cam.configure(config)

    if TIMESTAMP_OVERLAY:
        font = cv2.FONT_HERSHEY_SIMPLEX
        def _apply_timestamp(request):
            ts = time.strftime("%Y-%m-%d %X")
            with MappedArray(request, "main") as m:
                cv2.putText(m.array, ts, (10, 30), font, 0.8, (0, 255, 0), 2)
        cam.pre_callback = _apply_timestamp

    # repeat=True re-emits SPS/PPS so each flushed segment is independently
    # decodable; a ~1s keyframe interval lets segments start cleanly.
    encoder = H264Encoder(repeat=True, iperiod=FPS)
    circ = CircularOutput(buffersize=max(1, int(PRE_ROLL * FPS)))
    encoder.output = circ

    cam.start_encoder(encoder)
    cam.start()

    # Cloud-faithful step 1: detect the hotel and confine detection to it before
    # we start recording. Falls back to the whole frame if detection fails.
    roi = _resolve_record_roi(cam)

    log.info(
        "recorder up: main=%dx%d lores=%dx%d @ %dfps | pre=%.1fs post=%.1fs "
        "max=%.0fs roi=%s",
        MAIN_W, MAIN_H, LORES_W, LORES_H, FPS, PRE_ROLL, POST_ROLL, MAX_SEGMENT,
        roi or "full",
    )

    gate = _build_gate(roi)
    remux_pool = []  # list of threading.Thread for in-flight remuxes

    def _spawn_remux(h264_path: Path, mp4_path: Path):
        t = threading.Thread(target=_remux, args=(h264_path, mp4_path), daemon=True)
        t.start()
        remux_pool.append(t)
        remux_pool[:] = [t for t in remux_pool if t.is_alive()]

    # State machine.
    encoding = False
    seg_start = 0.0
    last_motion = 0.0
    cur_h264: Path | None = None
    cur_mp4: Path | None = None

    # Per-segment / rolling stats for tuning.
    triggers = 0
    total_clip_time = 0.0
    warmup_deadline = time.monotonic() + WARMUP_SECONDS
    frame_i = 0
    last_stats_log = time.monotonic()

    # Periodic background-model rebuild so the gate tracks slow scene changes.
    # The model is already fresh here (gate just built + warmup below).
    next_bg_reset = (
        time.monotonic() + BG_RESET_INTERVAL if BG_RESET_INTERVAL > 0 else float("inf"))

    # Hot-reload of calibration.json written by the scheduled --calibrate job.
    calib_mtime = CALIB_FILE.stat().st_mtime if CALIB_FILE.exists() else 0.0
    tuning_mtime = TUNING_FILE.stat().st_mtime if TUNING_FILE.exists() else 0.0
    last_calib_check = time.monotonic()

    # First telemetry still shortly after warmup, then every interval.
    next_telemetry_image = (
        warmup_deadline if TELEMETRY_IMAGE_INTERVAL > 0 else float("inf"))

    def _open_segment(now_mono: float, reason: str):
        nonlocal encoding, seg_start, cur_h264, cur_mp4, triggers
        cur_h264, cur_mp4 = _snippet_paths(datetime.now())
        circ.fileoutput = str(cur_h264)
        circ.start()
        encoding = True
        seg_start = now_mono
        triggers += 1
        log.info("clip START (%s) -> %s", reason, cur_mp4.name)

    def _close_segment(now_mono: float, reason: str):
        nonlocal encoding, cur_h264, cur_mp4, total_clip_time
        circ.stop()
        encoding = False
        dur = now_mono - seg_start
        total_clip_time += dur
        log.info("clip STOP (%s) len=%.1fs -> remux %s", reason, dur, cur_mp4.name)
        _spawn_remux(cur_h264, cur_mp4)
        cur_h264 = cur_mp4 = None

    try:
        while _running:
            # Blocks until the next lores frame ~= paces the loop at FPS.
            buf = cam.capture_buffer("lores")
            now_mono = time.monotonic()
            frame_i += 1

            # Y-plane of YUV420 == grayscale, first W*H bytes (no padding at
            # 32-aligned widths).
            gray = buf[:LORES_W * LORES_H].reshape(LORES_H, LORES_W)

            # Warmup: build the background model, never trigger.
            if now_mono < warmup_deadline:
                gate.warm(gray)
                continue

            motion = False
            if frame_i % DETECT_EVERY_N == 0:
                motion, n_blobs, area = gate.update(gray)
                if motion:
                    last_motion = now_mono
            else:
                # Still keep the bg model current on skipped frames.
                gate.warm(gray)

            if motion and not encoding:
                _open_segment(now_mono, "motion")

            if encoding:
                # Force-rotate over-long segments (stuck scene / wind in frame).
                if now_mono - seg_start >= MAX_SEGMENT:
                    _close_segment(now_mono, "max-len")
                    if motion:  # still active -> immediately start a fresh clip
                        _open_segment(now_mono, "motion-continued")
                # Normal close: no motion for POST_ROLL.
                elif (now_mono - last_motion) >= POST_ROLL:
                    _close_segment(now_mono, "idle")

            # Periodic background-model rebuild. Deferred while a clip is open so
            # an active capture isn't cut short; fires as soon as the scene idles.
            if now_mono >= next_bg_reset and not encoding:
                gate.reset()
                warmup_deadline = now_mono + WARMUP_SECONDS  # re-learn before trusting motion
                next_bg_reset = now_mono + BG_RESET_INTERVAL
                log.info("background model rebuilt (every %.0fs); re-warming %.1fs",
                         BG_RESET_INTERVAL, WARMUP_SECONDS)

            # Pick up freshly-written calibration OR dashboard tuning without a
            # restart. Tuning is applied on top of calibration so it always wins.
            if now_mono - last_calib_check >= CALIB_RELOAD_SECONDS:
                last_calib_check = now_mono
                try:
                    cm = CALIB_FILE.stat().st_mtime if CALIB_FILE.exists() else 0.0
                except OSError:
                    cm = calib_mtime
                try:
                    tm = TUNING_FILE.stat().st_mtime if TUNING_FILE.exists() else 0.0
                except OSError:
                    tm = tuning_mtime
                if cm != calib_mtime or tm != tuning_mtime:
                    _apply_calibration(gate, load_calibration())
                    _apply_tuning(gate, load_tuning())
                    log.info("reloaded motion params: var=%.0f area=[%.0f, %.0f] "
                             "min_blobs=%d", gate.var_threshold, gate.min_area,
                             gate.max_area, gate.min_blobs)
                    calib_mtime, tuning_mtime = cm, tm

            # On-demand still requested by telemetry (picture / live view): the
            # telemetry service drops capture.request; if its content is "roi" we
            # overlay the motion ROI + nest-hole detections (a debug aid).
            req_file = TELEMETRY_QUEUE / "capture.request"
            if req_file.exists():
                try:
                    want_roi = req_file.read_text().strip() == "roi"
                except OSError:
                    want_roi = False
                _save_telemetry_still(cam, roi=roi, draw_roi=want_roi)
                try:
                    req_file.unlink()
                except OSError:
                    pass

            # Optional periodic still (off by default; TELEMETRY_IMAGE_INTERVAL=0).
            if now_mono >= next_telemetry_image:
                _save_telemetry_still(cam)
                next_telemetry_image = now_mono + TELEMETRY_IMAGE_INTERVAL

            # Periodic stats line for field tuning.
            if now_mono - last_stats_log >= 300:
                mins = (now_mono - last_stats_log) / 60.0
                log.info("stats: %d clips in last %.0f min, %.1fs recorded (%.1f%% duty)",
                         triggers, mins, total_clip_time,
                         100.0 * total_clip_time / (now_mono - last_stats_log))
                triggers = 0
                total_clip_time = 0.0
                last_stats_log = now_mono

    finally:
        if encoding:
            _close_segment(time.monotonic(), "shutdown")
        try:
            cam.stop()
            cam.stop_encoder()
        except Exception:  # pragma: no cover - best-effort teardown
            pass
        for t in remux_pool:
            t.join(timeout=30)
        log.info("recorder stopped")



# ---------------------------------------------------------------------------
# Calibration (on-Pi, one-time): learn the bee-sized blob window from YOLO
# ---------------------------------------------------------------------------
#
# YOLO is too slow for the Pi 4 hot path, but running it on a few hundred frames
# *once* is fine. We measure the MOG2 blob area of every YOLO-confirmed bee and
# freeze the 5th/95th-percentile window into calibration.json. The recorder then
# runs MOG2 only. This is the on-device version of cloud BeeMonitor's
# ``BlobDetector.learn_geometric_thresholds_from_video``.

def _lores_from_bgr(frame_bgr):
    """Mimic the Pi's lores stream from a full-res BGR frame: downscale + gray."""
    small = cv2.resize(frame_bgr, (LORES_W, LORES_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _bbox_overlap(a, b) -> bool:
    """Do (x1,y1,x2,y2) boxes a and b intersect at all?"""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def load_calibration():
    """Return the saved calibration dict, or None if absent/unreadable."""
    if not CALIB_FILE.exists():
        return None
    try:
        return json.loads(CALIB_FILE.read_text())
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable calibration %s: %s", CALIB_FILE, e)
        return None


def _apply_calibration(gate, calib) -> bool:
    """Apply a calibration dict to a live gate. Returns True if applied."""
    if not (calib and "min_area" in calib and "max_area" in calib):
        return False
    gate.min_area = calib["min_area"]
    gate.max_area = calib["max_area"]
    gate.min_blobs = calib.get("min_blobs", gate.min_blobs)
    gate.set_var_threshold(calib.get("var_threshold", gate.var_threshold))
    return True


# Manual motion-tuning overrides from the dashboard (telemetry writes this).
# Applied ON TOP of calibration.json so the nightly auto-calibrate can't clobber
# a value the user set by hand. Only keys present override; others fall through.
TUNING_FILE = CALIB_FILE.parent / "motion_tuning.json"


def load_tuning():
    """Return the dashboard motion-tuning overrides, or None if absent/bad."""
    if not TUNING_FILE.exists():
        return None
    try:
        return json.loads(TUNING_FILE.read_text())
    except (OSError, ValueError) as e:
        log.warning("ignoring unreadable %s: %s", TUNING_FILE, e)
        return None


def _apply_tuning(gate, tuning) -> bool:
    """Override gate params with any dashboard-set values. Returns True if any."""
    if not isinstance(tuning, dict) or not tuning:
        return False
    applied = False
    if tuning.get("min_area") is not None:
        gate.min_area = float(tuning["min_area"]); applied = True
    if tuning.get("max_area") is not None:
        gate.max_area = float(tuning["max_area"]); applied = True
    if tuning.get("min_blobs") is not None:
        gate.min_blobs = int(tuning["min_blobs"]); applied = True
    if tuning.get("var_threshold") is not None:
        gate.set_var_threshold(float(tuning["var_threshold"])); applied = True
    return applied


def _build_gate(roi):
    """Construct the MotionGate, applying calibration.json + manual overrides."""
    gate = MotionGate(roi=roi)
    calib = load_calibration()
    if _apply_calibration(gate, calib):
        log.info("loaded calibration %s: area=[%.0f, %.0f] (from %s bee blobs)",
                 CALIB_FILE, gate.min_area, gate.max_area,
                 calib.get("n_samples", "?"))
    else:
        log.warning("no calibration.json — using permissive defaults "
                    "(area=[%.0f, %.0f]); a scheduled --calibrate will tighten "
                    "these from recorded snippets", gate.min_area, gate.max_area)
    if _apply_tuning(gate, load_tuning()):
        log.info("applied dashboard motion tuning: var=%.0f area=[%.0f, %.0f] "
                 "min_blobs=%d", gate.var_threshold, gate.min_area,
                 gate.max_area, gate.min_blobs)
    return gate


# --- the scheduled, offline calibration pass over saved snippets -----------

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

    roi = _parse_roi()
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
        gate = MotionGate(roi=roi, min_area=4.0, max_area=1e9, min_blobs=1)
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

    min_area = float(np.percentile(bee_areas, 5))
    max_area = float(np.percentile(bee_areas, 95))
    calib = {
        "min_area": round(min_area, 1),
        "max_area": round(max_area, 1),
        "var_threshold": MOG2_VAR_THRESHOLD,
        "min_blobs": MIN_MOTION_BLOBS,
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
    log.info("calibration saved -> %s  area=[%.0f, %.0f] from %d samples",
             CALIB_FILE, min_area, max_area, len(bee_areas))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="BeeMonitor motion-gated recorder")
    parser.add_argument("--calibrate", action="store_true",
                        help="learn the bee blob-area window from recorded "
                             "snippets, then exit (scheduled job; no live bee needed)")
    parser.add_argument("--calibrate-from", nargs="+", metavar="MP4",
                        help="calibrate from these specific snippet(s) instead "
                             "of auto-scanning the recordings")
    parser.add_argument("--model", default=YOLO_MODEL,
                        help=f"YOLO model for calibration (default {YOLO_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="recalibrate even if calibration.json is recent")
    args = parser.parse_args()

    if args.calibrate or args.calibrate_from:
        paths = ([Path(p) for p in args.calibrate_from]
                 if args.calibrate_from else _find_snippets(CALIB_MAX_CLIPS))
        return calibrate(paths, args.model, force=args.force or bool(args.calibrate_from))

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    record()
    return 0


if __name__ == "__main__":
    sys.exit(main())
