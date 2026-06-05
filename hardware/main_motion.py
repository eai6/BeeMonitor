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
    (audit)        -> an optional heartbeat clip is recorded every hour so you
                      can verify detection quality remotely and recalibrate

Snippets are written as ``RECORD_DIR/YYYY-MM-DD/YYYY-MM-DD_HH_MM_SS.mp4`` — the
exact convention ``uploader.py`` already scans for, so the existing uploader
service ships them unchanged.

IMPORTANT — on-device gating means a missed detection is permanent data loss.
Defaults here favour *over*-triggering. Validate with the logged stats
(triggers/hour, snippet length) and the heartbeat clips before trusting it in
the field, then tighten thresholds.

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
import http.server
import json
import logging
import os
import signal
import socketserver
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

# Heartbeat *video clips*: disabled by default — the telemetry image (below)
# covers the "is the gate working / camera alive" audit far more cheaply.
# Left in the code for bench/WiFi debugging; set >0 to re-enable.
HEARTBEAT_INTERVAL = _env_float("BEEMONITOR_HEARTBEAT_INTERVAL", 0.0)
HEARTBEAT_SECONDS = _env_float("BEEMONITOR_HEARTBEAT_SECONDS", 10.0)

# Telemetry stills queue. Periodic capture is OFF by default now — telemetry is
# JSON-only over cellular; stills are captured ON DEMAND (picture / live view).
# Set BEEMONITOR_TELEMETRY_IMAGE_INTERVAL > 0 to re-enable periodic stills.
TELEMETRY_QUEUE = Path(os.environ.get(
    "BEEMONITOR_TELEMETRY_QUEUE", str(RECORD_DIR.parent / "telemetry")))
TELEMETRY_IMAGE_INTERVAL = _env_float("BEEMONITOR_TELEMETRY_IMAGE_INTERVAL", 0.0)
TELEMETRY_IMAGE_HEIGHT = _env_int("BEEMONITOR_TELEMETRY_IMAGE_HEIGHT", 720)

# 5c: on-demand live MJPEG stream over the LAN (WiFi). Bounded; LAN-only.
STREAM_PORT = _env_int("BEEMONITOR_STREAM_PORT", 8090)
WIFI_STREAM_MAX_SECONDS = _env_float("BEEMONITOR_WIFI_STREAM_MAX_SECONDS", 900.0)

# Detection cost knob: run MOG2 on 1 of every N lores frames (timing stays
# wall-clock based, so this only trades latency for CPU).
DETECT_EVERY_N = max(1, _env_int("BEEMONITOR_DETECT_EVERY_N", 2))

# MOG2 / blob params — ported from BlobDetector defaults, scaled for lores.
MOG2_HISTORY = _env_int("BEEMONITOR_MOG2_HISTORY", 500)
MOG2_VAR_THRESHOLD = _env_int("BEEMONITOR_MOG2_VAR", 16)
MORPH_KERNEL = _env_int("BEEMONITOR_MORPH_KERNEL", 5)
MORPH_ITERS = _env_int("BEEMONITOR_MORPH_ITERS", 2)
# Blob area filters in *lores* pixels. Defaults are deliberately permissive.
MIN_BLOB_AREA = _env_float("BEEMONITOR_MIN_BLOB_AREA", 20.0)
MAX_BLOB_AREA = _env_float("BEEMONITOR_MAX_BLOB_AREA", 5000.0)
# How many qualifying blobs constitute "motion".
MIN_MOTION_BLOBS = _env_int("BEEMONITOR_MIN_MOTION_BLOBS", 1)

# Optional detection ROI in lores coords: "x1,y1,x2,y2" (e.g. the hotel face).
# Empty = whole frame.
ROI = os.environ.get("BEEMONITOR_ROI", "").strip()

# Calibration: where the YOLO-derived blob-area window is stored, and the YOLO
# model used by the (scheduled, offline) calibration pass over saved snippets.
CALIB_FILE = Path(os.environ.get(
    "BEEMONITOR_CALIB_FILE", str(RECORD_DIR.parent / "calibration.json")))
YOLO_MODEL = os.environ.get("BEEMONITOR_YOLO_MODEL", "yolo11n.pt")
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
                 max_area=MAX_BLOB_AREA, min_blobs=MIN_MOTION_BLOBS):
        self.var_threshold = var_threshold
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=False)
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


def _save_telemetry_still(cam) -> None:
    """Capture one downscaled JPEG into the telemetry queue (best-effort).

    The telemetry service ships the latest queued image over cellular. We grab
    the main (recorded) stream so the still reflects the real framing, then
    downscale to keep it small. Never let a capture error stop recording.
    """
    try:
        frame = cam.capture_array("main")
        # Normalise to BGR regardless of the main stream's pixel format.
        if frame.ndim == 3 and frame.shape[2] >= 3:
            bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)
        else:  # YUV420 (I420) packed: shape (H*3/2, W)
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

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


# ---------------------------------------------------------------------------
# 5c: on-demand live MJPEG stream over the LAN (WiFi). EXPERIMENTAL — verify on
# hardware. Bounded by duration; reachable only on the device's local network
# (or via Raspberry Pi Connect), never proxied by the cloud dashboard.
# ---------------------------------------------------------------------------

class _StreamState:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None            # latest JPEG bytes
        self.active_until = 0.0      # time.monotonic() deadline
        self.server = None

    def active(self) -> bool:
        return time.monotonic() < self.active_until


_stream = _StreamState()


class _MJPEGHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request logging
        pass

    def do_GET(self):
        if self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while _stream.active():
                    with _stream.lock:
                        frame = _stream.frame
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(1 / 12.0)  # cap ~12 fps to the client
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            body = (b"<html><body style='margin:0;background:#000'>"
                    b"<img src='/stream.mjpg' style='width:100%;height:auto'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)


def _lan_ip():
    """First non-cellular IPv4 (wlan*/eth*/en*) — the address to reach the stream."""
    try:
        out = subprocess.run(["ip", "-o", "-4", "addr", "show"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1].startswith(("wl", "eth", "en")):
            return parts[3].split("/")[0]
    return None


def _start_stream(duration: float):
    """Start (or extend) the MJPEG server; return the LAN URL or None."""
    _stream.active_until = time.monotonic() + min(duration, WIFI_STREAM_MAX_SECONDS)
    if _stream.server is None:
        try:
            srv = socketserver.ThreadingTCPServer(("0.0.0.0", STREAM_PORT), _MJPEGHandler)
            srv.daemon_threads = True
            _stream.server = srv
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            log.info("mjpeg stream server listening on :%d", STREAM_PORT)
        except OSError as e:
            log.warning("could not start stream server: %s", e)
            return None
    ip = _lan_ip()
    return f"http://{ip}:{STREAM_PORT}/" if ip else None


def record() -> None:
    """Main capture loop. Blocks until SIGTERM/SIGINT."""
    if not HAVE_PICAMERA2:
        raise RuntimeError("record() needs picamera2 — run this on the Pi")
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    roi = _parse_roi()

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
    heartbeat_until = 0.0
    next_heartbeat = (time.monotonic() + HEARTBEAT_INTERVAL) if HEARTBEAT_INTERVAL > 0 else float("inf")

    # Per-segment / rolling stats for tuning.
    triggers = 0
    total_clip_time = 0.0
    warmup_deadline = time.monotonic() + WARMUP_SECONDS
    frame_i = 0
    last_stats_log = time.monotonic()

    # Hot-reload of calibration.json written by the scheduled --calibrate job.
    calib_mtime = CALIB_FILE.stat().st_mtime if CALIB_FILE.exists() else 0.0
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

            # Heartbeat: schedule a forced clip window when idle.
            if not encoding and now_mono >= next_heartbeat:
                heartbeat_until = now_mono + HEARTBEAT_SECONDS
                next_heartbeat = now_mono + HEARTBEAT_INTERVAL
                _open_segment(now_mono, "heartbeat")
            in_heartbeat = now_mono < heartbeat_until

            if motion and not encoding:
                _open_segment(now_mono, "motion")

            if encoding:
                # Force-rotate over-long segments (stuck scene / wind in frame).
                if now_mono - seg_start >= MAX_SEGMENT:
                    _close_segment(now_mono, "max-len")
                    if motion:  # still active -> immediately start a fresh clip
                        _open_segment(now_mono, "motion-continued")
                # Normal close: no motion for POST_ROLL and heartbeat window over.
                elif not in_heartbeat and (now_mono - last_motion) >= POST_ROLL:
                    _close_segment(now_mono, "idle")

            # Pick up a freshly-written calibration without restarting.
            if now_mono - last_calib_check >= CALIB_RELOAD_SECONDS:
                last_calib_check = now_mono
                try:
                    m = CALIB_FILE.stat().st_mtime if CALIB_FILE.exists() else 0.0
                except OSError:
                    m = calib_mtime
                if m != calib_mtime:
                    if _apply_calibration(gate, load_calibration()):
                        log.info("reloaded calibration: area=[%.0f, %.0f]",
                                 gate.min_area, gate.max_area)
                    calib_mtime = m

            # On-demand still requested by telemetry (picture / live view): the
            # telemetry service drops capture.request, we grab a frame and remove it.
            if (TELEMETRY_QUEUE / "capture.request").exists():
                _save_telemetry_still(cam)
                try:
                    (TELEMETRY_QUEUE / "capture.request").unlink()
                except OSError:
                    pass

            # 5c: on-demand LAN MJPEG stream. telemetry drops wifistream.request;
            # we start the server, advertise the URL via stream.status, and feed
            # frames below while active.
            sreq = TELEMETRY_QUEUE / "wifistream.request"
            if sreq.exists():
                try:
                    dur = float(sreq.read_text().strip() or WIFI_STREAM_MAX_SECONDS)
                except (OSError, ValueError):
                    dur = WIFI_STREAM_MAX_SECONDS
                url = _start_stream(dur)
                try:
                    (TELEMETRY_QUEUE / "stream.status").write_text(
                        json.dumps({"url": url, "until": time.time() + min(dur, WIFI_STREAM_MAX_SECONDS)}))
                    sreq.unlink()
                except OSError:
                    pass
            # While streaming, publish the current lores frame as JPEG (cheap).
            if _stream.active():
                ok, jpg = cv2.imencode(".jpg", gray, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    with _stream.lock:
                        _stream.frame = jpg.tobytes()

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


def _build_gate(roi):
    """Construct the MotionGate, applying calibration.json if present."""
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
