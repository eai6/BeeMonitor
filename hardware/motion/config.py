"""Configuration for the motion-gated recorder.

Every tunable is env-overridable and follows uploader.py's ``BEEMONITOR_*``
convention. This module is the leaf of the ``motion`` package — everything else
imports its constants from here, so all knobs live in exactly one place.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path


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

# --- Activity-frame sampling (taxonomic monitoring) ------------------------
# Per recorded activity (motion clip), sample a few crops of the *mover* and
# queue them for the telemetry service to ship over cellular; BioCLIP identifies
# the insect in the cloud. Crops only (tiny bytes). The WiFi-gated full video is
# unaffected. See memory/15_monitoring_agent_design.md.
ACTIVITY_FRAMES = _env_bool("BEEMONITOR_ACTIVITY_FRAMES", True)
# Which activities to sample/send crops for (BioCLIP review):
#   all       — every activity, confirmed or not (max data for cloud tagging)
#   confirmed — only activities the on-device bee-confirmer accepted (default;
#               with confirmation off, nothing is rejected so it sends all)
#   off       — sample/send nothing (no SD/CPU/cellular spend)
# A dashboard-pushed value wins over this env default (see overrides.py). The env
# default tracks ACTIVITY_FRAMES so the lite profile's BEEMONITOR_ACTIVITY_FRAMES=
# false still means "off" out of the box.
ACTIVITY_CROPS_MODE = os.environ.get(
    "BEEMONITOR_ACTIVITY_CROPS_MODE",
    "confirmed" if ACTIVITY_FRAMES else "off").strip().lower()
if ACTIVITY_CROPS_MODE not in ("all", "confirmed", "off"):
    ACTIVITY_CROPS_MODE = "confirmed"
ACTIVITY_FRAMES_QUEUE = Path(os.environ.get(
    "BEEMONITOR_ACTIVITY_FRAMES_QUEUE", str(RECORD_DIR.parent / "activity_frames")))
# How many crops to keep + queue per activity (the strongest-motion ones).
FRAMES_PER_ACTIVITY = max(1, _env_int("BEEMONITOR_FRAMES_PER_ACTIVITY", 1))
# Capture at most one candidate this often (s) during a clip, capped in number,
# so a long clip doesn't grab a main still on every frame.
FRAME_CAPTURE_INTERVAL = _env_float("BEEMONITOR_FRAME_CAPTURE_INTERVAL", 1.0)
FRAME_MAX_CANDIDATES = max(FRAMES_PER_ACTIVITY, _env_int("BEEMONITOR_FRAME_MAX_CANDIDATES", 4))
# Fractional padding added around the mover bbox for context before cropping.
FRAME_CROP_PAD = _env_float("BEEMONITOR_FRAME_CROP_PAD", 0.4)
# Downscale a crop so its longest side is at most this many px (keeps bytes tiny).
FRAME_MAX_SIDE = _env_int("BEEMONITOR_FRAME_MAX_SIDE", 384)

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
#
# NOTE: this file lives at hardware/motion/config.py, so the repo root is
# parents[2] (motion -> hardware -> repo root). The original main_motion.py sat
# one level up, hence the extra parent here.
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
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
# is picked up without a restart. Calibration is a slow background job, so this
# can be lazy.
CALIB_RELOAD_SECONDS = _env_float("BEEMONITOR_CALIB_RELOAD_SECONDS", 300.0)
# Interactive dashboard edits — the hotel-ROI override and motion tuning — are
# re-read MUCH faster than calibration, since a human makes them and expects to
# see the result quickly (in the live ROI photo). The check only does work when a
# file's mtime actually changes, so a short interval is cheap.
OVERRIDE_RELOAD_SECONDS = _env_float("BEEMONITOR_OVERRIDE_RELOAD_SECONDS", 15.0)

# Burn a timestamp into the recorded (main) frames like main.py does.
TIMESTAMP_OVERLAY = _env_bool("BEEMONITOR_TIMESTAMP", True)

# Manual motion-tuning overrides from the dashboard (telemetry writes this).
# Applied ON TOP of calibration.json so the nightly auto-calibrate can't clobber
# a value the user set by hand. Only keys present override; others fall through.
TUNING_FILE = CALIB_FILE.parent / "motion_tuning.json"
# Dashboard ROI editor: hotel ROI override (normalized [x1,y1,x2,y2], 0..1) and
# nest layout ([{id, box:[x1,y1,x2,y2] normalized}, ...]). telemetry writes them.
ROI_OVERRIDE_FILE = CALIB_FILE.parent / "roi_override.json"
NEST_LAYOUT_FILE = CALIB_FILE.parent / "nest_layout.json"
# Dashboard-pushed bee-confirmation mode (off|tag|gate). telemetry.py writes it
# from the heartbeat; the recorder hot-reloads it over the env default, so a
# no-shell unit can be switched between observe (tag) and filter (gate) remotely.
BEE_CONFIRM_MODE_FILE = CALIB_FILE.parent / "bee_confirm_mode.json"
# Dashboard-pushed toggle for sampling/sending BioCLIP review crops over cellular.
# telemetry.py writes it from the heartbeat; the recorder hot-reloads it over the
# env ACTIVITY_FRAMES default, so a no-shell unit can stop the 1-few crops/activity
# remotely (e.g. once on-device bee confirmation is trusted to guard activity).
ACTIVITY_FRAMES_FILE = CALIB_FILE.parent / "activity_frames.json"


# --- Bee confirmation (low-DL YOLO filter) --------------------------------
# Confirm motion is actually a bee with bee_tracking.pt on a few full frames per
# activity (async, off the capture hot path). See memory/17_bee_confirmation_design.
# Mode:
#   off  — no confirmation (record + count + send crops as before)
#   tag  — run YOLO, tag the clip + crops with the verdict, but still count + send
#   gate — (default) unconfirmed activities are not counted (telemetry) and their
#          crops are not sent (cellular/BioCLIP); the clip is still recorded +
#          uploaded, tagged, so nothing is lost.
BEE_CONFIRM_MODE = os.environ.get("BEEMONITOR_BEE_CONFIRM_MODE", "gate").strip().lower()
# Whole-frame bee detector — same weights as cloud/calibration (reuse YOLO_MODEL).
BEE_CONFIRM_MODEL = os.environ.get("BEEMONITOR_BEE_CONFIRM_MODEL", YOLO_MODEL)
# A notch above the 0.25 detect default, to bias against confirming noise as a bee.
BEE_CONFIRM_CONF = _env_float("BEEMONITOR_BEE_CONFIRM_CONF", 0.30)
# Native whole-frame inference size; do NOT shrink — small bees need the resolution.
BEE_CONFIRM_IMGSZ = _env_int("BEEMONITOR_BEE_CONFIRM_IMGSZ", 640)
# Mover-overlapping bee detections needed to confirm an activity.
BEE_CONFIRM_MIN_CONFIRMATIONS = max(1, _env_int("BEEMONITOR_BEE_CONFIRM_MIN_CONFIRMATIONS", 1))
# Tolerance (lores px) when matching a YOLO bee box to the MOG2 mover blob: the
# mover box is inflated by this much before the overlap test, so a bee whose
# detection box is *near* the motion (body vs wing-motion offset, slight lag)
# still confirms. 0 = strict overlap. A large gap (bee resting while something
# else moves elsewhere in frame) stays unconfirmed.
BEE_CONFIRM_OVERLAP_PAD = max(0, _env_int("BEEMONITOR_BEE_CONFIRM_OVERLAP_PAD", 40))
# Negative budget: after this many no-bee frames an activity is marked unconfirmed.
BEE_CONFIRM_MAX_RUNS = max(1, _env_int("BEEMONITOR_BEE_CONFIRM_MAX_RUNS", 3))
# Keep inference from starving the capture loop on the 4-core Pi.
BEE_CONFIRM_TORCH_THREADS = max(1, _env_int("BEEMONITOR_BEE_CONFIRM_TORCH_THREADS", 1))
# Bounded work queue; drop oldest on overflow (never back-pressure the recorder).
BEE_CONFIRM_QUEUE_MAX = max(1, _env_int("BEEMONITOR_BEE_CONFIRM_QUEUE_MAX", 8))
# A closed activity with no verdict by this long fails closed (uncounted, crops
# un-sent) — safe because the clip is already recorded + tagged.
BEE_CONFIRM_VERDICT_TIMEOUT = _env_float("BEEMONITOR_BEE_CONFIRM_VERDICT_TIMEOUT", 20.0)


logging.basicConfig(
    format="%(asctime)s %(levelname)s recorder %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("recorder")
