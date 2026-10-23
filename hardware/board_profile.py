"""Which board this unit runs on, and whether it runs light (memory/56).

One golden image serves every board. A board with under 1 GB of memory (a
Pi Zero 2 W, a Pi 3A+) can't hold PyTorch beside the recorder, so it runs in
light mode: 720p at 15 fps, no hotel detection or calibration with YOLO, no
activity crops, and no desktop session. Nothing on the card needs editing.

``BEEMONITOR_PROFILE=full|lite`` forces a profile. Anything set explicitly in
uploader.env still wins over the light defaults (they're only defaults), and
dashboard overrides apply on top as before.

Stdlib only: enroll.py imports this under the system python3 as root.
Named board_profile, not profile, which is a stdlib module.
"""

import os
import re

MODEL_FILE = "/proc/device-tree/model"
MEMINFO_FILE = "/proc/meminfo"
# A 1 GB board reports a little under 1024 MB (the kernel and GPU keep some),
# so 900 keeps a Pi 3B full and puts any 512 MB board in light mode.
LITE_BELOW_MB = 900

FULL, LITE = "full", "lite"

# Defaults a light board records with; explicit uploader.env values win.
LITE_DEFAULTS = {
    "BEEMONITOR_MAIN_W": "1280",
    "BEEMONITOR_MAIN_H": "720",
    "BEEMONITOR_FPS": "15",
    "BEEMONITOR_HOTEL_ROI_DETECT": "false",
    "BEEMONITOR_ACTIVITY_FRAMES": "false",
}


def board_model(path=MODEL_FILE):
    """'Raspberry Pi Zero 2 W Rev 1.0', or '' off a Pi."""
    try:
        with open(path, "rb") as fh:
            return fh.read().decode("utf-8", "replace").strip("\x00\n ")
    except OSError:
        return ""


def mem_mb(path=MEMINFO_FILE):
    """Total memory in MB, or None when it can't be read."""
    try:
        with open(path) as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def detect(environ=None, model_path=MODEL_FILE, meminfo_path=MEMINFO_FILE):
    """``"lite"`` or ``"full"`` for this board."""
    environ = os.environ if environ is None else environ
    forced = (environ.get("BEEMONITOR_PROFILE") or "").strip().lower()
    if forced in (FULL, LITE):
        return forced
    if "zero 2" in board_model(model_path).lower():
        return LITE
    mb = mem_mb(meminfo_path)
    if mb is not None and mb < LITE_BELOW_MB:
        return LITE
    return FULL


def apply_defaults(environ=None, **kw):
    """Fill in the light defaults this board needs, without touching anything
    already set. Returns the profile."""
    environ = os.environ if environ is None else environ
    profile = detect(environ, **kw)
    if profile == LITE:
        for key, value in LITE_DEFAULTS.items():
            environ.setdefault(key, value)
    return profile


def describe(**kw):
    """What telemetry reports: the board (without its revision, "Raspberry
    Pi Zero 2 W"), its memory and the profile."""
    board = re.sub(r"\s+Rev\s+[\w.]+$", "", board_model(kw.get("model_path", MODEL_FILE)))
    return {"board": board,
            "mem_mb": mem_mb(kw.get("meminfo_path", MEMINFO_FILE)),
            "profile": detect(**kw)}
