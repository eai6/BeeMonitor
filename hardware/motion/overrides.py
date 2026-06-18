"""Load + apply the JSON files that tune the gate at runtime.

Three sources, applied in this precedence (later wins):
    1. calibration.json   — the scheduled --calibrate job's learned blob window
    2. motion_tuning.json — manual dashboard overrides (only keys present win)
The ROI editor's roi_override.json / nest_layout.json are read here too. All of
these are filesystem contracts shared with the telemetry service and dashboard.
"""

from __future__ import annotations

import json

from motion.config import (
    log, CALIB_FILE, TUNING_FILE, ROI_OVERRIDE_FILE, NEST_LAYOUT_FILE,
    BEE_CONFIRM_MODE_FILE, BEE_CONFIRM_MODE, ACTIVITY_FRAMES_FILE, ACTIVITY_FRAMES,
    ACTIVITY_CROPS_MODE, LORES_W, LORES_H,
)
from motion.gate import MotionGate


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


def _load_json_file(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def load_roi_override_lores():
    """Dashboard hotel-ROI override (normalized) -> lores box, or None."""
    d = _load_json_file(ROI_OVERRIDE_FILE)
    if not (isinstance(d, list) and len(d) == 4):
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in d)
    except (TypeError, ValueError):
        return None
    box = (int(max(0.0, min(x1, x2)) * LORES_W), int(max(0.0, min(y1, y2)) * LORES_H),
           int(min(1.0, max(x1, x2)) * LORES_W), int(min(1.0, max(y1, y2)) * LORES_H))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    return box


def load_nest_layout():
    """Dashboard nest layout: list of (id:int, [x1,y1,x2,y2] normalized)."""
    d = _load_json_file(NEST_LAYOUT_FILE)
    if not isinstance(d, list):
        return []
    out = []
    for it in d:
        try:
            b = it["box"]
            out.append((int(it.get("id", 0)),
                        [float(b[0]), float(b[1]), float(b[2]), float(b[3])]))
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return out


def load_bee_confirm_mode() -> str:
    """Effective bee-confirmation mode: a dashboard-pushed value (bee_confirm_mode
    .json) wins over the env default, so a no-shell unit can be switched between
    off/tag/gate remotely. Falls back to the env BEE_CONFIRM_MODE."""
    d = _load_json_file(BEE_CONFIRM_MODE_FILE)
    if isinstance(d, dict):
        m = str(d.get("mode", "")).strip().lower()
        if m in ("off", "tag", "gate"):
            return m
    return BEE_CONFIRM_MODE


def load_activity_crops_mode() -> str:
    """Which activities to sample/send crops for: 'all' | 'confirmed' | 'off'.
    A dashboard-pushed value (activity_frames.json {"mode": ...}) wins over the env
    ACTIVITY_CROPS_MODE default, so a no-shell unit can switch remotely. Falls back
    to the legacy {"enabled": bool} key (old clouds) then the env default."""
    d = _load_json_file(ACTIVITY_FRAMES_FILE)
    if isinstance(d, dict):
        m = str(d.get("mode", "")).strip().lower()
        if m in ("all", "confirmed", "off"):
            return m
        if "enabled" in d:  # legacy bool toggle
            return "confirmed" if bool(d["enabled"]) else "off"
    return ACTIVITY_CROPS_MODE


def load_activity_frames_enabled() -> bool:
    """Back-compat shim: crops sampled at all (mode != off)."""
    return load_activity_crops_mode() != "off"


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
