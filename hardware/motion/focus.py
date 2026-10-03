"""Autofocus from the dashboard (the device page's Autofocus button).

Telemetry turns the "autofocus" command into ``focus.request`` (content
"reset" to also forget a pinned focus). The recorder, between clips, focuses on
the drawn ROI — else the hotel it detected, else the centre of the frame —
holds the lens there, saves the position to camera.json so a restart keeps it
(instead of autofocusing again in whatever light it boots into), and writes
``focus_state.json``, which telemetry sends with each beat for the page.

"Reset focus" forgets the saved position (the next start autofocuses again)
and autofocuses now without saving.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from motion.config import CALIB_FILE, LORES_H, LORES_W, log

FOCUS_REQUEST_FILE = CALIB_FILE.parent / "focus.request"
FOCUS_STATE_FILE = CALIB_FILE.parent / "focus_state.json"


def requested() -> bool:
    return FOCUS_REQUEST_FILE.exists()


def run_request(cam, cam_profile: dict, gate_roi, drawn: bool, is_oak: bool):
    """Act on a pending request. Returns the lens position now held (or None)."""
    from motion.camera import autofocus, focus_region, save_profile

    try:
        reset = FOCUS_REQUEST_FILE.read_text().strip() == "reset"
    except OSError:
        reset = False
    try:
        FOCUS_REQUEST_FILE.unlink()
    except OSError:
        pass

    region, source = focus_region(gate_roi, (LORES_W, LORES_H), drawn)
    key = "oak_lens" if is_oak else "lens"
    log.info("autofocus requested from the dashboard%s — on the %s %s",
             " (reset)" if reset else "", source, tuple(round(v, 2) for v in region))
    try:
        result = autofocus(cam, region, cam_profile.get("af_range", "normal"))
    except Exception as e:  # never stop recording over a focus
        log.warning("autofocus failed: %s", e)
        result = {"ok": False, "lens": None, "settled": False, "error": str(e)[:200]}

    lens = result.get("lens")
    try:
        if reset:
            save_profile(**{key: None})       # the next start autofocuses again
            cam_profile[key] = None
        elif lens is not None:
            save_profile(**{key: lens})       # a restart keeps this focus
            cam_profile[key] = lens
    except Exception as e:
        log.warning("autofocus: could not save the lens position: %s", e)

    write_state({**result, "region": source, "reset": reset,
                 "saved": (not reset) and lens is not None, "camera": "oak" if is_oak else "picamera2"})
    return lens


def write_state(state: dict) -> None:
    state = dict(state, at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    try:
        FOCUS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = FOCUS_STATE_FILE.with_name(FOCUS_STATE_FILE.name + ".part")
        tmp.write_text(json.dumps(state))
        os.replace(tmp, FOCUS_STATE_FILE)
    except OSError as e:
        log.warning("autofocus: could not write %s: %s", FOCUS_STATE_FILE, e)
