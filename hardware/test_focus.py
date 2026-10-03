#!/usr/bin/env python3
"""Autofocus from the dashboard — run it anywhere, including on the Pi.

    python3 hardware/test_focus.py

Pins what needs no camera: where autofocus looks (the drawn ROI, else the
detected hotel, else the centre — never the whole frame), that a dashboard
request keeps the focus in camera.json (or, on "reset", forgets it), and the
result telemetry reports. The lens itself is tested on a unit.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp())
os.environ["BEEMONITOR_CALIB_FILE"] = str(TMP / "calibration.json")
os.environ["BEEMONITOR_RECORD_DIR"] = str(TMP / "rec" / "beeHotel")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion import camera, focus  # noqa: E402
from motion.config import LORES_H, LORES_W  # noqa: E402

FAILS = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


wh = (LORES_W, LORES_H)
r, src = camera.focus_region((LORES_W // 4, 0, LORES_W // 2, LORES_H // 2), wh, drawn=True)
check("a drawn ROI is where it focuses", src == "roi" and r == (0.25, 0.0, 0.5, 0.5))
check("else the hotel it detected",
      camera.focus_region((0, 0, LORES_W // 2, LORES_H), wh, drawn=False)[1] == "hotel")
check("else the centre, not the whole frame",
      camera.focus_region(None, wh, drawn=False) == (camera.CENTER_REGION, "center"))
check("a sliver is not a region", camera.focus_region((10, 10, 11, 11), wh, drawn=True)[1] == "center")


class FakeOak:
    """Has .autofocus, as motion/oak.py's OakCamera does."""
    def __init__(self, result):
        self.result, self.regions = result, []

    def autofocus(self, region):
        self.regions.append(region)
        return dict(self.result)


profile = {"oak_lens": None, "af_range": "normal"}
cam = FakeOak({"ok": True, "lens": 142, "settled": True})
focus.FOCUS_REQUEST_FILE.write_text("focus")
lens = focus.run_request(cam, profile, None, drawn=False, is_oak=True)
state = json.loads(focus.FOCUS_STATE_FILE.read_text())
saved = json.loads(camera.CAMERA_FILE.read_text())
check("focuses on the centre with no ROI", cam.regions == [camera.CENTER_REGION])
check("the lens it landed on is returned", lens == 142)
check("kept across restarts (camera.json)", saved.get("oak_lens") == 142)
check("the request is consumed", not focus.requested())
check("the result is reported", state["ok"] and state["region"] == "center" and state["saved"])

focus.FOCUS_REQUEST_FILE.write_text("reset")
focus.run_request(FakeOak({"ok": True, "lens": 90, "settled": True}), profile,
                  (0, 0, LORES_W, LORES_H), drawn=True, is_oak=True)
saved = json.loads(camera.CAMERA_FILE.read_text())
state = json.loads(focus.FOCUS_STATE_FILE.read_text())
check("reset forgets the kept focus", saved.get("oak_lens") is None and state["reset"])
check("and refocuses on the ROI", state["region"] == "roi" and state["lens"] == 90)

focus.FOCUS_REQUEST_FILE.write_text("focus")
focus.run_request(FakeOak({"ok": False, "lens": 60, "settled": False}), profile, None,
                  drawn=False, is_oak=True)
state = json.loads(focus.FOCUS_STATE_FILE.read_text())
check("a focus that did not settle says so", state["ok"] is False and state["lens"] == 60)

print()
print("FAILED: " + ", ".join(FAILS) if FAILS else "all focus checks passed")
sys.exit(1 if FAILS else 0)
