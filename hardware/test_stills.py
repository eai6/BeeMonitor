#!/usr/bin/env python3
"""Full-resolution stills — run it anywhere, including on the Pi.

    python3 hardware/test_stills.py

Pins the parts that need no real camera: when a still is due (never mid-clip;
a "take one now" ignores the schedule), the 15-minute floor, the files a still
leaves for the uploader (marker last), a motion burst on a stand-in camera
(one switch each way, N frames, video restored), that the uploader sends a
still through the same calls as a video and — like a video — keeps it on the
card, and that telemetry's cleanup deletes only what the dashboard cleared.
The capture itself — the mode switch on a real OV64A40 — is tested on a unit.

Needs cv2 + numpy + requests. Exits non-zero on failure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp())
os.environ["BEEMONITOR_STILLS_DIR"] = str(TMP / "stills")
os.environ["BEEMONITOR_RECORD_DIR"] = str(TMP / "rec" / "beeHotel")
os.environ.setdefault("BEEMONITOR_API_BASE", "https://example.invalid")
os.environ.setdefault("BEEMONITOR_DEVICE_KEY", "bmk_device_test")
sys.path.insert(0, str(HERE))

from motion import stills  # noqa: E402

FAILS = []


def check(name, cond):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        FAILS.append(name)


# -- when ---------------------------------------------------------------------
d = dict(encoding=False, in_window=True, recording_on=True, requested=False)
check("due when the interval has passed", stills.due(100, 50, 900, **d))
check("not before it", not stills.due(40, 50, 900, **d))
check("never mid-clip", not stills.due(100, 50, 900, **dict(d, encoding=True)))
check("never mid-clip, even on request",
      not stills.due(100, 50, 900, **dict(d, encoding=True, requested=True)))
check("not outside the hour window", not stills.due(100, 50, 900, **dict(d, in_window=False)))
check("not with recording off", not stills.due(100, 50, 900, **dict(d, recording_on=False)))
check("off means off", not stills.due(100, 50, 0, **d))
check("a request ignores schedule and window",
      stills.due(0, 50, 0, **dict(d, in_window=False, recording_on=False, requested=True)))
check("0 = off", stills.interval_seconds(0) == 0)
check("15 min floor", stills.interval_seconds(5) == 900)
check("30 min", stills.interval_seconds(30) == 1800)
check("junk = off", stills.interval_seconds("x") == 0)

# -- the files it leaves ------------------------------------------------------
arr = np.random.default_rng(0).integers(0, 255, (3472, 4624, 3), dtype=np.uint8)
t = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
stills._write(arr, t, "16mp", 6.2, "schedule")
stem = TMP / "stills" / "2026-09-25_10_00_00"
meta = json.loads(Path(str(stem) + ".json").read_text())
check("full image written", Path(str(stem) + ".jpg").stat().st_size > 1_000_000)
import cv2  # noqa: E402
th = cv2.imread(str(stem) + ".thumb.jpg")
check("1280 px preview", th is not None and max(th.shape[:2]) == 1280)
check("marker carries size, mode, lens",
      (meta["width"], meta["height"], meta["sensor_mode"], meta["lens_position"])
      == (4624, 3472, "16mp", 6.2))
check("no temp marker left", not list((TMP / "stills").glob("*.tmp")))

# -- a motion burst on a stand-in camera ------------------------------------------
class FakeCam:
    sensor_resolution = (320, 240)

    def __init__(self):
        self.calls = []

    def stop_encoder(self):
        self.calls.append("stop_encoder")

    def start_encoder(self, enc):
        self.calls.append("start_encoder")

    def create_still_configuration(self, main, buffer_count, transform):
        return {"main": main, "controls": {}}

    def switch_mode(self, cfg):
        self.calls.append("switch:" + ("still" if isinstance(cfg, dict) else cfg))

    def capture_array(self, name):
        self.calls.append("capture")
        return np.zeros((240, 320, 3), np.uint8)


for p in (TMP / "stills").iterdir():
    p.unlink()
cam = FakeCam()
n = stills.take_burst(cam, object(), "video", None, None, 5)
import time as _t  # noqa: E402
for _ in range(50):   # the writer thread
    if len(list((TMP / "stills").glob("*.json"))) == 5:
        break
    _t.sleep(0.1)
check("burst: 5 frames", n == 5)
check("burst: one switch each way, encoder restarted",
      cam.calls == ["stop_encoder", "switch:still"] + ["capture"] * 5
      + ["switch:video", "start_encoder"])
metas = [json.loads(p.read_text()) for p in (TMP / "stills").glob("*.json")]
check("burst: 5 markers, one burst id, indexes 0-4",
      len(metas) == 5 and len({m["burst_id"] for m in metas}) == 1
      and sorted(m["burst_index"] for m in metas) == [0, 1, 2, 3, 4]
      and all(m["source"] == "burst" for m in metas))
for p in (TMP / "stills").iterdir():
    p.unlink()

# -- the upload: the video calls, kind "still" ---------------------------------
import uploader  # noqa: E402

still_dir = TMP / "stills"
for p in still_dir.iterdir():
    p.unlink()
stills._write(arr[:600, :800].copy(), t, "64mp", None, "manual")
calls = []


def fake_post(path, json):  # noqa: A002 - mirrors uploader._api_post
    calls.append((path, json))
    if path.endswith("initiate"):
        return {"storage_key": f"users/1/devices/2/stills/{json['kind']}.jpg",
                "upload_url": "https://put"}
    return {"still_id": 7}


with mock.patch.object(uploader, "_api_post", side_effect=fake_post), \
     mock.patch.object(uploader, "_put_to_s3") as put:
    pending = uploader._list_pending_stills(still_dir)
    check("uploader finds the still", len(pending) == 1)
    uploader._upload_still(pending[0])
kinds = [c[1].get("kind") for c in calls]
check("preview, full, complete — through the video upload calls",
      [c[0] for c in calls] == ["/api/v1/uploads/initiate", "/api/v1/uploads/initiate",
                                "/api/v1/uploads/complete"]
      and kinds == ["still_thumb", "still", "still"])
done = calls[-1][1]
check("complete names both keys and the metadata",
      (done["thumb_key"].endswith("still_thumb.jpg"), done["width"], done["source"])
      == (True, 800, "manual"))
check("two PUTs", put.call_count == 2)
side = Path(str(pending[0]) + ".uploaded")
check("kept on the card, like a video, with the cloud id",
      Path(str(pending[0])[:-5] + ".jpg").exists() and "still_id=7" in side.read_text())
check("not uploaded twice", uploader._list_pending_stills(still_dir) == [])

# -- telemetry's cleanup: only what the dashboard cleared ---------------------------
import telemetry  # noqa: E402

posted = []
with mock.patch.object(telemetry, "requests") as rq:
    rq.post.side_effect = lambda url, headers, json, timeout: posted.append(json)
    rq.RequestException = Exception
    count, freed = telemetry._cleanup_stills([7, 99], "https://x/", {})
check("cleared still deleted from the card", not list(still_dir.iterdir()))
check("both confirmed (99 was not here)", count == 2 and posted == [{"deleted_stills": [7, 99]}])
check("nothing cleared, nothing deleted", telemetry._cleanup_stills([], "https://x/", {}) == (0, 0))

print()
print("FAILED: " + ", ".join(FAILS) if FAILS else "all stills checks passed")
sys.exit(1 if FAILS else 0)
