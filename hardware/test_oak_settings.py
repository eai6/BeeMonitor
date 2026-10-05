"""OAK detail settings (memory/42 part A): HEVC clips remux to a playable .mp4,
and the ISP / exposure settings reach the camera only when set.

    python3 test_oak_settings.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from motion import remux  # noqa: E402
from motion.oak import OakCamera  # noqa: E402


def _cmd_for(suffix):
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / f"clip{suffix}"
        raw.write_bytes(b"x")
        def fake_ffmpeg(cmd, check):
            Path(cmd[-1]).write_bytes(b"mp4")   # what ffmpeg would write
        with mock.patch.object(remux.subprocess, "run", side_effect=fake_ffmpeg) as run:
            remux._remux(raw, Path(tmp) / "out" / "clip.mp4", 20)
        return run.call_args[0][0]


def test_hevc_is_named_and_tagged():
    cmd = _cmd_for(".h265")
    assert cmd[cmd.index("-i") - 2:cmd.index("-i")] == ["-f", "hevc"], cmd
    assert "hvc1" in cmd, cmd
    cmd = _cmd_for(".h264")
    assert "hevc" not in cmd and "hvc1" not in cmd, cmd
    print("ok   remux: HEVC forced + hvc1 tag; H.264 unchanged")


def test_work_file_extension_follows_codec():
    now = datetime(2026, 10, 5, 12, 0, 0)
    assert remux._snippet_paths(now)[0].suffix == ".h264"
    assert remux._snippet_paths(now, "h265")[0].suffix == ".h265"
    print("ok   work file: .h264 / .h265 by codec")


class _Ctrl:
    def __init__(self):
        self.calls = {}

    def __getattr__(self, name):
        return lambda v: self.calls.__setitem__(name, v)


def _isp(settings):
    cam = object.__new__(OakCamera)
    cam.isp = settings
    sent = []
    cam._ctrl_q = mock.Mock(send=sent.append)
    cam._dai = mock.Mock(CameraControl=_Ctrl)
    cam._apply_isp()
    return sent[0].calls if sent else None


def test_isp_only_what_is_set():
    assert _isp({"luma_denoise": -1, "chroma_denoise": -1, "sharpness": -1,
                 "max_exposure_us": 0}) is None
    calls = _isp({"luma_denoise": 0, "chroma_denoise": 9, "sharpness": -1,
                  "max_exposure_us": 1000})
    assert calls == {"setLumaDenoise": 0, "setChromaDenoise": 4,
                     "setAutoExposureLimit": 1000}, calls
    print("ok   ISP: unset values left alone, 0..4 clamped, exposure cap sent")


if __name__ == "__main__":
    test_hevc_is_named_and_tagged()
    test_work_file_extension_follows_codec()
    test_isp_only_what_is_set()
    print("\nall OAK settings checks passed")
