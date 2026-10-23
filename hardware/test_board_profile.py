#!/usr/bin/env python3
"""A unit picks light mode on a small board, by itself (memory/56).

    python3 hardware/test_board_profile.py

Stdlib only: the board's model and memory files are faked.
Exits non-zero on failure.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_profile  # noqa: E402
import enroll  # noqa: E402

ZERO_2W = "Raspberry Pi Zero 2 W Rev 1.0\x00"
PI_4 = "Raspberry Pi 4 Model B Rev 1.5\x00"


class _Board:
    """Fake /proc/device-tree/model and /proc/meminfo files. Hold on to the
    instance while its ``kw`` is used: the folder goes when it does."""

    def __init__(self, model, mem_kb):
        self.dir = tempfile.TemporaryDirectory()
        self.model = os.path.join(self.dir.name, "model")
        self.meminfo = os.path.join(self.dir.name, "meminfo")
        if model is not None:
            Path(self.model).write_bytes(model.encode())
        Path(self.meminfo).write_text(f"MemTotal:       {mem_kb} kB\nMemFree:  1 kB\n")

    @property
    def kw(self):
        return {"model_path": self.model, "meminfo_path": self.meminfo}


class DetectTests(unittest.TestCase):
    def test_a_zero_2w_is_light(self):
        board = _Board(ZERO_2W, 427_000)
        self.assertEqual(board_profile.detect({}, **board.kw), "lite")

    def test_a_pi_4_is_full(self):
        board = _Board(PI_4, 3_884_000)
        self.assertEqual(board_profile.detect({}, **board.kw), "full")

    def test_any_512_mb_board_is_light(self):
        board = _Board("Raspberry Pi 3 Model A Plus Rev 1.0", 443_000)
        self.assertEqual(board_profile.detect({}, **board.kw), "lite")

    def test_a_1_gb_board_stays_full(self):
        board = _Board("Raspberry Pi 3 Model B Rev 1.2", 945_000)   # what a 1 GB Pi 3B reports
        self.assertEqual(board_profile.detect({}, **board.kw), "full")

    def test_off_a_pi_it_is_full(self):
        board = _Board(None, 16_000_000)
        self.assertEqual(board_profile.detect({}, **board.kw), "full")

    def test_the_profile_can_be_forced_either_way(self):
        zero, pi4 = _Board(ZERO_2W, 427_000), _Board(PI_4, 3_884_000)
        self.assertEqual(board_profile.detect({"BEEMONITOR_PROFILE": "full"}, **zero.kw), "full")
        self.assertEqual(board_profile.detect({"BEEMONITOR_PROFILE": "lite"}, **pi4.kw), "lite")


class DefaultsTests(unittest.TestCase):
    def test_light_defaults_fill_only_what_is_unset(self):
        env = {"BEEMONITOR_FPS": "10"}
        board = _Board(ZERO_2W, 427_000)
        self.assertEqual(board_profile.apply_defaults(env, **board.kw), "lite")
        self.assertEqual(env["BEEMONITOR_FPS"], "10")                   # uploader.env wins
        self.assertEqual(env["BEEMONITOR_MAIN_W"], "1280")
        self.assertEqual(env["BEEMONITOR_HOTEL_ROI_DETECT"], "false")
        self.assertEqual(env["BEEMONITOR_ACTIVITY_FRAMES"], "false")

    def test_a_full_board_gets_nothing(self):
        env = {}
        board = _Board(PI_4, 3_884_000)
        board_profile.apply_defaults(env, **board.kw)
        self.assertEqual(env, {})

    def test_telemetry_gets_the_board_without_its_revision(self):
        board = _Board(ZERO_2W, 427_000)
        out = board_profile.describe(**board.kw)
        self.assertEqual(out, {"board": "Raspberry Pi Zero 2 W", "mem_mb": 416, "profile": "lite"})


class LightBootTests(unittest.TestCase):
    """enroll.py turns the desktop off on a light board, once."""

    def _run(self, profile, default_target, env=None):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=default_target + "\n", stderr="")

        with mock.patch.object(board_profile, "detect", return_value=profile):
            changed = enroll._light_boot(env or {}, run=run)
        return changed, calls

    def test_a_light_board_boots_to_the_console(self):
        changed, calls = self._run("lite", "graphical.target")
        self.assertTrue(changed)
        self.assertIn(["systemctl", "set-default", "multi-user.target"], calls)
        self.assertIn(["systemctl", "stop", "--no-block", "display-manager.service"], calls)

    def test_once_done_it_does_nothing(self):
        changed, calls = self._run("lite", "multi-user.target")
        self.assertFalse(changed)
        self.assertEqual(calls, [["systemctl", "get-default"]])

    def test_a_full_board_keeps_its_desktop(self):
        changed, calls = self._run("full", "graphical.target")
        self.assertFalse(changed)
        self.assertEqual(calls, [])

    def test_keep_desktop_opts_out(self):
        changed, calls = self._run("lite", "graphical.target", {"BEEMONITOR_KEEP_DESKTOP": "1"})
        self.assertFalse(changed)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
