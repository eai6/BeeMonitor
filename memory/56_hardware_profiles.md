# 56 — Units detect their board and run light when they need to

**Status:** built 2026-10-23 (`hardware/board_profile.py`); not yet tried on a real Zero 2 W. Supersedes the hand-applied preset in [[20_pi_zero2w_lite_profile]].

## Why
One golden image (64-bit Bookworm, built on a Pi 4) should run on a Pi Zero 2 W without anyone editing files on the card. Today a Zero 2 W would load PyTorch at recorder start (hotel detection) and again daily at 15:00 (calibration), which 512 MB can't hold, and records 1080p at 25 fps.

## Audit (what's there now)
- Every recorder knob is an env var read once in `hardware/motion/config.py` (`_env_int` / `_env_bool`), from `/etc/beemonitor/uploader.env` via each unit's `EnvironmentFile`.
- YOLO on the unit: `motion/roi.py` `_resolve_record_roi` → `detect_hotel_roi` at every recorder start unless `BEEMONITOR_HOTEL_ROI_DETECT=false` or a dashboard ROI override exists; `motion/calibrate.py` via `beemonitor-calibrate.timer` (daily 15:00).
- Defaults: 1920×1080, 25 fps, activity crops on.
- Dashboard overrides (`motion/overrides.py`: motion_tuning, activity frames, record settings) are applied on top at runtime.
- Telemetry reads `/proc/device-tree/serial-number` but doesn't report the board model or RAM.
- On-device bee confirmation is gone, so nothing else loads a model.

## Plan
1. **`hardware/profile.py`** (stdlib only): `detect()` → `"lite"` when `/proc/device-tree/model` says "Zero 2" or `MemTotal` < 1 GB, else `"full"`. `BEEMONITOR_PROFILE=full|lite` forces it.
2. **Lite defaults**, applied with `os.environ.setdefault` at the top of `motion/config.py`, so anything set in `uploader.env` still wins and dashboard overrides still apply on top:
   `BEEMONITOR_MAIN_W=1280`, `BEEMONITOR_MAIN_H=720`, `BEEMONITOR_FPS=15`, `BEEMONITOR_HOTEL_ROI_DETECT=false`, `BEEMONITOR_ACTIVITY_FRAMES=false`.
3. **Calibration skips itself on lite** (logs "skipped: lite profile") instead of the timer being disabled, so the unit file set stays identical across boards.
4. **Telemetry reports** `board` (model string), `mem_mb` and `profile`; the device page shows e.g. "Raspberry Pi Zero 2 W · light mode".
5. **No desktop in light mode.** The golden image is Raspberry Pi OS *Desktop* (confirmed 2026-10-23); the desktop session alone takes a large share of 512 MB. On a lite board, the root boot step (`enroll.py`, already a root oneshot before the app services on every boot, so no new unit file for the updater to install) sets the default target to `multi-user.target` (console) and switches to it, once. Packages stay on disk; full boards keep their desktop (the Tk focus preview needs it). `BEEMONITOR_KEEP_DESKTOP=1` opts out.
6. Ships through the normal remote update; Pi 4/5 units detect `full` and change nothing.
7. Tests: `hardware/test_profile.py` with fake model/meminfo files; defaults don't override explicit env.

## Decided
- Lite = "Zero 2" in the model, or under 900 MB reported (any 512 MB board); a 1 GB Pi 3B stays full.
- Lite records 720p / 15 fps to start; raise it in uploader.env if the board keeps up.

## Open questions
- Focus on a lite unit: the dashboard's stills/focus tools instead of the on-screen Tk preview.
- Validate on a real Zero 2 W: `free -m` under budget, clips record, no torch import in the journal.
