# 40 · Full-resolution 64 MP stills

Status: **plan, awaiting approval** (2026-09-26)
Design: https://claude.ai/artifact/HYjUiHDMeCqLFJVBEn4mnA (boards "Full-resolution
stills" and "Still viewer", bottom of the canvas)

## Why

The recorder writes 1920×1080 H.264 (`hardware/motion/config.py:41`,
`recorder.py:122`): the Pi 4's hardware encoder tops out at 1080p, and the
OV64A40 reads its full 9152×6944 at only ~2–2.6 fps (Arducam), so 64 MP
*video* is out on any Pi. Stills are not limited by the encoder: a periodic
full-sensor JPEG carries the camera's real resolution for species ID, nest
checks and figures, while video stays 1080p for tracking.

## What exists (audit)

- `runFocus.py` saves "full-resolution" stills, but from the video config's
  1920×1080 main stream — no still at full sensor size exists anywhere yet.
- `recorder.py` loop: lores-paced state machine; clips open/close through a
  CircularOutput pre-roll buffer; activity crops grab `capture_array("main")`.
- On-demand capture: telemetry drops `capture.request` in the queue, the
  recorder writes a JPEG there (`telemetry.py:1525`), sent with a heartbeat
  (720 px, `TELEMETRY_IMAGE_HEIGHT`). Periodic telemetry stills exist but are off.
- Uploads: `uploader.py` uploads `*.mp4` and activity frame groups
  (`*/frames/*.json` + jpgs, WiFi) to `/api/.../frames` → `monitor.ActivityFrame`
  (kind crop | wide). Settings are dashboard-pushed as JSON files the recorder
  hot-reloads (`record_settings.json`).

## Design

**Device**
- New setting `stills_interval_min` (0 = off; 15 / 30 / 60), pushed like the
  other record settings, hot-reloaded. Default **off**; turn on per device.
- Only on the 64 MP sensor (`model_of(cam) == "ov64a40"`); ignored otherwise.
- Taken only inside the recording window and **never while a clip is open**:
  when due and idle, the recorder stops the encoder, `switch_mode_and_capture`
  a still configuration at the full sensor size (`buffer_count=1`), writes
  `stills/<ts>.jpg` (q90) + `.json` (taken_at, w, h, lens position, exposure),
  then restores the video configuration and restarts the encoder and motion
  gate warm-up. Expected pause ≈ 2 s; a trigger in that gap is missed.
- If a still is due while a clip is open, it is taken right after the clip closes.
- Memory: a 64 MP RGB buffer is ~190 MB; the Pi 4's CMA pool may not fit it
  beside the video buffers. The switch frees video buffers first; if allocation
  still fails, fall back to the 16 MP mode (4624×3472) and record which was used.
- Orientation: same transform as video (ISP flips); a 90/270° unit is rotated
  in software like `runFocus._save_still`.
- "Take one now" from the dashboard reuses the `capture.request` path with a
  `full` flag.

**Upload** — `uploader.py` picks up `stills/*.json` groups over **WiFi only**
(cellular never: ~15 MB each), posts to a new `/api/devices/<id>/stills/`
presigned-PUT flow (same as videos), deletes local copies once confirmed.
Local cap: oldest stills dropped past 2 GB so a device off WiFi can't fill the card.

**Web**
- Model `DeviceStill` (device, taken_at, storage_key in raw-videos under
  `stills/`, width, height, bytes, lens_position, source mode 64|16 MP).
- Advanced settings row (design board 1), with a live size estimate.
- Device page "Stills" section: grid by day, "Take one now".
- Viewer: fit / 100% with pan, download original, prev/next. Thumbnails are a
  1280 px derivative made on upload (a browser should not pull 15 MB per tile).

## Cost

~15 MB/still. Every 30 min over a 14 h window ≈ 28/day ≈ 420 MB/day/device in
S3 (~$0.30/month/device at Standard, less after the 90-day Glacier IR rule).

## Build order

1. Device: still capture in the recorder behind the setting (off by default);
   test on one 64 MP unit: pause length, CMA, sharpness vs the 1080p frame.
2. Upload path + API + `DeviceStill` + thumbnails.
3. Web: setting, device-page gallery, viewer.
4. Roll out: on per device from the website.

## Open questions

1. Default interval when turned on: 30 min?
2. Also take one at the start of each clip? (Better for species ID; costs a
   2 s pause before each clip — it would miss the bee's arrival.) Proposal: no.
3. Keep stills forever, or expire after N days?
