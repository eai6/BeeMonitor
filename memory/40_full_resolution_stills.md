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
- New setting `stills_interval_min` (0 = off; 15 / 30 / 60 — 15 is the
  floor, decided 2026-09-26), pushed like the
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

## Part 2: pipelines on stills

Design: boards "Pipeline editor: Stills input" and "Run on stills + results".

### Audit (2026-09-26)

- A run's input is ONE video: `engine.py:60-101` injects a video id into every
  `input.video` step; `launch_batch` makes one run per video; schedules
  (`DevicePipelineSchedule`, `devices/scheduling.py`) pick a device's videos in
  a window. `analysis.Job.video` is a required FK (`analysis/models.py:49`) and
  `config_hash`, `_spawn_gpu_job` (`analysis/views.py:186`), chunking, the
  StepResult cache key (`engine.py:30-46`), `aggregate.py` and the batch page
  (`<video>` player, thumbnails) are all video-keyed.
- A hidden legacy block `input.image_set` (`registry.py:81`) outputs `frames`
  that nothing consumes — revive/replace it.
- Closest GPU path: `_pre_annotate` (`inference.py:246`) — detector-agnostic
  `_detect(frame)` for SAM 3 and YOLO, returns `frames:[{frame_number, boxes}]`;
  only its loop reads a video. `detection_count` distinct/modal
  (`executors.py:771`, `ops.py:989`) already consumes that shape.
- Per-image blocks: Detect objects, Count detections (total/per image/over
  time), reference layout. Video-only: Track (MOT), events, foraging trips,
  interactions, visitation, colony activity (need motion over time).
- Species: BeeMachine only runs inside tracking; BioCLIP is a separate CPU
  endpoint used on activity-frame crops (`apps/monitor/pipeline.py`) — reusable
  on detection crops.
- **64 MP on the GPU:** YOLO at its default 640 px would shrink a ~60 px bee to
  ~4 px. SAM 3 resizes to ~1K. Detection must be **tiled** (~1280 px tiles,
  ~20% overlap, cross-tile NMS): ~40–60 tiles per still. A decoded 64 MP frame
  is ~190 MB, so no batching of whole frames.
- Upload caps: `api/frames.py:42`, `api/heartbeat.py:46` cap at 5 MiB —
  stills need the presigned-PUT flow (part 1).

### Design

- **Block** `input.stills` (replaces `input.image_set`): outputs `images`.
  `detect.objects` and the reference-layout blocks accept it; Count
  detections gets a per-image mode and an over-time mode keyed on each still's
  `taken_at` (not fps). Validation refuses tracking/visit/foraging blocks
  downstream of stills.
- **Identify species (new block)**: BioCLIP on each detection's full-resolution
  crop, reusing `apps/monitor` (location-prior candidate taxa).
- **Run identity**: one run covers a SET of stills (device + time range), not
  one run per still. `PipelineRun` gets `stills` (M2M to `DeviceStill`) and a
  `source` kind (video | stills). `analysis.Job.video` becomes nullable with a
  `kind` and an `input_keys` list, so the poller, pricing and status stay one path.
  Cache key hashes the image keys + config.
- **GPU task `detect_images`**: payload `{task, image_keys, classes,
  detector_kind, confidence, tile, overlap}`; downloads each still, tiles,
  runs YOLO or SAM 3 per tile, merges with NMS, returns
  `images:[{key, width, height, boxes (full-res px)}]` and writes a JPEG crop
  per detection for species ID. Batches ~20 stills per invocation (like
  `sample_label`) so one cold start covers many.
- **Starting runs**: a "Run on stills" picker (device, date range, count,
  GPU estimate) beside the Processing hub's video list; device schedules gain
  "run on new stills" (same reconciler clock).
- **Results page**: bees-per-still over time chart, grid of stills with box
  overlays (boxes drawn in the browser on the 1280 px thumbnail), species per
  still, CSV (still, taken_at, class, count, boxes, species).

### Cost (to be measured)

YOLO: ~50 tiles × ~20 ms ≈ 1 s GPU per still. SAM 3: 0.58 s per tile per
class ≈ 30 s per still per class — ~50× YOLO; offer it, default YOLO.

### Build order (part 2, after part 1 ships)

5. `detect_images` GPU task + tiling + tests (image built by CI, pulumi up by user).
6. Job/Run model changes (nullable video, stills set) + migrations.
7. `input.stills` block, validation, Count per-image/over-time, species block.
8. Run-on-stills picker, schedules, results page.

## Open questions

1. Default interval when turned on: 30 min? (15 min is the minimum offered.)
2. Also take one at the start of each clip? (Better for species ID; costs a
   2 s pause before each clip — it would miss the bee's arrival.) Proposal: no.
3. Keep stills forever, or expire after N days?
4. Pipelines on stills: default detector YOLO (cheap) with SAM 3 as an option?
5. Run species ID (BioCLIP) on every detection, or only when the block is added?
