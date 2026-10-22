# 53 — Analyse part of a clip

**Status:** built 2026-10-22. Design: canvas board "Analyse part of a clip" (https://claude.ai/artifact/9kjvmu9SWhZQ47CzB2eAB1).

## Why
A 34-minute ground-nest clip: the user wants events from 1 or 5 minutes of it, not the whole clip (time and credits).

## What exists
The GPU worker already analyses a frame range: `start_frame` / `end_frame` in the payload (`sagemaker_backend/inference.py:191`, `cloud/wrapper/pipeline.py:687`), outputs stay **absolute** in the clip. Production uses it only to split long clips (`analysis/views.py:497 _chunk_ranges`, `BEEMONITOR_CHUNK_TRACKING=1`). Nothing lets a user ask for part of a clip.

## Plan
- **Run option** on the Processing page, under the pipeline picker: *How much of each clip* — the whole clip, or **from** (m:ss) **for** (1 / 5 / 10 / 15 minutes). Applies to every selected clip; clipped to each clip's length (a clip shorter than the start is left out, and the message says so).
- Stored on the run (Input Video step config: `start_seconds`, `end_seconds`), carried into the GPU job config, turned into frames with the clip's fps when the job is sent. Long windows are still chunked, inside the window.
- **Credits** estimated on the window, not the whole clip (5 of 34 minutes ≈ 0.1 credits on YOLO).
- **Cache**: the window is part of the job config, so 0–5 min and the whole clip are different results; neither overwrites the other.
- **Results** say "Analysed 0:00–5:00 of 34:32"; event times stay positions in the whole clip. A button "Analyse the whole clip" re-runs without the window.
- API: the same two fields on pipeline runs and jobs.

## Built
- Run panel: "How much of each clip" (whole / part: from m:ss for 1, 5, 10, 15 minutes or to the end). No cost line, per the panel's existing rule (no pre-run estimates); credits are still checked on the part (`quota.window_seconds`, `pipeline_work(..., window)`).
- `engine.clip_window` trims per clip; clips ending before the start are left out and counted in the message. `steps_with_video(..., window)` puts `start_seconds`/`end_seconds` on Input Video; `_exec_input_video` passes them on; `build_detect_and_track_config` copies them into the job config (and so the cache key).
- `_spawn_gpu_job` forwards them; `_launch_gpu` → `_window_frames` (fps) → `start_frame`/`end_frame` on one invocation, or `_chunk_ranges(video, kind, window)` splits inside the window.
- "Analysed 2:00–7:00 of 34:32" on the run page and the job results page (`apps/videos/timecode.py`). API: `start_seconds` / `end_seconds` on `POST /api/v1/pipelines/<id>/run/`; a job created with them in `config` gets the same treatment.

**Feedback:**
