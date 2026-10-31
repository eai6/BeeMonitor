# 69 — Re-track on the web: only detection needs the GPU

**Status:** built (2026-10-31) with the proposed answers; the worker frame log needs a GPU image push. Design canvas: https://claude.ai/artifact/JxnhiViavkbqSfs2DSQZGn. Requested by the user, who changed "Keep a lost track" from 3 s to 1 s and wants to re-extract without the GPU: "the only part of this pipeline that needs GPU is the detection (and identity)… edit the pipeline, reconfigure and re-extract."

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## What exists (audit, 2026-10-31)
- **Detection and tracking are one online GPU pass** (`src/beemonitor/tracking/bee_tracking.py`).
  - Every frame is decoded, and detection is **motion-gated** (`two_mode_tracking`):
    - **Motion mode:** a stateful MOG2 blob check on the ROI; no YOLO. `tracker.update([])` is still called each frame, which ages the tracks.
    - **On motion:** the last 0.5 s of frames are batch-detected and replayed into the tracker with their old frame numbers (the *lookback*). The trigger frame is fed twice.
    - **Tracking mode:** YOLO runs every frame until 30 frames pass without motion.
- **`detections.csv`:**
  - Columns: `frame, x1..y2, cx, cy, confidence, source, taxon, mode`.
  - It holds exactly the rows `update()` received, after the detector's confidence threshold, NMS and class filter. Nothing is filtered by ROI.
  - **Not recorded:** which frames ran YOLO vs. were motion-only vs. found nothing; the motion decisions; the frame range. `mode` is the mode *after* the frame. The trigger frame appears twice. The sort within a frame isn't stable.
  - A clip with no tracks has no file.
- **BeeTracker** (`tracking/mot/bee_tracker.py`):
  - **Inputs:** `update(rows of [x1,y1,x2,y2,conf,source,taxon], frame_num)`. It only needs numpy and scipy, but importing it through the package pulls in cv2 and ultralytics, so it would be vendored.
  - **Settings:** web seconds become frames by `int()` on cv2's float fps; `max_resurrection_seconds=0`; `match_distance_multiplier` is always 8.0.
  - **The node's "Match overlap (IoU)"** doesn't affect matching at all: it sets the **detector's NMS**, so it can't change from saved detections.
- **The other trackers** (bytetrack, botsort) need ultralytics and torch, and botsort needs the frames. ocsort and sfsort need only numpy plus lap or scipy.
- **After tracking:**
  - **Backfill** of confirmed tracks' earlier frames.
  - **Lost rows:** `detected=0`, with the last box but the Kalman centre.
  - **Duplicates** are removed.
  - **No** smoothing or merging.
  - **Chunked clips:** one tracker per chunk; the web namespaces their ids and stitches them (`analysis/chunk_stitch.py`).
- **Crops:**
  - `track_crops.csv` has `track_id, frame, crop_key, sharpness`, at most 50 per track, with **no box**.
  - `track_votes.csv` has one species reading per crop, by `track_id, frame`.
  - A crop's box can be recovered from the old tracking CSV (old track, frame, `detected=1`), so crops can be **remapped** to new tracks by box overlap. They can't be re-cropped or re-picked.
- **Keyed by track id:**
  - the overlay (stored beside the tracking CSV);
  - `identification.by_track`, `crops_manifest`;
  - Identify Species' groups;
  - **event review votes** (`action|reference|track|frame`).
  - The worker writes no events; the web derives everything from the tracking CSV.
- **Web plumbing that already exists:**
  - writes to S3 (proxy, overlay);
  - a background reconcile thread with a recount sweep every 120 s;
  - `regroup.apply_pipeline` + `pipeline_changes` ("Update this batch") already tell web-side from GPU-side changes;
  - outdated review votes are tracked as "gone".

## The plan

### 1. A CPU replay of BeeTrack from saved detections
- **The tracker:** `apps/pipelines/retrack.py` with a vendored copy of BeeTracker. Add scipy to the web requirements; CI builds the image.
- **The replay, for each clip (each chunk for chunked clips, then the existing stitch):**
  - feed every frame from the first to the last analysed, calling `update([])` where `detections.csv` has no rows;
  - drop the lookback rows of a frame that also has a later tracking row (the trigger frame fed twice);
  - keep the original row order.
- **Settings** come from the run's MOT node exactly as the worker resolves them: the same defaults, `int()` frames, and the clip's fps.
- **Output:** the same tracking CSV shape (`detected`, backfill, lost rows) uploaded to `processed/…/retrack/<run>/tracking.csv`. The run's MOT output gets the new `tracking_csv_path`, keeping the old one as `gpu_tracking_csv_path`. The overlay rebuilds beside it by itself.
- **Close, not exact, for clips already run:** the motion-mode frames aren't recorded, so a quiet stretch is replayed as empty updates. The motion gate also called the tracker on those frames, so lost tracks age at the same rate. The lookback order can't be rebuilt exactly.
- **Agreement check:** a replay with the **unchanged** settings is compared with the GPU's own tracks (tracks matched by box overlap, events equal). The batch page shows that agreement before the user relies on it.

### 2. Exact from now on (one worker change)
- The worker also uploads a small **frame log**: per frame, whether YOLO ran (live or lookback) and the order of `update()` calls, so a replay matches exactly.
- **Deploy order:** a GPU image change, so push the code first, then bump the image tag; the user runs Pulumi (memory: GPU image tag = push head). Older clips keep the close replay.

### 3. Species and groups follow the new tracks
- Each crop's box is looked up in the GPU tracking CSV and assigned to the new track with the highest overlap at that frame (≥ 0.5); otherwise it's left out.
- `track_votes.csv` is rewritten with the new ids. Each track's species is re-voted, and the groups are recounted by Identify Species as now.
- `crops_manifest` is remapped the same way, so the player's crop strip follows.

### 4. Review votes follow their events
- After a re-track, each event vote is moved to the new event with the same action and tube within 0.5 s, when exactly one matches. The others stay as "gone" (already shown as no longer detected).
- Trip votes follow their exit and entry events the same way.

### 5. "Update this batch" does it
- **Tracking changes become web-side:** `pipeline_changes` moves **Keep a lost track** and **Confirm a track after** (BeeTrack, plus sfsort and ocsort if added) from GPU to web.
- **Still GPU:** the detector, its confidence, what to detect, **Match overlap (IoU)** (renamed in the node to say it's the detector's overlap filter), bytetrack and botsort, and the species model.
- **Order:** **Update this batch** re-tracks first, then recounts.
  - It runs in the background, clip by clip, a few at a time inside the reconcile thread.
  - Progress shows on the batch page ("Re-tracking 46 of 117 clips").
  - The clips stay usable meanwhile with their old counts.
- **Clips with no `detections.csv`** (run before the worker saved it, or with no tracks at all) are listed: they need one GPU run.

### Not planned now
- Re-running detection cheaper (sampling, smaller models): memory/48.
- Re-cropping or re-picking the sharpest crops for new tracks.

## Build order
1. **Vendor BeeTracker; build `retrack.replay(detections_df, settings, frames)`.** Tests on synthetic detections: same settings give the same tracks as a direct run; 1 s vs 3 s lost windows split a track.
2. **Agreement check against real GPU output.** Use a fixture from a small saved clip, if one can be added without real footage, else synthetic.
3. **Store the new tracking CSV; repoint the run; overlay; remap crops and votes; re-vote species.**
4. **Move review and trip votes.**
5. **"Update this batch":** re-track in the background with progress; the GPU/web split in `pipeline_changes`.
6. **Worker frame log** (a separate push and image tag).

## Questions
1. **Old clips:** is a close replay acceptable for clips already run, with the agreement check shown (proposed)? Or only exact replays (clips run after the worker change)?
2. **Review votes:** should they move to the matching new event automatically (proposed), or stay as "gone" for you to review again?
3. **The worker frame log** (a GPU image change that you deploy with Pulumi): now (proposed), or later?

**Feedback:**

## Built (2026-10-31)
- **`apps/pipelines/retrack/`:**
  - `bee_tracker.py`: a vendored copy; a test keeps it identical to the source.
  - `replay.py`:
    - `tracker_params` holds the worker's defaults plus what the node sends;
    - `calls_from_log` is the exact replay;
    - `calls_from_detections` rebuilds the calls: every frame in order, with empty updates, and each lookback run replayed before its last frame over a 0.5 s buffer;
    - `run_calls` / `tracking_rows` mirror `process_frame`, with backfill read only after current calls.
  - `run.py`:
    - `retrack_run` handles chunks (each its own tracker, then `chunk_stitch`);
    - crops and species votes are remapped by box IoU ≥ 0.5 and re-voted with the worker's `_summary`;
    - new `tracking.csv`, `track_votes.csv` and `track_crops.csv` are stored under `retrack/<run>/` beside the GPU result;
    - the run's outputs point there, with `gpu_result` kept;
    - `effective_result(job)` gives clip pages the run's view.
  - `votes.py`: event and trip votes move to the unique new event with the same action and tube within 0.5 s. Votes from other runs of the same job are copied, not moved.
  - `check.py`: the replay with the GPU's own settings vs the GPU's tracks, comparing boxes tracked, same-track share and events.
- **When re-tracking happens:**
  - **Update this batch** copies BeeTrack's `max_age` / `min_hits` (`REPLAYABLE`) into each run's MOT node, saving what the GPU used under `_gpu_mot`. Those runs are queued for the sweep.
  - `recount_run` re-tracks first when `needs_retrack`, then moves the votes.
  - A clip without detections gets a `retrack_note`.
  - `batch_template` drops `_gpu_mot`, so added clips are tracked on the GPU with the node as it is now.
- **Clip pages read the run's tracks:** the results page, the overlay (stored beside the new CSV), crops (with `?run=`), the CSV downloads, `job_tracks` and period review.
- **Batch page:**
  - change list: "Keep a lost track: 3 → 1 (tracks again on the web)";
  - progress; clips re-tracked (exact or close); clips that can't be re-tracked;
  - a **How close is the web re-tracking?** check (`pipelines:batch_retrack_check`, cached a week).
- **MOT node:** BeeTrack's "Match overlap (IoU)" is renamed "Detector duplicate-box overlap (IoU)", since it is the detector's NMS.
- **Worker:**
  - `bee_tracking.py` numbers every `update()` call;
  - `detections.csv` gains `seq`, with a stable sort;
  - `update_log.csv` (`seq, frame, kind`) goes to `summary_stats.update_log_csv`.
  - **Deploy:** GPU image tag, then Pulumi by the user. Clips run before that replay closely; a chunked clip's later chunks also replay closely, since the merge keeps one chunk's log.
- **Tests:** `apps/pipelines/tests/test_retrack.py` (13) and `src/beemonitor/tests/test_detections_export.py::UpdateLogTests`.

