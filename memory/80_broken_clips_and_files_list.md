# 80 — Broken clips, and a files list on the batch page

**Status:** planned (2026-11-12), waiting for the go-ahead. Design canvas: https://claude.ai/artifact/Cxz2ZaC48jvnUBz5s8pVDs

**Goal:** a clip whose file can't be played is found and named as a broken file, not sent to the GPU to fail as "another reason". The batch page offers a list of every clip in the batch: what ran, what failed, and what was skipped, and why.

## What happened (2026-11)
- A Mendels + Natalies batch: 1,113 failed runs.
- 748 were `No space left on device`: the worker never emptied `/tmp/beemonitor_work/<job>` after a clip (cleanup dropped in the AWS migration, 56ce13b). Fixed in `predict_fn` (`finally: pipeline.cleanup`), not yet deployed.
- 366 were `Cannot open video`: the files on S3 are broken. A bucket scan of 111,119 MP4s found three kinds:
  - **empty**: 0 bytes (284).
  - **unclosed**: header and video data, no `moov` index at the end, `mdat` size 0. The recording stopped before the file was closed (power loss, reboot). 80+ (e.g. `mendels_2024-04-17_19_00_00`).
  - **truncated**: the `mdat` box claims more bytes than the file has; the upload was cut short (e.g. `natalies_2024-05-22_19_50_00`: says 291 MB, is 278 MB).

## Audit
- No file-health state is stored. `Video.Status` has uploading/ready/processing/archived, but every path writes READY.
- The probe (`thumbnails.extract_thumbnail`, background thread) fills fps/duration/size; when cv2 can't open the file it swallows the failure, so the clip just has blank fps/duration.
- Upload "complete" handlers: `api/web_uploads.py:118`, `api/multipart.py:150` (no existence check), `api/uploads.py:168` (device), `api/pipelines.py` uploads/complete, the legacy DRF create, `sources` bulk import (`s3://` keys, measured by ffprobe over a URL).
- Launch: `run_on_videos` → `engine.launch_batch`. Clips left out (raw deleted, too short, over cap) get no PipelineRun, and appear only in the flash message, so the batch page never shows them.
- Failures panel: `failures.py`, which already has `unreadable_clip` (not retryable) and `disk_full`, added with the worker fix.
- Batch downloads: `aggregate.available_downloads` → `batch_combined_csv` → `sharing.batch_csv` → `_csv_response` (streamed).
- Big batches (memory/76, 77): never load `context`; stream exports; nothing slow per clip inside a request.

## The check
- `apps/videos/filecheck.py`: `check(storage, key, size)` walks the MP4's top-level boxes with S3 range reads (16 bytes per box, about 3–4 reads per file, no download).
- Result: `ok`, `empty`, `unclosed`, `truncated`, `not_mp4`, or `unknown` (couldn't read; never blocks).
- MOV and MP4 share the box layout. AVI and other containers get `ok` without the check (they go through transcode, which has its own ffmpeg failure).
- Stored on the Video row: a new `file_state` field (blank = not checked yet) plus `metadata.file_check = {state, checked_at, detail}` (detail: "says 291 MB, file is 278 MB"). One migration; indexed so batch counts are one query.

## When it runs
1. **On arrival**: in every upload-complete handler, synchronously (about 0.1 s), and for bulk imports in `copying.measure`.
2. **At launch**: any clip in the launch turn with no `file_state` is checked first (threaded, 16 at a time; about 2 s per 300-clip turn). This also covers the 111k clips already stored, so no separate backfill is needed (a management command exists for anyone who wants one).
3. **On the GPU, as a last guard**: after the download, `CloudPipeline.process` runs the same box walk on the local file and raises `Broken clip file: <kind>`, instead of cv2's "Cannot open video". When the platform sees that error, it saves the state on the Video.

## What a broken clip does in a batch
- It still gets a PipelineRun, so the batch page and files list know about it, but it is never sent to the GPU: status FAILED, error `Skipped: broken file (<kind>)`, no GPU time, no cost.
- The batch page gets a **Skipped** filter next to Failed and Cancelled. A skipped run is a failed run whose error starts with `Skipped:`, the same way Cancelled is read.
- Failures panel: a "Broken clip files" group, split by kind, not retryable, with a "Download the list" link. "Re-run these N clips" leaves skipped clips out.
- Launch message: "Launched 1,113 clips · 366 skipped: broken files".
- Re-run on a skipped clip checks it again (a file replaced on S3 then runs).

## Files list (CSV)
- New download on the batch page: **Files list (CSV)**, one row per clip in the batch: `files` kind in `sharing.batch_csv`, `batch/<id>/combined/files.csv`.
- Columns:
  - clip
  - site
  - unit
  - recorded_at (local)
  - s3_key
  - size_mb
  - duration_min
  - file_state
  - file_detail
  - run_status (completed / failed / skipped / cancelled / running / pending)
  - failure_cause (the failures.py title)
  - error (the recorded text)
  - gpu_hours
  - run_id
- Built from `_batch_runs` (slim) plus one `Video.only(...)` query, streamed 500 rows at a time; no `context`, no S3 calls.
- Also on the share link's downloads.
- Clips left out at launch for other reasons (raw deleted, too short) have no run and aren't listed. That's out of scope; worth a later note.

## Cost
- Web only, plus the small worker guard (rides the next GPU image).
- S3: about 4 range GETs per clip, once. 111k clips is about 450k GETs, about $0.18.

## Open questions
- Should broken clips be hidden from the Videos page or the launch picker, or only flagged? Planned: flagged with a badge; still launchable, and skipped at launch.
- Try to recover unclosed recordings (rebuild the index from a good clip of the same camera, e.g. untrunc)? Planned: not in this change.

## Tests
- `filecheck`: ok, empty, unclosed, truncated, not_mp4, 64-bit box sizes, moov first (faststart), an S3 error returns `unknown`.
- Launch: a broken clip gets a Skipped run, is never invoked, and costs nothing; an unchecked clip is checked at launch.
- Re-run leaves skipped clips out; a re-run of one skipped clip re-checks it.
- The files CSV: one row per run, every status, streamed; the batch page still never selects `context`.
- The worker guard raises `Broken clip file`; failures.py maps it.
