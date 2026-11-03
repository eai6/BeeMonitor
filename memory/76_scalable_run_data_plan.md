# 76 — Run data that scales with the project

**Status:** built (2026-11-03), all four phases. Not yet committed.

**Goal:** no page, download or background loop may cost more as a project gains clips, except what it shows: a page listing N clips reads a few KB per clip, and one clip's heavy data is read only when that clip is opened or exported.

## Audit (2026-11-03)
- **Storage:** each run's results live in one JSON field, `PipelineRun.context`. That holds every step's output: event rows, interaction rows, the GPU result with `summary_stats` (stored twice, in the tracks step and in the species step), per-track species rows, `track_groups` and `names_seen`.
- **The flaw:** many pages load that field for every clip, to use a few numbers from it. In a simulation, a 400-clip batch page took about 440 MB.
- **Production, 2026-11-02/03:** 6 workers were killed for memory in 12 h, each just after the 408-clip batch page loaded.
- **Readers that load every clip's context** (none use `.only()` / `.defer()`):
  1. **Batch page, batch CSVs and review metrics** (`views._batch_runs`). Each filter change reloads the whole batch, and it uses almost every part of the context.
  2. **Public share link** (`sharing.shared_runs`). It reloads the whole batch on every thumbnail, video and overlay request, which only need `steps` or `gpu_job_id`.
  3. **Event-classifier dataset** (`eventml.dataset`): every run of every reviewed batch, on a page GET, a CSV download and training.
  4. **Period review** (`period.runs_for_videos`, `batch_runs`) and **trips CSV** (`trip_rows`). The trips CSV joins 2 full runs per trip review.
  5. **Layout save** (`recount.runs_for_clips`): a whole site's runs, only to count them.
  6. **Others:**
     - `grouping_preview`: whole batch;
     - `batch_retrack_check`: whole batch, uses 2;
     - `batch_apply_groups` with "reviewed": the batch loaded twice;
     - `reconcile_user_runs`: every 10–12 s per open tab, for job ids only.
- **Hot single-clip paths:**
  - results page: about 10 full fetches of the same run when not pinned, 2 when pinned;
  - player events: 4–5;
  - the `run_status` poll every 4 s re-saves the full context for every finished job.
- **Also:**
  - `batch_rows` loads every `JobResult` whole;
  - `primitive_csv` makes 2 queries per run;
  - thumbnails download whole clips;
  - gunicorn workers are never recycled (2 workers × 4 threads on 1 vCPU / 2 GB).
- **Instance:** the user keeps it at 1 vCPU / 2 GB, so the fix must be in the code.

## Design
**A. A per-run summary.**
- `PipelineRun.summary`: a JSON field of a few KB, kept beside `context`.
- It is written whenever the context is. All context writes go through one helper, `runs.save_context(run, context)`, which covers `save()`, the recount and retrack queryset updates, and engine saves.
- **Contents**, exactly what multi-clip pages use:
  - video id and step kinds present;
  - event counts by action and the interaction count;
  - the **event index**: `[key, action, reference, track, frame, ml_confidence, at_edge]` per event, about 60 B each;
  - `filtered_out` (`event_keys`, `by_group`);
  - insect-group summary (`group_filter`, `leave_out`, `group_options`, `group_counts`, whether `track_groups` exist);
  - step errors (failed runs);
  - `retracked` settings and note;
  - fps;
  - whether identification `by_track` is present;
  - photo detections;
  - interactions `per_reference` and total duration (for the share page's contact table);
  - the job ids of steps still running (for the poll).
- **A version number.** The summary carries one. A backfill migration fills it and logs the size distribution (p50 / p95 / max of `context` and `summary`) to CloudWatch, so we see the real numbers.

**B. Multi-clip readers use `.defer("context")` and the summary.**
- batch page, metrics, update/apply, retrack-check (loads only the 2 it uses), remove/cancel/retry;
- share link: the page uses the summary, and per-file requests use column queries only;
- period review, trips CSV, layout save (a count query), grouping preview (one run at a time, up to its deadline);
- the poll (`reconcile_user_runs` from the summary's running job ids);
- `batch_rows` reads `JobResult` with `.only()`.
- The 300-clip in-page cap from memory/27 can then go.

**C. Heavy reads, one clip at a time and streamed.**
- Batch CSV downloads (events, interactions, tracking, trips) become `StreamingHttpResponse`, walking runs with `.iterator()` in small chunks. Each run's reviews and added events are loaded per chunk, not per run.
- The event-classifier dataset first picks the runs with votes in SQL, then reads them one at a time.

**D. One fetch per clip per request.**
- `_run_for_job` remembers its answer on the job object, as `pin_run` does, so the results page does 1 fetch, not 10, and the player events 1, not 5.
- The `run_status` poll stops re-saving the contexts of jobs already applied.

**E. Server hygiene.**
- gunicorn `--max-requests 300 --max-requests-jitter 50`, so a worker that grew is replaced.
- One recount loop for the instance instead of one per worker (a database lock).
- Thumbnails read the frame they need through a ranged read of the presigned URL, not a whole-clip download.

**F. Guardrails.**
- A test builds a batch of 1,000 synthetic runs with 200 KB contexts. It asserts that the batch page, metrics, share link and period page never select `pipelines_pipelinerun.context`, and that peak memory stays under a fixed budget.
- The existing slow-request log stays.

**Later, only if the summary is not enough (more than 10k clips per batch):**
- a `RunEvent` table (one row per event, indexed), so review metrics become SQL aggregates joined to `EventReview`;
- moving interaction rows and per-track data out of `context` into their own table or S3.

## Phases (each shippable alone)
1. **Summary and batch page.** Phases A and B for the batch page, review metrics and share link, plus E's gunicorn recycling. This is the change that stops the memory kills.
2. **Hot single-clip paths (D)** and the remaining multi-clip readers (B).
3. **Streaming downloads (C)** and the event-classifier dataset.
4. **Guardrail tests (F)**, thumbnail ranged reads, and removing the 300-clip cap.

## Open questions
- Is the summary-first approach right, or should we go straight to the `RunEvent` table? The recommendation is summary first: smaller and faster to ship, and it fixes today's cause.
- Is it OK that a run's summary is rebuilt from its context on deploy? The backfill migration reads each run once, in chunks.

## Built (2026-11-03)
- **`PipelineRun.summary` (runsummary.py): the slim copy** has the same shape as `context`, with heavy parts dropped:
  - event rows keep key, score and time fields;
  - interaction rows, species rows and `names_seen` are dropped, with counts kept;
  - `track_groups` and `identification.by_track` become `{"_slim": true}`;
  - `summary_stats` values over 2 KB are dropped, except photo and fps.
- **Keeping it in step:**
  - `save()` rebuilds it whenever the context is saved, and so does `PipelineRun.write_context` (recount and retrack queryset updates).
  - A run read slim (`_slim`) refuses to save its context.
  - Migration 0023 only adds the columns (`summary`, `rerun_votes`). `manage.py fill_run_summaries` fills them in the background after start-up and logs result and summary sizes. `use_slim` rebuilds any missing summary on read.
- **Reading slim** (`slim_runs` / `use_slim`):
  - batch page, review metrics, share page, Update (including "reviewed");
  - in-flight reconcile, eventml selection, period selection (only the chosen ≤200 are read whole, via `period._whole`).
  - Share-link files (video, still, overlay) load only pk, steps and gpu_job_id.
- **Read one at a time:**
  - re-track check: lazy, stops after 2;
  - grouping preview: `iterator(chunk_size=10)`;
  - layout save: `defer("context")`, recounting each whole;
  - trips CSV: each run once, not joined per review.
- **Downloads:** `sharing.batch_csv` orders clips from the slim copies, reads each clip whole 10 at a time (`aggregate.whole_runs`), and `_csv_response` is a `StreamingHttpResponse`. Tests read downloads with `.getvalue()`.
- **Fewer repeated fetches:**
  - `_run_for_job` remembers the run on the job for the request, and `pin_run` skips a second fetch;
  - `on_job_finished` applies a job only to a step that is RUNNING and waiting on that job id. That stops the 4-second re-saves, and an old job of a clip re-run in place can never overwrite it.
- **gunicorn:** `--max-requests 300 --max-requests-jitter 50`.
- **Not needed:**
  - the 300-clip cap was already gone;
  - thumbnails stream to a temp file on disk, not into memory.
- **Guardrails (`tests/test_run_summary.py`):**
  - batch page and metrics render the same slim as whole, including with insect groups;
  - the batch page never selects `context`;
  - slim runs can't be saved;
  - a memory budget: 150 clips with about 150 KB of results each must stay under 40 MB on the batch page, metrics, share page and interactions CSV. The batch page peaks at 6 MB slim and 119 MB whole.

## Update with GPU changes, in place (user, 2026-11-03)
- The user asked for this: "Inside this batch like any other pipeline update would."
- `regroup.split_by_change`: chosen clips with GPU-side changes go to `regroup.rerun_on_gpu`, the rest recount.
- **How a re-run works:**
  - same run and batch; the pipeline's current steps are bound to the clip's own part (`_window_of`);
  - `engine.start_run`, after a credits check over all of them;
  - `rerun_votes=True`; on completion, `engine._move_rerun_votes` calls `votes.move_votes(same_tracks=False)`.
- **Update card:**
  - GPU changes enable Update, with a note giving GPU-hours a clip;
  - new choice "Clips the filter shows (N)", posted as picks;
  - a confirm on every GPU path giving the clip count and estimated GPU-hours (the batch's GPU time ÷ finished clips).
- **Tests changed:** two insect-group tests that changed the species model only incidentally now make a web-only change.
