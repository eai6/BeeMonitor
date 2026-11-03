# 75 — Review speed; web memory

**Status:** step 1 built (2026-11-03). Step 2 (batch summaries) still to plan.

## Seen (user, 2026-11-03)
- The review player was slow and unresponsive.
- The App Runner logs (12 h) showed 6 gunicorn workers SIGKILLed for memory, each right after a load of a 408-clip batch page (a1789cee, 1.3 MB of HTML).
- Slow requests: batch page avg 7 s, review metrics avg 6 s, poll up to 33 s, thumbs avg 7 s, results page up to 10 s.

## Cause
- A run's results (events, interactions and more) live in one JSON field, `PipelineRun.context`.
- Several reads loaded that field for many runs just to learn each run's GPU job id:
  - `review.shared_job_runs` read every reviewed run in the database, 2–3 times per vote and 8 times per batch page;
  - `review.member_job_ids` read every run of the reviewer's batches, on each `Job.for_viewer` / `clip_access` call;
  - `executors._run_for_job` read up to 400 runs;
  - `batches.neighbours` (Prev/Next) read the whole batch.

## Built
- **`PipelineRun.gpu_job_id`:** an indexed column. `save()` keeps it in step with the context whenever the context is saved. Migration 0022 fills it, and 0018 now defers it too.
- **Queryset updates of `context`** (recount, retrack) bypass `save()`. They never change the job, so the column stays right. Any new queryset update that changes the job id must also set `gpu_job_id`.
- **The scans above** now query the column. recount `runs_for_clips` and period `clip_runs` are narrowed by it too.
- **Simulated 900-clip batch, before → after:**
  - vote: 1.7 s → 5 ms;
  - results page: owner 5 s → 0.2 s, invited reviewer 16 s → 0.1 s;
  - player event data (reviewer): 11 s → 0.03 s.
- **"Re-run these N"** scrolls to the re-run form, rings it and focuses its pipeline menu. The bare #rerun jump went unnoticed.

## Still to do: step 2
- The batch page (`_batch_runs`) and every review-metrics POST still load every run's whole context. In the simulation, 400 runs took about 440 MB per load.
- **Plan:** a small per-run summary saved when results are written (counts, event keys, actions, ml_confidence, filtered_out keys), which the batch page and metrics read instead of `context`.
- **Instance:** the user chose to keep 1 vCPU / 2 GB for now.
