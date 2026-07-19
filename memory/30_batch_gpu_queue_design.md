# 30 — Queue-gate GPU launches + SageMaker retry/backoff

## Context / problem

Launching a large pipeline batch (~4,000 videos via "run on all filtered") mass-fails jobs. The pipeline path (`apps/pipelines/views.py:run_on_videos` → `engine.start_run` → `executors.submit_gpu_step`) created every job as `PROCESSING` and immediately invoked the SageMaker async endpoint, with **no concurrency gate** (unlike analysis `BatchJobView`, which enforces per-user `max_concurrent_jobs`). The endpoint runs only ~6 concurrently (`max_concurrent_invocations_per_instance=3` × autoscale `max_capacity=2`), and the boto3 client was **fail-fast, no retry** (`analysis/views.py:211`), so every throttled/queue-full invoke was marked `Failed` instantly. Jobs that grabbed a slot Completed; the flood Failed. Raising `PIPELINE_MAX_BATCH` to 50k enabled bigger floods.

## Fix (implemented)

1. **Queue-gate spawns** — GPU jobs are created `QUEUED`; the background reconciler promotes them to `PROCESSING` (spawns) only as global capacity frees up.
2. **Retry/backoff** — SageMaker + hot-path S3 boto3 clients use adaptive retries so transient throttles wait instead of instant-failing.

Deferred: scaling SageMaker `max_capacity` (raise `SAGEMAKER_MAX_CONCURRENT` to match when done); a strict cross-instance limiter; re-running already-failed videos (safe to re-run in smaller batches now).

## Key facts
- `Job.Status.QUEUED = "queued"` already exists (`apps/analysis/models.py:26`) and is the model default (`:56`) — create sites had been overriding it to `PROCESSING`. **No migration.**
- Reconciler: `apps/analysis/reconcile.py:reconcile_all` runs every `BEEMONITOR_RECONCILE_INTERVAL` (default 120s) on a daemon thread from `AnalysisConfig.ready()`; idempotent; runs in every worker/instance.
- In-flight count mirrors `BatchJobView` (`analysis/views.py:1439`): `Job.objects.filter(status=PROCESSING).count()`.
- Atomic claim (lock-free CAS): `Job.objects.filter(pk=pk, status=QUEUED).update(status=PROCESSING, started_at=now)` → returns 1 for the winning worker, 0 for losers; prevents double-spawn across instances.
- Safety net: `_handle_unspawned_job` (`analysis/views.py:893`) re-spawns a `PROCESSING` job with empty `modal_call_id` after `_SPAWN_GRACE_MINUTES` — covers a worker dying between claim and spawn.

## Changes
1. **Setting** `SAGEMAKER_MAX_CONCURRENT` (default 6, env-overridable) — tracks endpoint capacity; bump when infra scaled.
2. **Create-as-QUEUED (no immediate spawn)** at the two flood-prone sites that already use the single-job `spawn_gpu_job_async`: `pipelines/executors.py:submit_gpu_step` (the 4,000-video flood path) and `analysis/views.py` one-click. A QUEUED job's pipeline step stays `STEP_RUNNING` ("parked"), and `engine.on_job_finished` ignores non-terminal jobs — no pipeline-UI change.
   - **`BatchJobView` was left on its existing path** (deliberate deviation from the 3-site plan): it is already gated by per-user `max_concurrent_jobs` (429 at capacity), and it spawns via `_spawn_gpu_batch(...)` which builds the payload from separate args rather than `job.config` — routing its jobs through the single-job drain risked dropping `detection_mode`/model-path params. Its smaller bursts (≤ max_concurrent_jobs) are covered by the new adaptive retry. Revisit if batch bursts still throttle.
3. **`_drain_queue()`** in `reconcile.py`, called from `reconcile_all`: `slots = max(0, CAP - active_processing)`; promote that many `QUEUED` (FIFO by `created_at`) via the atomic claim, then `spawn_gpu_job_async`.
4. **Opportunistic inline drain** — best-effort `_drain_queue()` (try/except) after creating jobs in the views, so capacity-free launches start instantly and only overflow queues.
5. **Retry/backoff** — `_sagemaker_runtime` retries → `{"mode":"adaptive","max_attempts":5}`; same adaptive `Config` on the hot-path S3 clients (`_put_inference_payload`, `_poll_sagemaker_results`) via a shared `_boto_config()` helper.

Multi-instance caveat: each instance drains independently, so brief over-admission above CAP is possible; the atomic claim prevents double-spawning one job, and adaptive retry + SageMaker's async queue absorb modest overshoot. Keep CAP at/just under true capacity.

## Verification
- `../venv/bin/python manage.py check`.
- Unit: N `QUEUED` jobs + monkeypatched `spawn_gpu_job_async` → `_drain_queue()` with small cap promotes exactly `slots`; a second pass with no free slots promotes none; two passes over one job promote it once (atomic claim).
- Functional: launch ~30-video batch → jobs sit `QUEUED`, move to `PROCESSING` in waves ≤ CAP, none `Failed` from throttling; a single one-click job still starts near-instantly via the inline drain.
