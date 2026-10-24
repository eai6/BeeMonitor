# 58 — GPU time for sampling and pre-labeling: measured right, charged right, shown

**Status:** built (2026-10-24). Both open questions went as planned: batch handler time, failed batches recorded and free. Also: editor suggest-only pre-labels are now charged (they were free); the "Auto-label" kind is shown as "Pre-labeling"; training charges carry their project too.
Related: [[31_gpu_utilization_and_cost_accuracy]], [[50_tiers_quotas_and_platform_admin]], [[38_sampling_on_sagemaker]], [[57_test_set_and_yolo_prelabel]].

## Why
The user wants to see total GPU-hours for sampling and pre-labeling, and that time must be clearly recorded and charged as credits.

## Audit (2026-10-24)
- Both are charged through `quota.charge` → `UsageEvent` (idempotent on source). Unlimited accounts are recorded, not deducted.
- **Sampling over-counts.** `_apply` returns each clip's `scan_and_detect` wall time and `collect` sums them. The GPU runs 3 clips at once sharing one GPU lock, so the sum is up to ~3× the time the instance was busy. Download, upload and model load are left out. The batch's own `execution_seconds` (the whole handler) is thrown away.
- **YOLO pre-labeling is under-billed.** Since beb0e59 it runs on the SAM 3 endpoint (g5, 2 credits/h), but `auto_label_work` maps non-SAM 3 to g4dn (1 credit/h). `cost_usd` and the estimate history are wrong too.
- **Units are wrong.** Sampling counts failed and skipped clips (`sampling_remote.py:399`); auto-label uses the requested `max_frames`, not frames checked. Both skew the per-unit estimates.
- **Nothing kept per task.** No GPU seconds on `FrameSamplingTask`, `PreAnnotationTask` or `SamplingBatch`.
- **No project link.** `UsageEvent` has no project, so there is no per-project total anywhere. Users see one GPU-hours total; only platform admin has a per-kind split.
- Whole-batch failures: GPU ran, nothing recorded, nothing charged. Nothing reconciles charged seconds with SageMaker instance-hours.

## Plan
1. **Bill sampling by the batch's handler time** (`execution_seconds`: download → decode → detect → upload for the whole batch). One invocation holds one instance, so that is what the batch used. Split it across the batch's clips in proportion to each clip's `scan_and_detect`; failed and skipped clips get no share, and their share is not billed.
2. **Charge pre-labeling at the instance it ran on:** the endpoint the task was sent to (SAM 3 endpoint → g5), stored on the task at spawn. Units = frames checked.
3. **Keep the numbers:**
   - `SamplingBatch.gpu_seconds` (handler time) and `clips_completed`;
   - `FrameSamplingTask.gpu_seconds` (its share) plus the GPU's timing split (`download / scan / upload`, frames decoded) in a `timing` JSON;
   - `PreAnnotationTask.gpu_seconds`, `instance_type`.
4. **`UsageEvent.project`** (nullable FK, `SET_NULL`). Sampling charges one event per (batch, user, project); pre-label per task. Old events stay without a project.
5. **Show it, on existing pages only:**
   - Annotation project page, beside the sampling status: "GPU used: 3.4 GPU-h · 6.8 credits (sampling 3.1 h · pre-labeling 0.3 h)". Managers only.
   - Account usage page: this month's GPU-hours and credits split by kind (analysis, sampling, pre-labeling, training).
6. **Fix units** so estimates are per completed clip / per frame checked. Estimate history keyed by instance (YOLO and SAM 3 sampling both g5; same key, accepted).
7. **Failures:** whole-batch failures are recorded on the batch (`gpu_seconds`, error) but not charged.
8. **Not now:** reconciling with CloudWatch instance-hours, cold-start/idle time (memory/31 decision a: bill the handler window), back-filling old events.

## Tests
- Batch time split across completed clips; failed clip gets nothing; idempotent collect; units = completed clips.
- YOLO pre-label charged at g5; units = frames checked.
- Project total on the page sums only that project's events; other users' projects excluded.
- Usage page per-kind split.

## Open questions
1. Bill sampling by batch handler time (planned; charges for a batch drop to roughly a third of today's sum, plus download/upload now included)?
2. Whole-batch failures recorded but free (planned)?
