# 31 — GPU utilization, batched inference, and honest compute cost

## Context / problem

Two complaints, one root: nothing measures where a run's time goes.

1. **Compute Cost is fiction.** A 2026-09-08 job displayed `$0.0935`. That is
   `execution_seconds × GPU_TIERS[job.gpu_tier]["cost_per_sec"]`
   (`apps/analysis/views.py:1337-1340`), i.e. **305.6 s × the A10G rate** — on a
   run that happened on a T4.
2. **The GPU is underutilized** and nothing can prove or quantify it, because
   `execution_seconds` is one number covering download, decode, inference,
   post-processing and upload.

A codebase audit while planning this turned up that the cost bug is not a wrong
constant but a **phantom control**, that the fix has to land in three places,
and that the file we need to restructure carries the same dead-copy hazard that
caused the September `AnalysisResults` outage. Those findings reorder the work —
see "Audit findings" and the phased plan.

### Measured evidence (CloudWatch, `/aws/sagemaker/Endpoints`, `beemonitor-sm-dev`)

The job above ran 19:21–19:24 EDT (the UI renders UTC — `TIME_ZONE = "UTC"`,
`config/settings/base.py:118`):

| Time (EDT) | GPUUtilization | GPUMemoryUtilization | CPUUtilization |
|---|---|---|---|
| 19:07–19:19 | *(no data — scaled to zero)* | | |
| 19:21–19:24 | 100% | 27.98% | ~102% |
| 19:25+ | 0% | | → 0 |

(19:25 onward is contaminated by a concurrent Pulumi rollover — two instances,
which is why the GPUMem *average* halves while the max holds.)

**`GPUUtilization = 100%` is not saturation.** NVIDIA defines it as the fraction
of the sampling window in which *at least one kernel was resident*. A batch-1
YOLO call every ~30 ms pins it to 100% while using a small slice of the SMs. The
two numbers that carry signal:

- **GPU memory 27.98%** — ~4.5 GB of the T4's 16 GB. Batch of one, one model.
- **CPU ~102%** on a 4-vCPU `ml.g4dn.xlarge`, where SageMaker's scale is 0–400%:
  **one core pinned, three idle.**

The code explains it: `BeeTracking.process_video` (`tracking/bee_tracking.py:1363`)
is a strictly serial `cap.read()` (CPU decode) → `process_frame()` (GPU) →
`out.write()` (CPU encode, since `visualize=True` is the pipeline default) loop
on one thread. Decode and inference alternate; neither overlaps the other.

## Execution path (what runs, in order)

```
predict_fn                              sagemaker_backend/inference.py:88
  started = time.time()                 <- the only clock today
  CloudPipeline.process()               cloud/wrapper/pipeline.py:81
    Step 1  S3 download                 :127
    Step 2  ensure_models               :132   (cached after first run)
    Step 3  BeeMonitor.analyze_video     :168
              get_motion_tracking        core/video_analyzer.py:125
                BeeTracking.process_video  tracking/bee_tracking.py:1363
                  cap.read() / process_frame() / out.write()
    Step 4  foraging trips + interactions :194  (CPU/pandas)
    Step 5  S3 upload                    :274
  execution_seconds = time.time() - started   :141
```

## Audit findings

### F1 — `gpu_tier` is a phantom control for analysis jobs

Selected by the user (`analysis/templates/.../config_panel.html:29` →
`views.py:1433`), stored on the Job (`:1539`), and then used for **exactly two
things: the cost calculation and a success message**. It is never sent to
SageMaker and never selects hardware. Every analysis job runs on whatever the
endpoint is pinned to — `ml.g4dn.xlarge` (T4).

Consequences:
- A user picking "A100 (Fastest) — ~3 min/video" gets a T4 and is billed at
  `$0.000583/s`, **2.85× the true T4 rate**.
- Pipeline-launched jobs never set it → default `A10G` (`models.py:61`);
  `api/views.py:268` hardcodes `A10G`.
- The phantom propagates into billing reports: `accounts/views.py:72` groups
  usage by `gpu_tier`.

**Training does it properly** — `training/views.py:254` maps `_INSTANCE_BY_TIER`
(`:26-32`) to a real SageMaker instance type. The correct tier→instance mapping
already exists in this repo, in the wrong module, unused by the cost path.

### F2 — the cost math is duplicated

`analysis/views.py:1337-1340` and `api/views.py:627-629` are verbatim copies.
Fix one and the API keeps reporting the old number. A third consumer prices the
same seconds differently: `UserProfile.charge(credits_used, gpu_seconds=...)`
with `credits = int(exec_secs)`.

### F3 — the shadow-class bug is systemic

The September `AnalysisResults` outage (a stale in-module copy shadowing the real
class) has siblings:

| Name | Copies | Where |
|---|---|---|
| `Track` | 3 | `tracking/mot/base_mot.py:27`, `tracking/mot/bee_tracker.py:89`, `output/video_synthesizer.py:1288` |
| `Detection` | 2 | `detection/base_detector.py:17`, `tracking/mot/base_mot.py:16` |
| `SimpleBeeClassifier` | 5 | `detection/noise_filter.py:20` + 4 in `archives/` |

And the habit that hides them: **544 of the first 710 lines of
`bee_tracking.py` are a commented-out older version of the class below them**
(real code starts at line 687). `output/video_synthesizer.py` carries 169 more.
`bee_tracking.py` is the file Phases 4 and 5 restructure.

### F4 — the detector→tracker seam is untyped

Detections cross as bare positional lists, `list(bbox) + [conf, 'yolo', taxon]`,
built at `bee_tracking.py:1254`, `:1277`, `:1289`, documented only in a comment
at `:1284`, and re-parsed by position at `:1597`. Two `Detection` dataclasses
exist (F3) and neither is used for this hop. Batching would mint more copies of
the same untyped tuple unless the seam is fixed first.

### F5 — `analysis/views.py` is not a views file

2400 lines, 49 top-level definitions; roughly half is SageMaker orchestration
(`spawn_gpu_job_async`, `_drain_queue`, `_invoke_endpoint_async`,
`_chunk_ranges`, `_poll_sagemaker_results`, `_merge_chunk_results`,
`_apply_result_to_job`). The cost calculation sits at `:1294`, inside the poller,
inside this file.

### F6 — CI runs neither test suite

`.github/workflows/deploy.yml` runs `manage.py check` + `pytest cloud/tests/`
only. It does **not** run the ~133 Django app tests or `src/beemonitor/tests/`.
That is why the `AnalysisResults` break shipped and stayed green for four
commits. Compounding: `src/beemonitor/tests/test_tracking.py` fails to collect
(`cannot import name 'DetectionMode'`), which takes the whole suite down with
it; 4 `test_detectors.py` blob/SIFT tests fail; layout is split between
`beemonitor_web/tests.py` (1240 lines, project root) and 24 per-app files.

## Key facts (constraints any change must respect)

- **Two stateful, strictly-ordered components.** `BeeTracker.update(detections,
  frame_num)` accumulates track state, and `detect_motion` (`:1023`) drives a
  **MOG2 background subtractor** (`detection/blob_detector.py:119`) whose
  `apply()` mutates the background model per frame. Both must see every frame
  exactly once, in order. Frames may not be reordered or skipped; batching may
  only change *how detections are computed*, never the order they are fed in.
- **Mode transitions are decided by CPU motion, not by YOLO.** `process_frame`
  (`:1134`) runs a `motion_detection` <-> `tracking` machine; both transitions
  (`:1180` in, `:1244` out) test `detect_motion(roi_frame)`, a blob call. **No
  branch reads YOLO output.** So which frames need GPU work is knowable from
  cheap CPU state alone — this is what makes batching tractable.
- **Three YOLO call sites**, with different batching character:

  | Site | When | Batchable |
  |---|---|---|
  | lookback replay `:1194` | motion just started; replays the buffer | **Yes, trivially** — bounded loop, independent detections, tracker updated in order after |
  | transition frame `:1218` | the frame that triggered tracking | No — single frame by construction |
  | tracking mode `:1229` | every frame while tracking | **Yes, and it is the bulk of the work** — the hard one |

- **Precedent exists in-tree.** `_classify_species` (`:1039`) already batches the
  species classifier: *"at batch 1 the GPU is latency-bound and spends most of
  its time on launch overhead."* Same argument, applied to the detector.
- **`device` is already reported** from the real run: `_detect_device()`
  (`inference.py:409`) returns e.g. `cuda:Tesla T4`. That is the honest key for
  pricing — no extra AWS call, no guessed tier.
- **`detect()` flattens all results** across images, so passing it a list would
  silently merge frames' detections. Per-frame separation is the whole reason
  `detect_batch` is a new method rather than a widened signature.

## Plan (phases in dependency order)

### Phase 0 — CI runs the tests  *(new; precondition for everything)*

Add the Django app suite and `src/beemonitor/tests/` to `deploy.yml`. Fix or
delete `test_tracking.py`'s collection error first — one broken import currently
takes the whole `src/beemonitor` suite down, so adding it to CI as-is just makes
CI red.

**Why first:** every phase below is guarded by tests that CI does not currently
run. The equivalence test protecting the Phase 5 loop restructure is worthless
otherwise, and F6 is the direct cause of the last outage.

### Phase 1 — Delete the dead prologue in `bee_tracking.py`  *(new; precondition for 4 and 5)*

Its own commit, zero behavior change: drop the 544-line commented-out class
(and the 169 lines in `video_synthesizer.py`). Restructuring `process_video`
inside a file where 44% of the top is a stale copy of the same class is how F3
reproduces, and it makes the functional diff reviewable.

### Phase 2 — Instrumentation

`core/profiling.py`: a thread-safe `StageProfiler` accumulating
`{stage: {seconds, calls}}`, plus a module-level `PROFILER` reset per run.

- `decode`, `inference`, `encode` recorded in the frame loop.
- `download`, `upload`, `postprocess` recorded in `CloudPipeline.process`.
- `inference` recorded **inside `YOLODetector`** — at the one place GPU work
  happens, rather than inferred from outside.
- `predict_fn` returns `gpu_seconds` + `stage_seconds` next to the existing
  `execution_seconds`.

Store `stage_seconds` as a **JSON field** on `Job`, not one column per stage, so
adding a stage later is not a migration.

`inference` is wall time around *synchronous* detector calls, not pure kernel
time — Ultralytics copies results to the host before returning, so the call
already blocks on the GPU. Honest for "how long was the GPU step"; not for "how
long were kernels resident".

**Connects:** purely additive; context managers around existing calls. No
control-flow change.

### Phase 3 — Honest cost  *(rewritten around F1/F2)*

Ordered, because the constant is the last part:

1. **Decide the phantom tier (F1).** Either make it real — route to
   differently-sized endpoints the way training does — or delete the selector and
   price from the endpoint that actually ran. **Recommendation: delete it.**
   Making it real means more endpoints and more cold starts, which is explicitly
   out of scope; and the control has never done anything, so removing it costs
   users nothing but a misleading choice.
2. **Extract `apps/analysis/pricing.py`** — GPU name → instance type → real
   SageMaker per-second rate, plus the credits rule — and route **all three**
   call sites through it (`analysis/views.py`, `api/views.py`,
   `UserProfile.charge`). Seed it from `training/views.py:_INSTANCE_BY_TIER`,
   which already holds the correct mapping.
3. **Then** the formula change is one line in one module:
   `compute_cost_usd = execution_seconds × rate(device)`. Surface the stage
   breakdown on the job detail page.

**Open decision (a):** per-job cost can only honestly bill the **handler
window**. Cold start and the 180 s scale-in cooldown belong to no single job, and
scale-to-zero is out of scope. Proposal: bill handler time at the true rate,
labelled "compute (excl. cold start)". The example job becomes **$0.063**.

### Phase 4 — Overlap decode with inference

Reader thread decoding into a bounded `queue.Queue`; the main loop pops instead
of calling `cap.read()`. Order preserved exactly; one frame at a time into
`process_frame`. Depth 24 ≈ 150 MB at 1080p (a 1080p BGR frame is ~6 MB).
`BEEMONITOR_FRAME_QUEUE=0` restores the inline path — that flag exists so the
equivalence test can run both and compare.

**Connects:** `process_video` only. `process_frame`, `BeeTracker` and MOG2 are
untouched and never learn a thread exists.

### Phase 5 — Batch the detector  *(amended for F4)*

1. **Type the seam first.** Give the detector→tracker hop one documented
   converter (or a small dataclass) instead of four hand-built positional lists.
   Batching otherwise adds a fifth and sixth copy of an undocumented tuple.
2. **`YOLODetector.detect_batch(frames) -> List[List[Detection]]`**, with
   `detect()` becoming `detect_batch([f])[0]` so every existing caller is
   unchanged.
3. **5a — lookback replay.** One batched call instead of N. Provably identical
   output, low risk. Ship first.
4. **5b — tracking mode.** The real win. Because mode transitions are
   CPU-decided (Key facts), run the cheap motion pass ahead over queued frames to
   learn exactly which of the next K frames will be in tracking mode, batch YOLO
   for precisely those, then drive the state machine in order with detections
   precomputed. Nothing speculative, no wasted inference — but it restructures
   `process_video`'s inner loop, i.e. the scientific core.

**Open decision (b):** 5b in or out. Phases 0–4 + 5a still deliver CI, the dead
-code cleanup, instrumentation, honest cost and the decode overlap; 5b can follow
once the profiler quantifies what tracking-mode inference actually costs.

## Explicitly out of scope

**Cold start / scale-to-zero.** ~10 min billed for ~4 min of work, dominated by
pulling the 9.7 GB image, is the single largest dollar item — and is deliberately
left alone per direction. Do not "fix" it as a side effect. (Related: the *video*
image bakes SAM 3 weights it rarely uses, `sagemaker_backend/Dockerfile.gpu`.)

**F5 (`analysis/views.py` at 2400 lines)** is noted, not scheduled. Phase 3's
`pricing.py` extraction is the one slice of it this work needs; a full split of
the orchestration layer out of `views.py` is its own project.

Parallelism **across** videos already exists via `max-capacity: 4` and
`SAGEMAKER_MAX_CONCURRENT` ([[30_batch_gpu_queue_design]]); it is the throughput
lever for a backlog and is orthogonal to per-video efficiency.

## Verification

The equivalence test is the safety story: run a synthetic clip through the serial
path and the new path, assert the resulting tracking DataFrames are **identical**
— same frames, track ids, boxes. Plus: `detect_batch` per-frame separation;
reader-thread shutdown on a mid-loop exception (every exit path posts exactly one
sentinel); GPU-name → rate mapping; and a test that the three cost call sites
agree, so F2 cannot silently return.

All of it only counts once Phase 0 lands — see F6.
