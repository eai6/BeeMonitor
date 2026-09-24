# 38 — Frame sampling + pre-annotation in one GPU pass (SageMaker)

Status: **PLAN v3 — sampling and pre-annotation combined (§4.1); calibrated on real clips (§4.5); SAM 3 speed being measured (§6)** (2026-09-24). v2: v1 was
audited against the code (SageMaker side, web side) and live AWS (endpoints,
CloudWatch, S3, real clips); v2 folds in every finding. Decisions are in §9.
Owner of every `pulumi up`: Edward.

## 1. Why

Frame sampling runs inside the web server (`beemonitor_web/apps/annotations/
sampling.py`, a 2-thread pool in the App Runner container). Measured from App
Runner logs, 2026-09-22 → 24:

| Problem | Evidence |
|---|---|
| Too slow | ~50 tasks/hour (42–70); 606 tasks in two days. Container is **1 vCPU / 2 GB**, decoding every 1080p frame in Python. |
| Killed by restarts | 25 × `cannot schedule new futures after interpreter shutdown` in 7 days (deploys, worker restarts). Tasks killed mid-decode stay `processing` forever: `poll_frame_sampling_tasks` re-feeds only `queued` (its docstring claims otherwise). |
| Motion picks too few frames | Frames/task over 606 tasks: 0 → 82, 1 → 207, 2 → 141, 3–9 → 160, ≥10 → 16 (target 20). At 320 px a bee is a few pixels and the 3×3 open erases it. |
| Hurts the website | Decode shares the one vCPU with page requests. |

## 2. Goals

- Sampling never runs in the web server.
- 9,000 clips in **~2 h** on 4 instances (~3–4 h on 2), plus a 9–16 min cold start (§6).
- Deploys never lose or strand sampling work; nothing is paid for twice.
- Motion sampling returns its requested frames on clips with bees.
- Frames are stored exactly as today: key `frames/{blob_path.replace('/', '_')}/f{n:06d}.jpg` in `processed`, one `Annotation` row per frame, `n` = OpenCV sequential decode index. Editor, auto-label, export untouched.

## 3. Facts the design rests on (verified)

**Endpoint.** `beemonitor-sm-dev-sam3`: `ml.g5.xlarge` (4 vCPU = 2 physical cores, 16 GB, A10G), autoscale 0–4, **1 invocation per instance**, FIFO async queue. It runs the **unified GPU image** (`Dockerfile.gpu` → ECR `beemonitor-sm-dev`, handler `sagemaker_backend/inference.py`, served by `sagemaker_backend/serve` — gunicorn 1 worker × 4 threads, timeout 14,400 s). `sagemaker_backend/sam3/*`, `Dockerfile.sam3` and the `sam3-image-tag` config are **unused** by the endpoint (`infra/aws-sagemaker/__main__.py:679-688`).
- `ffmpeg` and `opencv-python-headless` are already in `Dockerfile.gpu`.
- The role reads `raw-videos` and writes `processed`. The Django policy may invoke `-sam3` and put/get the async buckets. **No IAM change.**
- It also runs **SAM 3 tracking** (chunked long-video jobs) and **auto-label**: sampling is a third tenant.
- CPU history: 360%/400% CPU on 2026-09-09 made `/ping` fail ("could not get a response"); `BEEMONITOR_CPU_THREADS=2` exists for that. Sep 10 auto-label: CPU 107–124% avg, GPU 71–76%, memory 12–16%.

**Image drift.** The main endpoint runs `beemonitor-sm-dev:3a52a4b` (= `image-tag` in `Pulumi.dev.yaml`). The **SAM 3 endpoint runs `155d53c`** (Sep 9), one day older — it lacks `6a16579` ("bound the CPU pools") and `3a52a4b`. Any `pulumi up` also brings SAM 3 up to date.

**Autoscaling.** Target tracking, 5 queued requests per instance (`ApproximateBacklogSizePerInstance`). Scale out after 3 min above target; scale in after **15 min** below 4.5 (180 s cooldown). From zero: +1 instance per alarm, 300 s cooldown. So **K requests in flight → about ceil(K/5) instances**: K=8 → 2 instances, K≥20 → 4.

**Cold start and idle.** 0 → 1 instance measured at **520 s and 947 s** (9–16 min): instance provisioning plus a 9.7 GB image pull, not model load. After the last request an instance stays up **~21 min**, about $0.50 per scale-up at $1.41/h.

**Queue behaviour.** Sep 10 auto-label run (75 requests): time in backlog avg 745 s, max **2,207 s**; model time avg 316 s, max 1,085 s. Callers already pass `InvocationTimeoutSeconds=3600`.

**Clips.** 87,059 clips, 878 GB, median 4.3 MB. All **H.264 High 1920×1080, 25 fps (slightly variable), keyframe every 25 frames**. Decoding is the whole cost: ~5 ms/frame single-threaded (M3 Pro); scale/fps/gray filters don't cut it (H.264 decodes every frame); `-skip_loop_filter all -flags2 fast` saves ~20%. Estimate per g5.xlarge with 4 workers: **~10–13 s per average clip → ~1,100–1,400 clips/hour**. A 450 MB (16k-frame) clip is ~200 s, far under 3,600 s.

**Storage classes — blocking.** Bucket rule `raw-videos-tiering`: STANDARD_IA at 30 days, **GLACIER (Flexible Retrieval) at 90 days**. **34,027 of 87,059 clips (39%, every June clip) are in GLACIER now** and cannot be read without a restore. July moves in October. This breaks sampling on any backend, and also playback, analysis and auto-label for those clips (§9, decision 1).

**Web app.** The reconciler thread runs in **every gunicorn worker** (2 per instance), App Runner scales 1–25 instances, and deploys overlap old and new instances. No `select_for_update` anywhere in `reconcile.py` or `apps/annotations`. Production DB: Postgres (`db.t4g.micro`, private); tests: SQLite.

**Existing bugs found (fix alongside):**
- Pre-annotation drain has no atomic claim: two processes can invoke the same task (paid twice); finalize can double-charge credits and hit `IntegrityError` (`annotations/views.py:1745-1770, 1570-1711`). `PreAnnotateView` bypasses the cap.
- `_frames_per_clip()` uses the interval default (100), but adds use motion (20): the add-page confirm shows **5×** the real frames (`annotations/views.py:895`).
- `CancelSamplingView` exists but nothing links to it.
- 12 × `column devices_device.motion_var_threshold does not exist` during a deploy: old code ran against the new schema. Column removals need two deploys (§7).

## 4. Design

### 4.1 GPU: `task: "sample_label"` in `sagemaker_backend/inference.py`
**One pass per clip does sampling and pre-annotation** (decision 4): decode
once, find candidate frames by motion, confirm and box insects with SAM 3,
keep the top N. Frames reach the editor already pre-labelled. It extends the
existing `_pre_annotate` (which already keeps only frames with detections)
with motion candidates, clip batches and a single decode.

- `predict_fn` routes `task == "sample_label"` to `_sample_label_batch`.
  `input_fn`'s required keys (`job_id`, `user_id`, `video_blob_path`) are
  exempted for this task.
- Payload: `{task, batch_id, result_key, classes, confidence, clips: [{task_id,
  video_blob_path, frame_prefix, max_frames, min_gap_s, roi, polygon}]}`.
  `classes` = the labels the user ticked (as in the Auto-label panel, decision
  5): they are the SAM 3 prompts **and** decide which frames are picked.
- **Per clip, CPU side** (a `ThreadPoolExecutor(3)`; each thread drives an
  `ffmpeg -threads 1` subprocess; no `multiprocessing` — CUDA is initialised
  after the first request, fork is unsafe; ≥1 vCPU stays free for `/ping`):
  1. Download from `raw-videos` (`pipeline._storage`).
  2. Decode once with **`cv2.VideoCapture`** in sequential order, one decode
     thread per clip — the same decoder and order the editor uses, so frame `n`
     is the same frame there by construction (chosen over an ffmpeg pipe, which
     would need index mapping). Measured: 13.2 ms/frame on one M3 thread
     including motion scoring (grey-then-resize saved ~25%).
  3. Motion score per frame (§4.5 settings): ROI crop at 640 px wide, clock
     masked, MOG2, bee-sized blob count; frames with >20% of the ROI moving are
     "handling" and score 0.
  4. Keep a bounded set of **candidates** in memory: the top ~C by motion
     (C ≈ 15, see §6), at least 0.5 s apart, **plus** ~C/4 evenly spaced
     frames so still insects (camera 7) are checked too.
- **GPU side** (one SAM 3 model, serialised through its existing lock):
  run the candidates through SAM 3 with the chosen prompts, **batched** where
  the processor allows. Rank frames by detections (count × confidence).
- **Only moving detections count** (measured 2026-09-24): SAM 3 prompted
  `bee` on a bee hotel boxes every mud-plugged nest hole as a bee (42–48 boxes
  per frame, conf 0.2–0.6, identical positions every frame). A detection counts
  as activity — and is kept as a pre-label — only if it overlaps a bee-sized
  motion blob in that frame (dilated ~1 blob width). Nest plugs never move and
  drop out; a sitting insect that shifts slightly (camera 7) stays.
- **Pick** the top N ranked frames, ≥ `min_gap_s` apart (default 0.5 s). A
  clip with no detections yields **0 frames** (correct for empty
  motion-triggered clips).
- **Write** the picked frames as JPEG to `processed` (the same key
  convention) and return `{task_id, frames: [{n, key, w, h, boxes:[…]}],
  motion: {profile, picked, frames}, candidates, seconds: {decode, gpu,
  upload}, error?}`. Boxes use the same shape `finalize_preannotation_task`
  writes today.
- Per-clip `try/except`: one bad clip never fails the batch.
- The whole result also goes to a **fixed key** `sampling-results/{batch_id}.json`.
- One log line per clip with timings.

### 4.2 Web: batches, claims, dispatch
- New model **`SamplingBatch`**: id, status (`claimed → invoked → collected | failed`), `attempts`, `claimed_at`, `invoked_at`, `output_uri`, `failure_uri`, `result_key`. `FrameSamplingTask` gains FK `batch`, `attempts`, and an index on `(status, created_at)`.
- **Setting `SAMPLING_BACKEND`** = `local` (default, dev/tests) | `sagemaker`. Set in `infra/aws/__main__.py` `env_vars` as well as live, or `pulumi up` reverts it (same as `BEEMONITOR_CHUNK_TRACKING`).
- One entry point `sampling.start(tasks)`: `local` hands the tasks to the pool as today; `sagemaker` leaves them `queued`.
- **Dispatcher** (reconciler tick, `sagemaker` only):
  1. Take a Postgres advisory lock for the in-flight check (count **batches**, not tasks).
  2. Claim up to B `queued` tasks with `select_for_update(skip_locked=True)` ordered by `created_at`. Create a batch in `claimed`.
  3. Commit, then invoke outside the transaction with `InferenceId=batch_id`, `InvocationTimeoutSeconds=1800` and `RequestTTLSeconds` ≈ 3 h.
  4. Store `invoked_at` and `output_uri`/`failure_uri`.
- **Batch size B by work, not clip count**: aim for ~5 min per batch, using clip file size or duration as the proxy. That is about 10–25 average clips; a very large clip gets a batch of its own.
- **Clips SageMaker can't read** are resolved **before** dispatch, never inside a reconciler tick:
  - External `s3://` keys: ingest in a pool, as auto-label does.
  - Archived (Glacier) clips: per decision 1.

### 4.3 Web: collect
- Poll `result_key` (and the failure path) for `invoked` batches.
- Claim the batch with a conditional UPDATE `invoked → collecting`, so only one process collects it.
- Per task, in `transaction.atomic()`, apply only if the task is still `processing` and belongs to **this** batch:
  - Skip cancelled or superseded tasks.
  - Run "replace unlabelled" (web-side, as today).
  - Upsert `Annotation` rows with `_record_frame`, idempotently, **with the
    SAM 3 boxes** merged exactly as `finalize_preannotation_task` merges them
    (the frame is pre-labelled, not `sampled_only`-empty). Never overwrite boxes
    on a frame someone has labelled or reviewed.
  - Store `motion`.
  - Mark the task completed, or failed with the clip's own error.
- Collecting twice changes nothing.
- The collector runs whatever `SAMPLING_BACKEND` is, so switching back to `local` still drains batches already in flight.

### 4.4 Recovery, retries, cancel
- **Claimed but never invoked** (crash between steps 3 and 4) for more than 10 min: tasks go back to `queued`, and the batch is marked failed.
- **A failure record exists** (it is authoritative: SageMaker writes one on TTL or invocation timeout): each task gets `attempts += 1`. Tasks re-queue as batches of one clip each, up to 2 attempts, then fail with the reason.
- **No result and no failure** after TTL + invocation timeout + margin (about 4 h): the same retry path, **after** checking both keys once more. No 45-minute timeout.
- **Clip-level errors** fail that clip only.
- **Cancel**: queued tasks are cancelled at once. In-flight tasks are marked cancelled; the collector skips them, and the batch holds its GPU slot until its output or failure appears. Add a **Cancel sampling** button to the project page.
- **Supersede**: a new task for the same (project, clip) cancels older queued ones. The collector skips a task if a newer non-cancelled task exists for that clip.
- **One-off recovery (command):**
  - Re-queue last night's `failed` tasks and stuck `processing` tasks.
  - Offer a re-sample of clips that got ≤2 frames.
  - Report counts, since the database isn't reachable from outside the app.

### 4.5 Calibration results (done 2026-09-24, 13 clips, 7 cameras)
Clips: hotels (cameras 2, 3), red box (7), yellow pan traps + an indoor test
clip (8), a hand over the lens (6), flower platform (12). All 1080p, 25 fps.

| Finding | Consequence |
|---|---|
| At 320 px a bee is 3–10 px; the 3×3 open erases it | Score at 640 px **inside the ROI crop**; area filter 12–2000 px (at 640), no open |
| The burned-in clock (top-left) changes every second | Always mask it (top 5 % × left 35 %) |
| A hand / laptop / lighting change fills the frame (camera 6 up to 80 %) | Frames with >20 % of the ROI moving score 0 ("handling"): dropped 130 of 433 frames on one clip |
| A still insect (camera 7) barely moves | Motion alone gives 0–1 frames; add evenly spaced candidates |
| Wind in grass, sun/shadow flicker (camera 3 at 06:24: 70–141 "blobs" on the hotel face; flower platform: every pick was grass) | **Motion alone cannot rank bees outdoors, even inside a rectangular ROI** → confirm with a detector (SAM 3, §4.1); a traced polygon ROI helps |
| Some motion-triggered clips contain no bee at all (camera 2, 32 s) | 0 frames is correct; the detector makes that safe |
| Many clips are ~6 s | A 1 s gap caps them at ~6 picks; default gap 0.5 s |

Baseline (today's code) vs tuned motion, frames picked of 20: camera 7 still
insect 0 → 11; hotel 2 → 9; flower 2 → 20 (but all grass without an ROI).
The detector stage is what makes the picks right; motion only proposes.

### 4.6 UI and text
- **Project page:** a sampling status line (queued · running · failed · finished with 0 active frames). Today a 0-frame clip looks the same as "not sampled".
- **Add-page confirm:** fix `_frames_per_clip` (use `motion_params`), and replace the time estimate with recent **throughput** (tasks completed per hour).
- **Fix text that becomes false:**
  - "no GPU is used" (`annotations/views.py:2204`);
  - "two clips at a time" (`add_videos.html:123`);
  - "on the server CPU" (`detail.html:730`);
  - `sample_workers: 2`;
  - docstrings in `sampling.py`, `models.py`, `views.py`.
- **Motion strip query:** load it only for the clips shown, latest task per clip. It currently loads every task in the project.

### 4.7 Archived footage (decision 1: A + B) — do first, it is urgent
The raw-videos rule sends clips to Glacier Flexible Retrieval at 90 days, where
they cannot be read without a restore. That saves ~$1.75/month on June's 197 GB
compared with Standard-IA, and makes the footage unusable. **July clips (33,190,
234 GB) start crossing 90 days on 2026-09-29.**
- **A. Lifecycle rule** (`infra/aws/__main__.py:128-142`, rule
  `raw-videos-tiering`): 90-day transition `GLACIER` → `GLACIER_IR` (Glacier
  Instant Retrieval: ~$0.004/GB-month, readable in milliseconds, ~$0.03/GB when
  read, 90-day minimum). STANDARD_IA → GLACIER_IR is a supported transition.
  `pulumi up` in `infra/aws` (Edward) **before 2026-09-29**. Changes storage
  class only; deletes nothing.
- **B. Bring June back** (34,027 clips, 196.7 GB, all uploaded 2026-06):
  a management command / script, idempotent and resumable:
  1. `restore-object` with the Bulk tier (5–12 h), `Days=7`, for every
     GLACIER object under `raw-videos` (skip ones already restoring).
  2. When `x-amz-restore: ongoing-request="false"`, `copy-object` onto itself
     with `StorageClass=GLACIER_IR` (all clips < 5 GB, so single-call copy).
  3. Report progress; re-run until nothing is left in GLACIER.
  The bucket is versioned: the in-place copy makes a new GLACIER_IR version and
  the old GLACIER version becomes noncurrent, expired after 30 days by the same
  rule (`noncurrent_version_expiration`). The clips are already past Glacier's
  90-day minimum, so there is no early-deletion charge.
  Cost: request fees ≈ a few dollars one-time. This is a live AWS change;
  Edward runs it (or approves Claude running it) after a dry run lists counts.
- **Until June is back:** the clip pickers and sampling mark clips whose
  storage class is GLACIER as "archived — restoring" and skip them rather than
  fail. `Video` gains `storage_class` (backfilled from S3 listing; refreshed by
  the restore script).

### 4.8 Fix while here
Give the pre-annotation drain and finalize the same claim helper: conditional UPDATE `queued → processing` before spawning; collect by compare-and-swap.

## 5. Rollout

| # | Step | Who | Live? |
|---|---|---|---|
| 0a | Lifecycle rule → GLACIER_IR in `infra/aws` (done, commit below); `pulumi preview` needs the stack passphrase, so Edward runs it | Claude + Edward | No |
| 0b | **`pulumi up` in `infra/aws` before 2026-09-29** | **Edward** | Yes (S3 rule) |
| 0c | Restore-and-copy June (dry run → run); `Video.storage_class` + "archived" handling in pickers/sampling | Claude + Edward | Yes |
| 1 | Calibrate motion scoring (§4.5); measure per-clip time with the ffmpeg pipeline | Claude | No |
| 2 | **Done** — `src/beemonitor/processing/sample_label.py`, `Sam3Detector.detect_many` (batched, falls back to single frames), `sample_label` task in `sagemaker_backend/inference.py`; 11 tests | Claude | No |
| 3 | Web: models + migration, `start()`, dispatcher, collector, recovery, cancel/supersede, backend gate, claims for pre-annotation, UI fixes; tests (§8) | Claude | No |
| 4 | Push → CI builds `Dockerfile.gpu` → `beemonitor-sm-dev:<sha>`; web deploys with `SAMPLING_BACKEND=local` (no behaviour change) | Claude | Web only |
| 5 | Set `beemonitor-sagemaker:image-tag` to the new sha; `pulumi preview`; hand over the diff and command. Expected: 2 Models + 2 EndpointConfigs created, **both** endpoints updated in place, no deletes. The main YOLO endpoint moves too; SAM 3 also catches up from `155d53c`. | Claude | No |
| 6 | `pulumi up` in `infra/aws-sagemaker`, **when neither endpoint has jobs in flight**. Each endpoint may start one instance briefly (`initial_instance_count=1`). | **Edward** | Yes |
| 7 | Set `SAMPLING_BACKEND=sagemaker` (live + `infra/aws` env_vars); smoke test 5 clips, then 500; read throughput and cost from CloudWatch | Claude + Edward | Yes |
| 8 | Recovery command (§4.4) | Claude | Yes |
| 9 | Remove the in-process pool after a few stable days | Claude | Yes |

Each step is its own commit. Steps 1–5 change nothing live.

## 6. Numbers

**SAM 3 measured on the live endpoint (2026-09-24, g5.xlarge, 1080p frames):**

| Test | GPU | Per frame |
|---|---|---|
| 20 frames, prompt `bee`, 3 clips | 11.6–11.9 s | 0.58 s |
| 16 frames, `bee`, warm | 9.3 s | 0.58 s |
| 20 frames, `bee, wasp, ant` | 34.7 s | 1.73 s — **linear in the number of classes** |
| Cold start to first result | ~8 min (486 s) | |

Detections: red box (camera 7) 20/20 frames, one box on the insect — correct.
Hotel (camera 2) every frame, 42–48 boxes, all nest plugs — false (hence the
motion-overlap rule in §4.1). Hotel (camera 3, conf 0.3) and pan traps
(camera 8): 0.

**Throughput of the combined pass** (C candidates × P classes × 0.58 s GPU,
decode in parallel on the CPU threads; single-frame SAM 3 as today —
batching is expected to help and is measured in step 7):

| C × P | GPU/clip | Per instance | 4 instances | 9,000 clips | ≈ Cost |
|---|---|---|---|---|---|
| 30 × 1 | 17 s | ~210/h | ~840/h | ~11 h | ~$60 |
| **15 × 1** | 9 s | ~400/h | ~1,600/h | ~5.5 h | ~$30 |
| 15 × 3 | 26 s | ~140/h | ~550/h | ~16 h | ~$90 |

Cost = instance-hours × $1.41 plus ~21 min idle per scale-up. Credits are
charged per clip from GPU seconds (decision 3), so more classes cost more;
the panel shows the estimate before starting.

## 7. Risks
- **Auto-label waits behind sampling** (FIFO, 1 per instance). With batch ≈ 5 min and K bounded, the wait is ≈ (K / instances) × 5 min. Keep K modest (decision 2), or move sampling to its own endpoint later.
- **Tracking jobs share the queue** and can run up to an hour each. K should account for them.
- **`pulumi up` rolls both endpoints**; in-flight async jobs at that moment may fail. Schedule it for an idle time.
- **Migrations that remove columns** break the old code still serving during a deploy. Add first; remove in a later deploy.
- **A GPU instance used for CPU work.** Fine at this volume. `ffmpeg -hwaccel cuda` (NVDEC on the A10G) could be several times faster but is unverified (it depends on the ffmpeg build and `NVIDIA_DRIVER_CAPABILITIES=video`). Try it in step 1 only as an option.

## 8. Tests
- **GPU:** active window picked; still clip → 0; ROI respected; picked `n` matches the cv2 sequential index on a variable-frame-rate clip; one-pass writer outputs the right frames; bad clip isolated.
- **Web:**
  - Two concurrent dispatches give disjoint batches.
  - Collecting twice gives identical rows, with no `IntegrityError`.
  - A cancelled task gets no rows.
  - A superseded or stale batch result is ignored.
  - A crash between claim and invoke is recovered.
  - A failure record re-queues clips singly, up to 2 attempts.
  - The `local` backend is unchanged. Existing tests that patch `spawn_sampling_async` keep passing with `SAMPLING_BACKEND=local`.
- **Live:** 5 clips end to end, then 500 with throughput and cost.

## 9. Decisions (made 2026-09-24)
1. **Archived clips: A + B.** Lifecycle rule to Glacier Instant Retrieval,
   and restore June into it (§4.7).
2. **Scaling: fast, up to 4 instances.** Keep ~20 sampling batches in flight
   (≈ 4 instances at target 5). 9,000 clips ≈ 2 h. Auto-label can wait up to
   ~25 min behind a big sampling run; revisit (lower K or a dedicated endpoint)
   if that hurts.
3. **Credits: charge for GPU sampling.** Charge per clip sampled, using the
   same credit mechanism as pre-annotation (`annotations/views.py:1702-1706`),
   priced from measured GPU time per clip (§6). Show the cost on the add-page
   confirm and the Sample panel before starting.
4. **Combine sampling and pre-annotation** into one GPU pass (§4.1): decode
   once; SAM 3 confirms insects and its boxes are kept. No separate
   motion-only mode. The batch Auto-label panel stays for frames sampled
   before this.
5. **The user picks the classes** (as in the Auto-label panel); they are the
   SAM 3 prompts and decide which frames are picked.
6. **Measure SAM 3 speed first** (§6), before sizing candidates and batches.

## 10. Out of scope
- A global "top X frames across all clips".
- A dedicated sampling endpoint (revisit if auto-label waits prove painful).
