# 48 · More clips per GPU instance (cost per clip)

Status: **paused** (2026-10-08). Infra knob built, test not run. Dev is at 1.

## Ask
Cut cost per clip by running more than one clip on each video instance. Measure
CPU, RAM and GPU first to confirm there is room.

## Cost today
- Video endpoint: `ml.g4dn.xlarge` (1× T4, 4 vCPU = 2 physical cores
  hyper-threaded, 16 GB RAM), about $0.74/instance-hour on demand, us-east-1.
  That price is from memory and was not checked against the live price list.
- 98-clip batch `91dc8dc3` (10-min clips, 1920×1080 @ 25 fps): GPU time 18.1 h,
  about $13.30, about $0.14/clip. Billing also covers boot (~5 min) and idle
  time before scale-in (≥3 min), so the real cost is a few % higher on a batch
  and up to ~2× on a lone re-run.
- `max-capacity` is 4, so 98 clips took about 4.5 h of wall clock.

## Measured: 1 clip per instance (that batch, 2026-10-08, ~01:50–06:20 UTC)
CloudWatch `/aws/sagemaker/Endpoints` (variant-level, 5-min):

| | average | peak | of |
|---|---|---|---|
| CPUUtilization | ~130% | ~150% | 400% |
| GPUUtilization | ~23% | ~28% (spikes to 70%) | 100% |
| GPUMemoryUtilization | 6–7% (~1 GB) | 7% | 16 GB |
| MemoryUtilization | 18–21% (~3 GB) | 23% | 16 GB |

From the worker logs, over 94 clips:
- Tracking took a median 616 s for ~15,000 frames: **23.9 fps**
  (p10–p90 574–804 s, range 16–34 fps).
- Post-tracking (crops, upload) took a median 26 s.
- ModelLatency per clip was a median 644 s.

Reading: the GPU is mostly idle. The CPU work (decode, Python tracker, crops)
sets the pace, and a clip uses about 1.3 vCPU.

Earlier data point (2026-10-07, image of that day): 3 clips per instance gave
~12 fps each, so **~36 fps per instance = 1.5× throughput**, with each clip
taking twice as long. That run also starved gunicorn's `/ping`, giving
"could not get a response" failures.

## Why not 2×: one process
`sagemaker_backend/serve` runs gunicorn with `--workers 1 --threads 4`, so all
clips are threads in one Python process. C code (OpenCV decode, torch/YOLO,
numpy) releases the GIL and overlaps. The tracker loop (`bee_tracker.py`),
row building and crop bookkeeping are pure Python and take turns.

- `inference.py` caps the OpenCV and torch pools at
  `BEEMONITOR_CPU_THREADS=2`, process-wide.
- Each clip builds its own `BeeMonitor`, which loads its own YOLO models
  (`video_analyzer.py:47-48`), so concurrent clips don't share a model object.

## Built (7c3fce6)
- Pulumi config `invocations-per-instance` (default 1) sets
  `max_concurrent_invocations_per_instance`.
- It also sets the autoscaling target (queued jobs per instance), so an
  instance is added only when the running ones are full.
- The value is in the Model/EndpointConfig name (`…-r5-c<N>`), so changing it
  rolls the endpoint like an image bump, in either direction.
- `Pulumi.dev.yaml` has it at **"1"** while paused.

## Plan when resuming
1. Set `invocations-per-instance: "2"` and `pulumi up`. Do this when no clip
   is processing, because an endpoint update can kill an in-flight clip.
2. Run 8 clips from batch 91dc8dc3 through the same pipeline with
   **"From scratch" ticked**. Without it, the step cache hands back old
   results and the GPU never runs.
3. Compare against the table above:

| Measure | Baseline (1 per instance) | Decision |
|---|---|---|
| Frames per second per clip | 24 | 16–20 expected |
| Frames per second per instance | 24 | keep if ≥ ~34 (1.4×) |
| CPU | 130% / 400% | stay under ~320% |
| GPU | 23% | fine up to ~80% |
| "could not get a response" | 1 in 98 | any rise means go back to 1 |

4. Back out: set "1" and `pulumi up`.

## Next option if threads plateau (~1.5×): a process per clip
- Keep one light gunicorn process for `/ping` and `/invocations`.
- Hand each clip to a pool of N spawned worker processes, using
  `ProcessPoolExecutor` with `mp_context="spawn"` (not fork, which breaks
  CUDA). Each worker builds its own `CloudPipeline` once.
- Each clip then gets its own interpreter and GIL. Expect close to 2× per
  instance, limited by the 4 vCPUs (2 × ~130% ≈ 260%).
- `/ping` is never starved by clip work.
- Cost: ~0.5–1 GB GPU and 1–2 GB RAM per worker, ~20 s warm-up per instance,
  and a GPU image build.
- Two gunicorn workers would not be a substitute: gunicorn isn't load-aware,
  so both clips can land in one process.

## Other levers not yet looked at
- Raise `max-capacity` above 4 (account quota for ml.g4dn.xlarge endpoint
  usage). This shortens wall clock but doesn't change cost per clip.
- An instance with more vCPUs per T4 (e.g. g4dn.2xlarge, 8 vCPU). Pricing has
  not been checked.
- Make the Python tracker cheaper per frame.
