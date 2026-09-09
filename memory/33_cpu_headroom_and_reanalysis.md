# 33 — CPU headroom on the GPU boxes, and re-analysis without re-paying

Two pieces of forward work, recorded so they are not re-derived. Neither is
built yet. See [[32_pipeline_failures_rerun_and_results_review]] for how the
capacity problem was found.

## Part A — bounding per-job CPU

### Where it stands

The capacity fix landed: one SAM 3 job per instance across four instances, and
one model per process. Measured on the 2026-09-09 re-run (12 clips):

| | Before (≈3/instance) | After (1/instance) |
|---|---|---|
| GPU memory | 72% | **20%** |
| CPU peak | 360% of 400% | **251% of 400%** |
| CPU mean | — | ~120% |
| Outcome | 3 of 12 | **12 of 12, 1 late failure** |

The shared model is visible in the GPU-memory drop. But **one job still peaks
above 60% of a 4-vCPU box**, and one clip in the re-run still died on
"could not get a response" — the same starved-health-check failure, now rare
instead of routine.

So the 8-concurrent rung (2 jobs per instance) is not available: two jobs at
that peak is 500% of a 400% box.

### Why one job uses two cores

Phase 4 gave decode its own thread so it overlaps inference. That was the right
change for a single run — and it doubled the CPU threads per job, from roughly
one core to two. The gain was real; the cost was invisible until three jobs
shared a box.

On top of that, **OpenCV's thread pool is unbounded by default**: `cv2` sizes it
to the machine, so decode and resize inside one job can each fan out to 4
threads. Nothing in the container caps it, and nothing sizes it against the fact
that several invocations share one process.

### What to do, in order

1. **Cap the OpenCV pool** — `cv2.setNumThreads(n)` at container start, n≈2.
   One line, no architecture change, and it is the lever with the best ratio of
   effect to risk. Measure the peak again before anything else.
2. **Cap torch's intra-op threads** likewise (`torch.set_num_threads`), which
   otherwise sizes itself the same way.
3. **Only then** revisit 2-per-instance, with CloudWatch confirming the peak
   stays near 300% before trusting it.
4. **Consider `ml.g5.2xlarge`** (8 vCPU, same A10G) if the caps cost throughput.
   Right-sizing beats packing when the constraint is CPU rather than GPU.

The measurement to take first: the `stage_seconds` breakdown from one clean run.
If `decode` is a small share, the reader thread is buying little on SAM 3 clips
and could be disabled for that detector alone — cheaper than any of the above.

## Part B — re-analysing without paying for the GPU again

### The vision

Open a run, change its ANALYZER — visitation to foraging trips, say — and see
the new answer without re-running detection and tracking.

### It is already half-built, by accident

`engine._gpu_cache_key` hashes the user, the block type, and the effective job
config (clip + ROI + analysis flags). **An analyzer is a local step**: swapping
it does not change the detect/track step's config, so the cache key is
identical and the GPU result is reused. Re-analysis is already free — there is
simply no way to ask for it.

The same cache is a hazard for benchmarking (a GPU-side fix returns the old
result, which is why "re-run all" passes `fresh=1`) and the enabler here. Both
follow from it keying on the work rather than on the build.

### What is missing

1. **Per-run pipeline variation.** `rerun` re-runs the run's pipeline as it
   stands now, so editing the analyzer edits it for every run of that pipeline.
   What is wanted is a fork: this clip, this graph, one step changed.
2. **A "change the analyzer" affordance** on a run — swap the terminal step and
   re-run from there. `retry_step` already resets a step and its descendants in
   place, which is most of the machinery.
3. **Saying the cost out loud.** "Re-analyse — reuses the tracking, no GPU
   time" is the sentence that makes it obvious this is cheap. Users will not
   infer it from a cache they cannot see.

### The shape to aim for

A run's page offers **Re-analyse with…**, listing the analyzers compatible with
what the run already produced (its artifact type — tracks or detections). Choose
one, and a new run is created with the same frozen steps, the analyzer swapped,
and the GPU step served from cache. It completes in seconds and costs nothing.

Worth checking before building: whether `StepResult` rows survive long enough to
rely on, and what happens when the cached result predates a change to the
tracking CSV's schema.
