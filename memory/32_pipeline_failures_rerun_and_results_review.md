# 32 — Batch failures, re-running as a benchmark, and reviewing results

## Context

Batch `5af72b17-0a72-4000-aa0e-cfd1d11edc30`: **9 failed, 3 completed**. Two
things follow from that, plus one that has been asked for alongside:

1. Find out why the 9 failed and fix what is fixable.
2. Make this batch **re-runnable as a benchmark**, so the next fix can be
   measured against the same 12 clips rather than against a feeling.
3. Separately: reviewing *results* is as hard as reviewing footage was before
   [[31_gpu_utilization_and_cost_accuracy]] — mock the UX first, as with the
   review workspace, and only build once it has been commented on.

Work order is 1 → 2 → 3. This document is the audit and the plan; no code yet.

## Audit findings

### F1 — the GPU result cache cannot be invalidated by a code change

`engine._gpu_cache_key` hashes exactly three things:

```python
{"u": run.user_id, "b": step["block_type"], "c": {video + job config}}
```

**No image tag. No analyzer version.** So after shipping a GPU-side fix, re-running
the same pipeline on the same clip returns the OLD cached output and reports
success — the fix is never exercised. For ordinary use that is the point (the
cache is what makes a re-run cheap); for a benchmark it is fatal, and silently
so.

`_persist_step_result` is only reached on a successful step (`engine.py:176`,
`:297`), so **failures are not cached** — a re-run does retry the 9. But the 3
that passed would be served from cache, so a regression introduced in the 3
would be invisible.

### F2 — there is no batch-level re-run

| Affordance | Scope | Where |
|---|---|---|
| `rerun` | ONE run, current pipeline, cache reused | `pipelines/views.py:380`, run.html |
| `retry_step` | ONE step + its descendants, in place | `:417`, _run_status.html |
| — | a batch | **nothing** — batch.html has no re-run control |

Re-running these 9 today means opening 9 run pages and clicking 9 times.

### F3 — failure reasons are scattered

A job's cause of death is written to `Job.error_message` by six distinct paths
(`analysis/views.py`), and the pipeline step shows `context[sid]["error"]`:

| Message | Meaning | Fixable? |
|---|---|---|
| `Never reached the GPU …` `:1100` | spawn lost, one respawn also failed | retry |
| `SageMaker inference failed: <body>` `:1170` | the container raised — real error inside | depends on body |
| `Timed out: no result after Nh` `:1186` | past the async caps; request expired or lost | retry / capacity |
| chunk `_fail(...)` `:1273` | one chunk of a long video died | depends |
| `Ingest failed:` `:421`, `Spawn error:` `:479` | never got as far as the GPU | retry |
| step errors (`executors`) | e.g. "Selected video not found", bad ROI | code |

The batch page shows per-run status only; the reason lives one or two clicks
away, so "9 failed" cannot be triaged without opening 9 pages. That is why
step 1 below is evidence-gathering, not fixing.

### F4 — nothing records WHAT ran

A `Job` stores config, timings and now `gpu_seconds`/`stage_seconds`, but not the
analyzer version that produced them. Two runs of "the same" pipeline a week
apart are not comparable, and a benchmark cannot say which build a number
belongs to.

### F5 — recently fixed, and therefore likely in this batch's history

Three failure causes were fixed hours before this batch and may account for some
of the 9 (or may be excluded by their timestamps):

- `Selected video not found` — owner-only lookup on a device-shared clip.
- `'Sam3Detector' object has no attribute 'detect_batch'`.
- SAM 3 routed to the T4 because `SAGEMAKER_SAM3_ENDPOINT_NAME` was unset —
  the endpoint OOMs on SAM 3, so the failure would look like a crash, not a
  routing problem.

Whether these explain the 9 is exactly what step 1 settles.

## Root cause of the 9 failures — established, not inferred

Every failure object in the output bucket says the same thing:

> Amazon SageMaker could not get a response from the `beemonitor-sm-dev-sam3`
> endpoint. This can occur when CPU or memory utilization is high.

Nothing failed *inside* the analyzer: the container logs for that window carry no
traceback, and show clips decoding and crops being written. Three job ids
interleave in one stream — `pl_6b268626271f45`, `pl_4a661533088e44`,
`pl_7cf34c12fdba43` — i.e. three concurrent invocations sharing one instance.

CloudWatch names the exhausted resource:

| Metric | Peak during the batch | Capacity |
|---|---|---|
| **CPUUtilization** | **360%** | 400% (4 vCPU) — 90% saturated |
| GPUMemoryUtilization | 72% | headroom |
| MemoryUtilization | 27% | fine |
| DiskUtilization | 1.8% | fine |

**CPU, not memory.** The container could not answer SageMaker within the window
because its 4 vCPUs were fully committed, so the platform gave up on the request
and wrote a failure object. Jobs that finished before contention peaked are the
3 that passed.

Two things put it there:

1. **The SAM 3 endpoint never had a concurrency limit set.** The main endpoint's
   config sets `max_concurrent_invocations_per_instance=3`, and its comment
   justifies that for YOLO: *"One YOLO tracking job barely uses the T4 (the work
   is mostly CPU/IO), so pack several per instance."* The SAM 3 endpoint config
   passes **no `client_config` at all**, so it takes the platform default and
   packs several SAM 3 jobs onto one g5. The reasoning written for YOLO-on-a-T4
   was never re-examined for a heavy transformer on an A10G.
2. **Phase 4's reader thread doubled the CPU threads per job.** Decode used to
   alternate with inference on one thread (~1 core per job); overlapping them
   means ~2. Three concurrent jobs went from ~3 busy threads to ~6 on 4 vCPUs.
   That change made a single run faster and a packed instance worse, and this
   batch is the first to run three-up on it.

The same comment already anticipated the failure mode — *"Kept <= the
container's gunicorn --threads 4 so a thread stays free for /ping health"* — and
at 90% CPU even that spare thread cannot answer in time.

### The fix

1. **`max_concurrent_invocations_per_instance=1` on the SAM 3 endpoint.** At
   three concurrent it sat at 72% GPU memory; alone it would be ~24%, with CPU
   uncontended. Throughput comes from `max-capacity` adding instances, which is
   what autoscaling is for — packing a GPU box is the wrong lever when each job
   is already GPU-bound.
2. **Re-examine the main endpoint's 3** against the reader thread. Either drop to
   2, or cap the per-process OpenCV pool so decode cannot oversubscribe
   (`cv2.setNumThreads`), which is the cheaper and more general fix.
3. **Recognise this failure class in the app.** "could not get a response …" is
   actionable — it means retry with less concurrency, not "the analysis is
   broken" — and should not read like an unexplained crash.

None of the three recently-fixed causes (F5) is implicated. This is capacity.

## Plan

### Step 1 — evidence before fixes

Read the actual reasons, do not infer them. Two sources:

- `Job.error_message` for the batch's jobs (the app already has it — surfacing
  it is part of Step 3).
- CloudWatch `/aws/sagemaker/Endpoints/beemonitor-sm-dev[-sam3]` for the
  container-side traceback, which is what `SageMaker inference failed:` truncates
  to 500 characters.

Group the 9 by cause. Fix only what the evidence names. Expect a mix: some
retryable (lost spawn, timeout), some real (a code path in the analyzer), and
possibly some already fixed by F5.

### Step 2 — re-run a batch as a benchmark

1. **Batch re-run**, on the batch page: "Re-run failed (9)" and "Re-run all
   (12)". Creates a NEW batch from the same (pipeline, videos), linked to the
   old one so the two can be compared. Reuses `engine.launch_batch`, which
   already does per-video runs under one `batch_id`.
2. **A `fresh=1` mode that bypasses the cache.** Without it a benchmark re-run
   silently skips the 3 that passed (F1) and cannot exercise a GPU-side fix at
   all. Implementation: thread a flag through `advance_run` so the
   `StepResult` lookup is skipped — do NOT delete cached rows, since other runs
   legitimately want them.
3. **Stamp the analyzer version on the Job** (F4) — the image tag the endpoint
   is running, read once and recorded in `config`. Then a benchmark row reads
   "12 clips, build 33bab4f: 9 failed" and the next reads "build X: 1 failed",
   which is the comparison being asked for.
4. **A benchmark view**: batches of the same pipeline+videos side by side, with
   pass/fail counts and cost per build. Small; it is mostly a query over
   existing data once (1)–(3) exist.

**Open question for the user:** should a benchmark re-run bypass the cache by
default (correct for measuring, dearer) or only when asked (cheap, easy to
misread)? Recommendation: default to fresh for "Re-run all", reuse cache for
"Re-run failed" — failures were never cached anyway, so that combination is both
cheap and honest.

### Step 3 — reviewing results (mock first, then build)

Same approach that worked for the review workspace: design artboards in the
canvas, get comments, then code. Nothing here is built until that round-trip
happens.

What the mock has to answer, drawn from the audit:

- **Why did these fail?** — reasons grouped and counted, not one status pill per
  row (F3). Triage nine failures in one screen.
- **What came out?** — events, tracks, foraging trips, interactions, species —
  currently spread across per-job pages and CSV downloads.
- **Is this better than last time?** — the benchmark comparison from Step 2.
- **Re-run from where you are looking**, so a fix and its verification are one
  gesture rather than a hunt back to the batch page.

## Out of scope

Cold start and scale-to-zero remain untouched ([[31_gpu_utilization_and_cost_accuracy]]).
Phase 5b (batching the tracking loop) is still parked pending a `stage_seconds`
breakdown from a finished run — a successful run from this batch would supply it.
