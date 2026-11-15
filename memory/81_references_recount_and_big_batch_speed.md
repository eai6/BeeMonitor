# 81 — Fixing references across a big batch, fast recounts, a fast batch page

**Status:** built (2026-11-15), not yet pushed. Phase 4 needs `pulumi up` on `infra/aws`. Design canvas: https://claude.ai/artifact/6wYXiGUrmmm8cT2XWEgc9Q

**Goal:** fixing the tube references for the 6,306-clip Mendels + Natalies batch
(b437f428) is a day-by-day pass on one page, where it is always clear which
days are fixed and which clips still carry old counts; the recount of
thousands of clips takes minutes, not most of a day; and the batch page opens
in about a second.

## What happened (2026-11-15)
- The user fixed references by filtering the batch to a day, opening a clip,
  editing and saving "for this day", day after day. One save went to "this
  clip only", and some saved days were never recounted. Nothing says which.
- Re-extraction (the layout recount) is very slow, and the batch page is slow.

## Measured (App Runner logs, last 7 days)
| What | Loads | Avg | Max |
|---|---|---|---|
| Batch page `GET /pipelines/batch/b437f428…/` | 61 | 24 s | 45 s |
| … `?status=running` | 17 | 32 s | 116 s (past the 115 s worker timeout) |
| … `?day=…&site=…` (a filtered view) | 13 | 27 s | 36 s |
| Next page of rows (`rows/`, 100 clips) | 4 | 6.5 s | 8.7 s |
| Retry failed | 4 | 74 s | 199 s |

- The page's HTML is **2.4 MB** on every load (the clip index for 6,306 clips).
- Recount: **~140 runs an hour** at best (both workers), ~50 s a run a worker.
  Every recount logged a `BeeTracker initialized`: **each one re-tracked the
  clip on the web** before counting it (141 re-tracks vs 137 recounts in the
  01:00 hour). One re-track: 2–25 s of tracker CPU, a download of the saved
  detections, three CSV uploads.
- Each batch page load also downloads up to 8 whole raw files of clips with no
  measured length, to probe them: those are the broken files, which never get
  a length, so the same ones are downloaded on every load
  ("thumbnail: no decodable frame": 112 on 2026-11-15; S3 pool-full warnings).

## Audit

### Recount (`pipelines/recount.py`, `analysis/reconcile._recount_loop`)
- Runs as one daemon thread in each of the 2 gunicorn workers of the one
  1 vCPU / 2 GB App Runner instance, beside the page requests.
- A layout recount of more than 8 runs sets `counting_rules=0` on them; the
  sweep takes 60 at a time, newest-completed first, within a 90 s budget, and
  rests 20 s when the queue is empty. No order by request, no progress shown.
- Per run: the whole `context` read, `_retrack_if_needed`, Identify Species
  regrouped, references resolved, events and interactions recomputed (tracking
  CSV from S3, pandas), the whole context and its slim copy written, votes
  moved.
- **The re-track is redundant for this batch.** `retrack.run.tracked_with`
  takes a run never re-tracked as tracked "under older rules" (no
  `lost_rules`), so every run whose MOT node uses `lost_search=speed` or
  `stay_in_reference` differs from `settings_of` and is re-tracked once. The
  GPU image has carried lost rules v2 since 08993f3 (`Pulumi.dev.yaml` bumped
  2026-11-01 14:11), and the web tracker is the same code, so a GPU job from
  after that was tracked exactly as the web would re-track it.

### The layout workflow (`videos/views.py` LayoutEditorView, LayoutRecountView)
- Save and recount are two buttons. A save changes nothing already counted.
- Scopes: site (one `Layout` on the Site), day (`same_day_clips`: a copy on
  each clip of that local day), clip (one copy). Lookup is clip, then site,
  then unit (`devices.layouts.layout_for_video`), so **a later site save never
  reaches a clip that has its own copy**, and a day save makes every clip of
  that day its own copy.
- What a run was counted with is only a label on the reference step's output
  (`source_label`, e.g. "site mendels (v3)" or "this clip (v2)"), not an id, so
  "counted with an older layout" can't be asked.
- The batch page has a Day filter, but reaching a day's layout means opening a
  clip; there is no list of days with their layouts.

### The batch page (`pipelines/views.batch_detail`)
- Rows are paged (100 drawn, the rest fetched by `batch_rows`), **but every
  number is recomputed from all 6,306 runs on every load**. Nothing is stored
  per batch.
  - `_batch_runs`: every run with `steps` (tens of KB each, de-duplicated only
    after loading) and its slim `summary` (every event row kept).
  - `aggregate.batch_rows`: all Videos, Jobs, JobResults; review counts and
    reviewers for all.
  - `row_attrs` for every clip → `clip_index` JSON in the page (the 2.4 MB).
  - Review metrics with trip scores, insect-group summary, update status,
    `regroup.batch_changes` (each run's steps against the pipeline), failure
    groups, downloads, reference summary, period reviews.
- The filter is client-side: `?day=` or `?status=` builds the whole batch and
  then hides rows in the browser, so a filtered view costs the same.
- `PROBES_PER_RENDER` probes clips with no duration, which re-downloads the
  broken files on each load (memory/80's file check fixes the cause; until
  then a failed probe must not be retried on every load).

### Infra (for moving work off the web)
- RDS `db.t4g.micro` (≈85 connections, burstable CPU), private subnets, a NAT
  gateway already, App Runner reaches it through a VPC connector.
- The web image is in ECR (`latest`), so a Lambda can run the same image.

## Plan

### Phase 1 — stop the waste (web only, small)
1. **No redundant re-track.** The GPU worker writes the tracker's
   `lost_rules` into `summary_stats`; `tracked_with` uses it. For jobs already
   done: a job created after the 08993f3 image went live (2026-11-01 14:11
   -05:00) is taken as lost rules 2. A run whose MOT settings were changed
   since still re-tracks, as now.
   - Effect: a layout recount is counting only. Expect a few seconds a run
     (to be measured), so 6,306 runs ≈ 1–2 h even on the web.
2. **Recounts go in the order asked for.** A `recount_requested_at` on the run
   (set by a layout save/recount, Update this batch, a rules bump); the sweep
   takes oldest request first, so the day just fixed is done next, not after
   the rest of the library. The 20 s idle rest is skipped when a request
   arrives (a cache flag the loop checks every 2 s).
3. **A failed probe isn't repeated on each load.** `metadata.probe_failed_at`;
   the batch page skips those (the file check in memory/80 then names them).

### Phase 2 — the batch page in about a second (web only)
1. **Per-run columns for listing and filtering**, kept up to date wherever a
   run's results or reviews are saved: `video` (FK), `local_day`, `site_label`,
   `clip_state` (completed / failed / cancelled / skipped / running),
   `failure_cause` (the failures key), `events_n`, `interactions_n`,
   `tracks_n`, `gpu_seconds`, `reviewed`, `reviewers` (text), `layout_id`
   (Phase 3). Indexed on `(batch_id, …)`. One migration + a backfill command
   (chunks of 200, from the slim copy).
2. **Filtering and paging in SQL.** The filter is the query string
   (`?day=&site=&status=&reviewed=&by=&q=`); the page is the first 100 rows of
   that query; each menu's counts are one GROUP BY. No clip index sent: the
   page drops from 2.4 MB to ≈150 KB. Select-all, Remove clips, Retry, Update
   these act on "the filter" server-side, in turns of 300 (memory/77), not on
   a posted list of 6,000 ids.
3. **A stored roll-up per batch** (`BatchStats`: counts by state, failure
   groups, days, sites, reviewers, GPU-hours, update status, "pipeline
   changed"), marked stale when a run of the batch finishes, is recounted or
   reviewed, and rebuilt by the background worker, never in the request. The
   page shows "as of 2 min ago" while a rebuild is pending.
4. **Heavy panels after the page shows**: review metrics, insect groups,
   references and "pipeline changed" load as fragments.
- Target: first load < 1.5 s, any filter < 1 s.

### Phase 3 — a References page for the batch
- **Each run records the layout it was counted with**: `layout_id` (and the
  number) on the reference step's output and on the run column; older runs
  backfilled by parsing `source_label` "(vN)" against that clip's or site's
  versions.
- **Out of date** = counted with a layout that isn't the clip's current one.
- **Batch → References** (one row per site and local day):
  frame with the current layout drawn on it · layout in use (site vN, this
  day's own vN, or mixed) · clips · counts up to date (e.g. 14 of 22) ·
  actions: **Edit day**, **Copy previous day's**, **Use the site's** (clears
  the day's own copies), **Recount N**.
  - Filters: "Out of date", "Own layout", "Mixed", a site.
  - A **Recount everything out of date (N)** button with live progress.
- **The editor, opened from it**: "Save for this day" queues that day's
  recount itself (a tick, on by default); **Save and next day** goes to the
  next day with the same drawing preloaded. Saving for the site warns: "312
  clips on 14 days have their own layout and won't change: apply to them too /
  keep them".

### Phase 4 — recounts off the web: SQS + Lambda
- An SQS queue; the web sends `{run_ids}` (10 a message) wherever it now sets
  the recount flag. A Lambda (the web image, entry point
  `apps.pipelines.recount_lambda.handler`, `awslambdaric` added) recounts
  them, claiming each as the sweep does now.
- Reserved concurrency 8 (the micro database: connections and its CPU);
  15 min visibility, a dead-letter queue after 3 tries; private subnets with
  the RDS security group; an S3 gateway endpoint so S3 reads skip the NAT.
- The in-process loop stays as the fallback when no queue is configured
  (development, tests) and sweeps up anything the queue missed.
- Expected: 6,306 runs in ≈15–25 min at 8 workers; cost ≈ $1–2 a full pass.
- Pulumi change (the user runs it) + a CI step that also points the Lambda at
  the new web image (it must be a Docker v2 manifest, as for SageMaker).
- Later candidates for the same queue: batch Retry and Cancel turns, the
  `BatchStats` rebuild, exports of big batches.

## Order
Phase 1 (unblocks the recount now) → Phase 2 (the page; its columns feed
Phase 3) → Phase 3 (the References page) → Phase 4 (Lambda). Fixing days can
carry on during Phases 2–3; Phase 1 makes each fix recount in minutes.

## Open questions
1. Phase 1.1 assumes every job after 2026-11-01 14:11 -05:00 ran on 08993f3 or
   later: the time `pulumi up` actually went through may be later. A run near
   the cut that re-tracks anyway only costs time, not a wrong count.
2. Phase 3 "out of date" for a run counted before this ships relies on the
   "(vN)" in its label; a label without one is shown as "unknown", not as
   out of date.
3. Phase 4: Lambda vs a Fargate task. Lambda chosen: per-run work is seconds,
   it scales to zero and the queue gives retries and progress for free.

## Built (2026-11-15)
- **Phase 1:** `retrack.run.gpu_lost_rules`: a GPU job created after
  2026-11-01 20:00 UTC (first 08993f3 instance 19:19 UTC, from the
  endpoint's log streams), or a result whose `summary_stats.lost_rules` says
  so (cloud/wrapper/pipeline.py writes it from the next GPU image), isn't
  re-tracked. `PipelineRun.recount_requested_at` (0027) orders the sweep by
  request; `recount.queue()` is the one way to ask, and wakes the loop
  (`recount.WAKE`). `thumbnails.unreadable`: a probe that found no frame and
  no length isn't repeated by pages.
- **Phase 2 (changed from the plan):** the clip filter stays in the browser
  (selection, review metrics and the re-run form all follow it). What made
  the page slow was reading every clip's results to build it:
  `PipelineRun.listing` (0028, `listing.py`) holds the clip, errors, counts
  and the layout counted with; `listing.rows` builds every row from it, the
  clips' rows, jobs and reviewers by GPU job. `batchstats.py` keeps the
  totals that need every clip's results (downloads, references, review and
  group summaries, update status, changes) on `BatchStats`, under a
  fingerprint of the batch's runs, reviews and pipeline; a big batch works
  them out in a background thread when stale. The page is gzipped.
  Synthetic 3,000-clip batch locally: 4.3 s → 0.78 s a load.
- **Phase 3:** `references.py` compares the geometry a run's reference step
  counted with against the clip's layout now (no layout ids needed, so old
  runs work too). `references_views.py` + `batch_references.html`; the
  editor's recount tick, Save and next day (`?from=` carries a drawing),
  previous / next day links, and the site-save question (`override_own`).
- **Phase 4:** `recount_queue.py` (SQS, 20 runs a message),
  `recount_lambda.py` (Lambda runtime API loop, no new dependency), the
  queue, DLQ, function (concurrency `recount-concurrency`, default 8) and
  IAM in `infra/aws/__main__.py`; deploy.yml builds the web image as a Docker
  v2 manifest and points the function at each new image. Unset
  `BEEMONITOR_RECOUNT_QUEUE_URL` and the web's loop recounts as before.
