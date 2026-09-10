# 34 — Data consistency & quality of analysis and visualisation

**Status:** Phases A and B shipped (`ceeb51b`, `a4a0a5c`, `d466359`, `c532c8a`).
One operational step outstanding — see "What still needs doing" at the end.
**Supersedes cost work.** Nothing in here is about $ or GPU throughput. The
CPU-thread cap (`6a16579`) stays committed-but-undeployed; that is a separate
thread and is not touched by any of this.

## Why this now

The pipeline runs and the pages render. What is not yet trustworthy is the
*numbers on them*. This audit traced a value from the Pi's SD card through
ingest → GPU → aggregation → template, looking for places where a number
changes meaning silently. There are six, in descending order of how wrong they
make the science.

---

## Finding 1 — fps is fabricated on the aggregation path (CORRECTNESS BUG)

Four call sites resolve frames-per-second, with three different key sets:

| site | expression | correct? |
|---|---|---|
| `apps/pipelines/ops.py:299` `fps_of` | `fps`, `video_fps`, `frame_rate` | ✅ |
| `apps/analysis/foraging.py:88` | `v.fps`, `video_fps`, `fps` | ✅ |
| `apps/pipelines/aggregate.py:102` | `video.fps or stats.get("fps") or 30.0` | ❌ **never reads `video_fps`** |
| `apps/analysis/views.py:576` (chunk planner) | `video.fps or 30.0` | ❌ |

The GPU backend writes the real rate as `summary_stats["video_fps"]`
(`cloud/wrapper/pipeline.py:320`). `summary_stats["fps"]` does not exist. And
`Video.fps` is **declared but never assigned by any code path** — grep for
`.fps =` across `beemonitor_web/` returns only the model field itself. So on
the batch-results path both terms are always empty and `DEFAULT_FPS = 30.0`
always wins.

The Pi records at 25 fps (`recorder up: main=1920x1080 lores=640x480 @ 25fps`).
Every frame→seconds conversion on the batch page is therefore **20% too long**:
dwell seconds, visit lengths, trip durations, interaction durations. The
per-video run page (`ops.fps_of`) uses 25 and the batch page uses 30 — the same
clip reports two different dwell times depending on which page you open. That
is the single worst consistency defect in the system.

**Fix**
1. One resolver, `ops.fps_of(summary, video=None)`, taking the video so the
   precedence order (`video.fps` → `summary.video_fps` → `fps` → `frame_rate`
   → default) is written once. Every call site uses it; `aggregate.DEFAULT_FPS`
   keeps its name and becomes that resolver's default.
2. Capture fps + duration + resolution **at ingest**, from the file, so the
   fallback is almost never reached: extend `apps/videos/thumbnails.py` (it
   already opens the clip with `cv2.VideoCapture` and already reads
   `CAP_PROP_FPS` at line 126 and throws the value away) to write
   `fps`, `duration_seconds`, `width`, `height` back onto the row in the same
   decode. Zero extra I/O — it is one `cap.get()` per property on a capture
   that is already open.
3. Make the fallback **visible instead of silent**: when the resolver falls
   through to the default, record it and surface "frame rate assumed 30 fps"
   on the page. A wrong number that announces itself is recoverable; one that
   doesn't is not.

`duration_seconds` is likewise never written, and the chunk planner reads it
(`views.py:573`): unknown duration → `dur <= limit` → the clip is **not
chunked**, so a long SAM3 video runs unchunked straight into the 1 h async
invocation cap. Fixing ingest fixes a live failure mode, not just a display.

## Finding 2 — `recorded_at` has no provenance

`apps/api/uploads.py:196`: `recorded_at or parsed_recorded_at or timezone.now()`.
Three sources of very different quality collapse into one field with no record
of which was used:

- device-supplied ISO (trustworthy),
- filename parse (trustworthy),
- **upload wall-clock** (not a recording time at all).

A Pi that buffered a backlog offline and flushed it on reconnect stamps every
one of those clips with the flush time. They then land on the wrong day, in the
wrong time-of-day bucket, in every time series — and look exactly like good
data. `apps/api/web_uploads.py:173` and `apps/api/pipelines.py:353` have the
same `or timezone.now()`.

**Fix:** write `metadata["recorded_at_source"] = "device" | "filename" |
"upload_time"` at every create site, and treat `upload_time` as *unknown* in
aggregation — same bucket as the existing "no recorded-at timestamp" skip,
which `aggregate.py:95` already handles honestly. Surface the count.

## Finding 3 — identity does not survive a boundary (KNOWN, PARTLY DOCUMENTED)

Track ids are unique only within one decode. Three places cross a boundary:

- `aggregate_visitation` sums `unique_visitors` across clips. **Already
  documented in the docstring and stated on the page** — leave the behaviour,
  keep the caveat.
- `_merge_chunk_results` (`views.py:1384`) sums `unique_tracks` across chunks
  of *one video*. A bee spanning a chunk boundary counts twice. This one is
  **not** disclosed anywhere, and unlike the cross-clip case it is fixable:
  `_remap_chunk_track_id` already namespaces ids, and chunks are contiguous
  frames of one clip, so boundary tracks can be stitched by IoU + frame
  adjacency.
- `aggregate_trips` pairs Exit→Entry per nest on the absolute timeline using
  only `last_exit`, with **no identity check** — `exit_track_id` and
  `entry_track_id` are recorded and never compared. A cross-video trip asserts
  the returning bee is the one that left. For a hotel with one active tube that
  is a fair assumption; with several it manufactures trips.

**Fix:** stitch chunk-boundary tracks (mechanical, do it). For trips, do not
silently assume — split the reported figure into *same-track* trips and
*inferred* trips and show both, so the assumption is a number the user can
judge rather than an invisible one. Cross-clip visitor dedup stays out of
scope; it needs re-ID, which is a research project, not a fix.

## Finding 4 — aggregates have no denominator

`aggregate_*` all report `clips = len(outputs)` — the clips that *succeeded*.
Failed and skipped clips vanish. A day where 9 of 12 runs failed (batch
`5af72b17`) renders as a clean 3-clip day with no indication that 75% of the
footage is missing. `collect_sources` already builds a `skipped` list with
reasons; the aggregators never see it.

**Fix:** carry `attempted / analysed / failed / skipped` through
`analyzer_results` into every result panel, and render coverage next to every
total. "14 visits across 3 of 12 clips" is a usable number; "14 visits" is not.

## Finding 5 — the triage counts don't sum to the total

`ProcessingHubView.triage` (`views.py:757`) counts `bee_confirmed=True` and
`bee_confirmed=False`. Clips where the key is **absent** — every web upload
(`web_uploads.py` never sets it) and every pre-tagging device clip — are in
neither bucket, so the two numbers silently fail to add up to the library size.

**Fix:** add the third bucket ("not tagged") and show it. Cheap, and it stops
the page from looking broken.

## Finding 6 — chunked runs drop `summary_stats` payload

`_merge_chunk_results` rebuilds `summary_stats` from scratch with only
`video_fps`, `chunked`, `note`. `nest_bboxes`, `total_nests`, `crops_manifest`
and the per-nest event breakdown are discarded. The `note` is honest about the
annotated video and crops; it does not mention the nest data.

**Fix:** merge `nest_bboxes` across chunks (union by nest id) and take
`total_nests` as the union's size rather than dropping it.

---

## Order of work

**Phase A — make the numbers right** (correctness; no UI change)
1. Single `fps_of` resolver + every call site through it.
2. Ingest captures `fps` / `duration_seconds` / `width` / `height` in the
   existing thumbnail decode; backfill command for the existing library.
3. `recorded_at_source` at all four create sites.
4. Chunk-boundary track stitching + `nest_bboxes` merge.

**Phase B — make the numbers honest** (visualisation)
5. Coverage (`attempted/analysed/failed/skipped`) through the aggregators and
   onto every result panel.
6. Assumed-fps and inferred-trip disclosure badges.
7. Third triage bucket.

**Phase C — regression floor**
8. Tests that pin each of the above: a 25 fps fixture that must not report
   30, a chunk-boundary track that must count once, an aggregate whose
   coverage denominator must include the failures.

## Lessons carried forward from 31–33

- **One resolver per derived quantity.** Findings 1 is four copies of one
  decision drifting apart. Same disease as the `table_kind` discriminator
  before it was centralised.
- **A silent default is a bug even when the number is right.** 30.0 was
  invisible for the whole life of the project because nothing said it was a
  guess.
- **`exclude(json__key=...)` drops rows where the key is absent** (SQL NULL) —
  re-learned in Finding 5. Prefer explicit id sets or three-way counts.
- **Ship honesty before cleverness.** Disclosing that visitors are summed
  beat implementing re-ID. Findings 3 and 4 follow the same rule.


---

## What actually shipped, and one thing it changed

Finding 1 was **worse than the audit found**. The four drifting call sites were
real, but so was a fifth failure they masked: the executors pass the whole GPU
result to `ops.fps_of`, and `PipelineResult.to_dict()` is a plain `asdict()`, so
`video_fps` sits nested under `summary_stats`. The resolver only checked the top
level, found nothing, and assumed 30 for **every analyzer** — visitation dwell,
colony-activity bins, detection windows — not just the batch page. The resolver
now descends one level.

Shipped:

- `ops.fps_with_source()` — one resolver, reports `video` / `analysis` /
  `assumed`; every call site through it; descends into `summary_stats`.
- Ingest measures `fps`, `duration_seconds`, `width`, `height` in the decode
  `videos.thumbnails` already performs. No extra I/O.
- `backfill_video_props` for the existing library.
- `recorded_at_source` at all four create sites; `Video.resolve_recorded_at`.
- `analysis.chunk_stitch` — rejoins tracks cut by a chunk seam (greedy
  best-IoU, a few frames of slack); `unique_tracks` counted from stitched rows;
  `nest_bboxes` unioned instead of dropped; chunks record `start_frame`.
- `aggregate.coverage_of()` — attempted / analysed / missing / assumed-fps,
  rendered on the batch page as "from 3 of 12 clips (25%)" plus a note that
  partial totals are a floor.
- Trips split into `confirmed_trips` (same track id, same clip) and
  `inferred_trips`; `same_track` in the CSV export.
- Third triage bucket ("Not tagged") with a matching filter.

569 tests pass, up from 530.

## What still needs doing

1. **Run the backfill against production** — nothing measured is retroactive:

   ```
   python manage.py backfill_video_props --dry-run --limit 20   # see the rates
   python manage.py backfill_video_props
   ```

   It downloads each clip once, so use `--limit` on a large library. Until it
   runs, existing clips keep resolving through the analysis value (correct where
   a run recorded one) or the assumed default (now disclosed on the page).

2. **Re-run the benchmark batch** after the backfill, so the durations on it are
   computed at the measured rate. `batch_rerun` with the `fresh` flag already
   exists for exactly this.

3. **Not attempted, deliberately:** cross-clip visitor dedup. Track ids are
   unique only within a decode, so recognising the same bee in two clips needs
   re-identification. The honest disclosure stays in place instead.
