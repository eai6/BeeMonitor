# 60 — Activity extraction: one source of truth for tracks, events and interactions

**Status:** built 2026-10-29 (not committed). Decided in review: never-left = `stay`; edges counted + flagged (`at_edge`); start/end from detected rows; overlap = warn (nearest centre); site time zones; old clips left as they are; confidence settings only on Detect (ML scoring on/off, `ml_confidence` in the CSV); the worker never runs the nest model itself; trips and the other higher-level analyses removed. Build notes at the end.
Related: [[46_live_tracking_overlay]].

## Principles (from the user's review)
1. **One source of truth.** The GPU worker measures (detections, tracks). The web computes everything derived (events, interactions) in one place. The worker no longer writes events, interactions or trips files, and no count anywhere comes from the worker.
2. **Only what the pipeline defines.** The ROI and the references come from the pipeline (a saved layout, or a Detect node for the reference class). There are no fallbacks, no worker-found nests and no hidden rules. If a reference is missing, the result says so instead of guessing.
3. **Measured, not manufactured.** We don't infer or patch data. A value we can't measure is left blank and the reason is given.
4. **Low-level blocks.** We provide tracks, events and interactions. Trips, visit rates and other higher-level analyses are left to the user. We remove the trips outputs.
5. **Timestamps, not frames.** Rows carry real date-times (clip start + frame ÷ fps), plus seconds into the clip so the moment can be found in the video.
6. **Distances in body lengths**, measured from the insect's own detections, not as pixels or fractions of the frame.

## Findings in today's code (v1, still true)

| # | What happens today | Fix (step) |
|---|---|---|
| F1 | The Events table mixes in the worker's own Entry/Exit rows. Those are counted against the worker's nests, use 40 px padding and an ML filter, and are marked `source=gpu`. | Step 3, 7 |
| F2 | With no reference in the pipeline, it falls back to the worker's detected nests (`nest_bboxes`). | Step 3 |
| F3 | A Detect node for the reference class that found nothing falls back to the worker's nests. | Step 3 |
| F4 | With no reference, the worker's bee↔nest interactions (50 px centre distance) are kept. | Step 3 |
| F5 | Trips (batch CSV, Foraging page, daily trips) pair the worker's events. | Step 8 (removed) |
| F6 | Event totals on the jobs list, video page, dashboard, analytics and monitor agent are the worker's counts. | Step 7 |
| F7 | A lost track keeps writing predicted rows for up to `max_age` (1 s), and the tracking CSV doesn't say which rows are predicted. Ends, durations and entry times are inflated. | Step 1 |
| F8 | The overlay caches events forever, so it shows stale events. | Step 6 |
| F9 | `class` in the Tracks CSV is blank (it reads `class`, but the CSV has `taxon`). | Step 5 (removed) |
| F10 | Event filters (≥0.2 s, hand-off, window edges) drop real fast exits. | Step 2 |
| **F11** | The Events node's **"Event Confidence"** setting is the worker's ML threshold. The events actually shown (computed on the web) **never use it**, so the setting does nothing to them. | Step 2 |
| **F12** | When no tubes are drawn, the worker runs the bee-hotel nest model by itself (`nest_fallback=True`), even if the pipeline never asked for it. | Step 3 |
| **F13** | Interactions drop spells under 0.5 s by default, which hides quick pass-overs. | Step 4 |
| **F14** | Insect↔insect proximity silently skips every frame with more than 200 tracks. | Step 4 |
| **F15** | Uploaded clips whose start time came from the file name store that wall-clock time *as UTC*, with no time zone. Clips with no known start use the upload time (flagged `recorded_at_source=upload_time`). | Step 9 |

## The event rule

Inputs:
- the pipeline's references (they shouldn't overlap; a run whose references do is warned, see step 3);
- each track's **detected** rows only (step 1), so predicted rows never decide anything.

"In R" means the centroid is inside R's drawn shape (the polygon outline when there is one), grown by the node's **padding** (in body lengths; default 0).

For each track, take its **start** (first detected row) and **end** (last detected row):

| start | end | left R in between? | events |
|---|---|---|---|
| in R | outside R | — | **exit** R (at start) |
| outside R | in R | — | **enter** R (at end) |
| in R | in R | yes | **exit** R (at start) + **enter** R (at end): it came out, flew, went back |
| in R1 | in R2 | — | **exit** R1 + **enter** R2 |
| in R | in R | no, never left | **stay** R (from start to end): came to the entrance and turned round, or sat basking |
| outside every R | outside every R | — | none. Passing over a reference is an **interaction**, with its duration. |
| no references in the pipeline | | | none, and the node says "No reference in this pipeline" |

- No duration minimum, hand-off filter or edge filter.
- **`at_edge`** = the event's time is the first or last frame analysed (the clip's, or the chosen window's). It is counted and flagged so it can be filtered out.
- **ML scoring (optional, off by default; decided 2026-10-29).** The worker's trained event classifier moves to the Events node as one on/off checkbox: "Score events with the ML classifier".
  - There is **no confidence setting** in the node, and nothing is dropped. When it is on, every exit/enter gets an `ml_confidence` (0–1) in the CSV, and the user decides their own cut-off.
  - The "Event Confidence" field (F11) is removed.
  - The node does not caveat the model; what to do with it is decided later.
  - With it off (the default) you see the plain rule.

## Columns after the change

Times:
- `timestamp` = clip start + frame ÷ fps, ISO 8601 to the millisecond, in the clip's local time with offset. It is blank when the clip's start time is unknown (`upload_time`).
- `clip_start` = the clip's recorded start, the same on every row of a clip. Blank when unknown, so anyone can fill it in and compute `clip_start + clip_sec` themselves.
- `clip_sec` = seconds into the clip, which is what the video player seeks to.
- Frames remain only in the Tracking CSV, where each row *is* a frame.

### Tracking CSV — one row per track per frame (worker writes; the web serves it with timestamps added)
| column | meaning |
|---|---|
| frame | Frame index in the clip. |
| timestamp, clip_sec | As above. |
| track_id | The id within this clip (not an identity across clips; that is `marker_id`). |
| detected | 1 = matched a detection this frame; 0 = the tracker's prediction while the insect was lost. **New.** |
| x1, y1, x2, y2 | Box, in pixels. On predicted rows it is the last detected box, moved to the prediction. |
| cx, cy | Centre, in pixels (the point events and interactions use). |
| confidence | Detector confidence of this frame's match (blank on predicted rows). |
| species, species_confidence, species_votes, species_vote_share | The track's species vote, as the model called it, with no floor (renamed from `taxon*`). |
| marker_id, marker_method, marker_confidence, marker_votes, marker_vote_share | The track's marker vote (renamed from `bee_id*`, so not bee-specific). |
| ~~source~~, ~~mode~~, ~~taxon_crops~~ | Removed. |

### Detections CSV — raw detector boxes, before tracking
`frame, timestamp, clip_sec, x1, y1, x2, y2, cx, cy, confidence, class` (`taxon` renamed to `class`: it is the detector's class, not a species vote). `source` and `mode` are removed.

### Tracks CSV — one row per track
| column | meaning |
|---|---|
| track_id | |
| start_time, end_time | Timestamps of the first and last **detected** row. |
| start_clip_sec, end_clip_sec | Same, in seconds into the clip. |
| duration_sec | end − start. |
| species, species_confidence, species_votes, species_vote_share | The model's call and its confidence, always. There is no floor and no `unidentified`; the user filters on `species_confidence`. `species_best_guess` is removed. |
| marker_id, marker_votes, marker_vote_share | |
| ~~class~~, ~~first_frame~~, ~~last_frame~~, ~~start_sec~~, ~~end_sec~~, ~~frames_seen~~ | Removed. Detected vs predicted rows are in the Tracking CSV. |

### Events CSV — one row per exit / enter / stay
| column | meaning |
|---|---|
| timestamp, clip_sec | The moment: start for exit, end for enter, start for stay. |
| track_id | The insect (renamed from `subject`). |
| action | `exit` / `enter` / `stay`. |
| reference | `reference_<n>`, the pipeline's id (renamed from `target`). |
| duration_sec | `stay` only: how long it stayed. |
| at_edge | true/false. |
| ml_confidence | 0–1, the classifier's score for exit/enter. Blank unless ML scoring is on; blank for `stay`. Nothing is filtered on it. |
| + species…, marker_id… | The insect's identity. |
| + provenance (batch) | video_title, device_name, site_name, location, … |
| ~~frame~~, ~~time_sec~~, ~~subject_kind~~, ~~target_kind~~, ~~source~~ | Removed. |

### Interactions CSV — one row per spell together
A spell is a run of **detected** frames in which the two were together. A gap longer than `gap_seconds` (default 0.6 s) or a change of reference ends it.

| column | meaning |
|---|---|
| start_time, end_time, start_clip_sec, end_clip_sec | |
| duration_sec | Detected frames held ÷ fps. |
| track_id | The insect (renamed from `a`). |
| with | `reference_<n>`, or the other insect's track id (renamed from `b`). |
| relation | `in_reference` (centroid in the reference, plus padding) or `near` (two insects within the radius). |
| min_distance_bl | `near` only: the closest they came, in **body lengths**. |
| + species…, marker_id…, with_species, with_marker_id | |
| ~~start_frame~~, ~~end_frame~~, ~~start_sec~~, ~~end_sec~~, ~~a_kind~~, ~~b_kind~~, ~~min_distance~~, ~~source~~ | Removed. |

**Body length** (for proximity, padding and `min_distance_bl`) is measured per frame from the detected boxes: the long side of each insect's box, averaged over the two insects so that A–B reads the same as B–A. It is not taken from the frame width. The radius setting becomes "Insect ↔ insect radius (body lengths)", default 2.

## Plan

**1. Detected vs predicted rows (F7).** *Worker + web.*
- The worker writes `detected` from `time_since_update == 0`, and blanks `confidence` on predicted rows.
- The web uses detected rows only for event starts and ends, interaction spells and track times. Predicted rows are still drawn in the overlay (dashed).
- **Old clips have no `detected` column**, and we won't infer it. They are left as they are for now (decided 2026-10-29).

**2. Event rule (F10, F11).** Rewrite `ops.compute_nest_events` as the table above.
- Remove `ENTRY_MIN_SECONDS`, `HANDOFF_*` and `EDGE_SECONDS`.
- Add `stay`, `at_edge`, `padding` (body lengths) and the ML scoring checkbox (on/off only). Remove the `event_confidence` field from the Events node and legacy blocks, and from the seeded templates.
- The web image needs scikit-learn and the classifier file. That is a Dockerfile change; CI builds it, no local build.

**3. References only from the pipeline (F1–F4, F12).**
- Drop `events_from_gpu`, the `detected_references` fallback, the `nest_bboxes` fallback in `detected_reference`, and `keep_gpu` in interactions.
- **The ROI is the tracking area, never a reference.** Remove the "use the hotel as the reference when there are no tubes" rule.
- **The worker never runs the nest model by itself** (decided 2026-10-29). Remove `nest_fallback` / `_fill_nests`' model path. The nest model runs only as a Detect node the pipeline contains (or the legacy `reference_source=detect`, which is the pipeline asking). No tubes drawn and no such node means no references.
- **References shouldn't overlap: warn, don't refuse** (decided 2026-10-29).
  - The layout editor marks overlapping shapes.
  - A run whose references overlap still runs, and its results carry a warning naming the pairs and the rule used.
  - The rule: a point inside two references belongs to the one whose centre is nearest. One rule, stated in the warning, so nothing is double-counted silently.
- Each output says where its references came from: `layout` / `detect:<class>` / `none`.
- With no reference you still get insect↔insect interactions, but no events and no insect↔reference interactions.

**4. Interactions (F13, F14).**
- The minimum length defaults to 0, so every pass-over is kept with its duration.
- Proximity and padding are in body lengths.
- Remove the 200-track cap: the pairwise distance works per frame, and a frame of 1,000 tracks is 1M distances, which is fine. If that ever proves slow, use a spatial grid, not a skip.

**5. Tracks (F9).** The columns above. `class`, frames and `frames_seen` are removed; times come from detected rows.

**6. Overlay (F8).**
- The cached file holds boxes only. Events, including `stay` and `at_edge`, load live from the run's saved rows, so any rule change shows at once.
- Bump `overlay.VERSION`.

**7. Worker stops writing derived files; counts come from the run (F1, F6).**
- Remove `events.csv`, `interactions.csv` and `foraging_trips.csv` from the worker. That takes out `EventProcessor` scoring, `InteractionAnalyzer` and `compute_foraging_trips` from the job.
- `JobResult.total_events` / `entry_count` / `exit_count` / `interaction_count` are filled from the run's computed rows when the run finishes (and on recount). The jobs list, video page, dashboard, analytics and monitor agent keep reading the same fields, so their numbers become the pipeline's.

**8. Remove trips (F5).**
- Remove the batch Trips CSV and its button, the Foraging page (`analysis/foraging.py`), `compute_daily_trips`, `DailyForagingSummary`, and trip charts on the batch page.
- The legacy `analyze.foraging_trips` / `analyze.visitation` blocks, still in saved pipelines, run as plain Events / Interactions.
- The stored daily-trip table is exported once to CSV, then dropped (decided 2026-10-29).

**9. Timestamps (F15).**
- `timestamp` is built from `Video.recorded_at` + frame ÷ fps (the clip's measured fps).
- Time zone: device clips use the device's `tz_name`. Uploaded clips are named `site_timestamp`, and that time is the site's local time. **Each site gets a time zone** (decided 2026-10-29): a new `Site.tz_name`, asked for on the site form and at upload when the site has none. The stored `recorded_at` becomes true UTC from it. Until a site's time zone is set, its clips' `timestamp` is written without an offset and the site page asks for it. Existing clips are re-based when their site's zone is first set.
- **When the clip's start can't be established** (`recorded_at_source=upload_time`: not from the device, the file's metadata, the file name or the uploader), `timestamp` is **blank**. `clip_sec` is always filled, and each row carries `clip_start` (blank in the same case), so the user can add their own start time (decided 2026-10-29).
- Caveat: frame ÷ fps assumes a constant frame rate. Phone clips can be variable-rate. Check at ingest and say so on the clip; don't correct it.

**10. Recount.**
- Bump `recount.CURRENT_RULES` 2 → 3. Clips that have a `detected` column are recounted from saved tracks (no GPU).
- Add a management command that recounts everything at once rather than waiting for the sweep.
- Old clips without `detected`: left as they are for now, labelled "counted under the old rules".

**11. Downloads and contract.**
- The column renames and removals above apply to every download: clip page, batch, public share link, API.
- Internal code that read the removed names (`subject`, `target`, `a`/`b`, `*_kind`, `source`, `frame` on events) is updated with the rest. Nothing keeps a dead column for compatibility.
- The API docs and the user docs page "What an event is" explain the rule, the worked cases and every column.

**12. Tests.**
- One per row of the rule table.
- A predicted tail drifting off R.
- Overlapping references: the run completes with a warning naming the pair, and a point in the overlap counts once, for the nearest centre.
- No reference → no events, even with `nest_bboxes` and a worker events file present.
- Padding in body lengths. ML scoring on fills `ml_confidence` and drops nothing; off leaves it blank.
- Timestamps: device tz, file-name time, unknown start.
- On one clip, these agree: the events table, the Events CSV, the overlay and the job's counts.

Order: 1 (worker) → push → 2–6 and 9–11 (web) → 7–8 (removals) → 10 recount. The GPU image-tag bump is separate, and the user runs Pulumi.

## Confidence settings (decided 2026-10-29)
A confidence setting belongs only where it changes what the next step sees: the **Detect** node, where it decides which detections go on to tracking. Everywhere else the confidence is given in the CSV, and the user filters in post-processing.

| node | setting | change |
|---|---|---|
| detect.objects / detect.bee / detect.nest | `confidence` | **Kept.** |
| track.bee (detect + track in one) | `confidence` | **Kept** (it is the detection threshold). |
| analyze.events / analyze.foraging_trips | `event_confidence` | Removed. ML scoring is on/off; `ml_confidence` goes in the CSV. |
| identify.species | `min_mean_confidence` | Removed. Species and `species_confidence` always go in the CSV; there is no `unidentified` floor and no `species_best_guess`. |
| filter.confidence | `min_confidence` | Nothing to do: already hidden from the palette, and it passes data through. |
| analyze.detection_count | `iou_threshold` | Kept. It is not a confidence; it decides when two boxes are the same object. |

Saved pipelines that have these values keep running: the removed values are ignored, and the editor no longer shows them.

## Open questions
- ~~Q1~~ — old clips left as they are for now.
- ~~Q2~~ — overlapping references: warn (nearest-centre rule stated in the warning).
- ~~Q3~~ — each site gets a time zone.
- ~~Q4~~ — decided: remove the higher-level analyses. Remove the Foraging page, daily trips and the trips outputs (step 8). The stored daily-trip table is exported once to CSV, then dropped.
- ~~Q6~~ — no change: `filter.confidence` is already a hidden legacy block (not in the palette) that passes data through unfiltered (`_exec_filter_passthrough`).

## Build notes (2026-10-29)
- **The worker keeps its raw column names** (`taxon`, `bee_id*`, `source`, `mode`); the web translates on export (`tracks.tracking_export` / `detections_export`, `primitives.export_rows`). Old files need the translation anyway, and the web code reading `taxon`/`bee_id` keeps working.
- **Worker:** `detected` column (exact for BeeTracker: `time_since_update == 0` at emission); blank confidence on predicted rows; nest model only when the payload has `detect_nests: true` (sent by the web only for legacy `reference_source=detect`; the nest-only job still runs it); no events/interactions/trips files.
- **Saved rows stay in frames**, exported in time, so setting a site's time zone later moves every download without a recount.
- **Timestamps:** `apps/videos/timing.py` (`clip_start`, `iso`, `timestamp_at`). `Site.tz_name`; filename times localised in `Video.save()` from `metadata.recorded_wall`; re-based when a site's zone is set.
- **ML scoring:** `apps/pipelines/ml_scoring.py`. The 20 features are copied from `EventProcessor._extract_trajectory_features` (importing it pulls in torch). The web image gets scikit-learn 1.7.2 (the pickle's version) and `models/event_classifier_model.pkl`.
- **Overlay:** the stored payload has boxes only (VERSION 3); events and references are served live with `?part=events`. `lost` comes from `detected` only; older files draw every box solid.
- **Counts:** `recount.sync_job_counts` copies the run's counts to JobResult at completion and on recount. `CURRENT_RULES = 3`; the recount skips (keeps) clips without `detected`. `manage.py recount_runs [--all]` does everything at once.
- **Removed:** `analysis/foraging.py`, `DailyForagingSummary` (migration 0013 exports the rows to `processed/exports/daily_foraging_summary_<ts>.csv`, then drops the table; it aborts if the upload fails), the trips sweep, `compute_daily_trips`, `backfill_foraging_trips`, the batch Trips CSV, the unrouted Analytics page and `analytics.py`, the filtered downloads that streamed the worker's files, and the species floor everywhere.
- **Layout editor:** overlapping reference objects are drawn amber and named under the hint.

## After deploy (2026-10-29)
- **No events in production:** with Identify species on, the tracking table's `taxon` column holds the voted species, and the analyzers matched it against the Detect class ("bee"), which emptied every one. Fix: `executors._tracks_label` skips class filtering when the pipeline has one Detect class. The worker now writes the detector's class in its own `class` column (never overwritten), and `ops._CLASS_COLS` prefers it.
- **Backfill everything (the user's call):** every finished run is recounted from its saved tracks (`CURRENT_RULES = 4`; sweep 60 runs per pass, within 90 s). Clips without `detected` use all their rows and carry a note that starts and ends may include predicted positions. This replaces the earlier "old clips left as they are".
- **One source, per table (2026-10-29):** the clip page computed interactions for a run whose pipeline had no Interactions step (`primitives_for_job` fell back to `recompute_primitive`), and so did batch exports (`primitive_csv`, `available_downloads` "tracking ⇒ both"), while the batch table read only saved outputs ("—"). Now a run's tables are only what its own steps saved: `executors.measured_kinds(job)`; the clip page shows "—" + "No Interactions step in this pipeline", its CSV 404s, and batches offer only the tables some run's steps made. Jobs with no pipeline run (the old analysis form) are still counted against the clip's saved layout.
