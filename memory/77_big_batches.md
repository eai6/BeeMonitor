# 77 — Batches of thousands: open the page, launch in turns

**Status:** built (2026-11-04), not yet committed. Follows [[76_scalable_run_data_plan]].

## What happened (production, 2026-11-04)
- A launch of about 6,300 clips stopped at 4,831. The user reports 1,500+ clips never registered. Workers were killed for memory at 01:00:57, 01:01:04, 01:07:51 and 01:12:03; the launch's own request was never logged, so which kill ended it isn't proven.
- Opening that batch killed workers again (18:48:00, 18:48:14). No load of it appears in 40 h of logs. The 408-clip batch loaded 261 times with no kills.
- Only 6 clips run at once: about 40 an hour, so about 4–5 more days from 2026-11-04.

## Measured (synthetic, sizes guessed: 20 KB steps, 100 KB result summaries on finished clips)
- Before: 353 MB peak at 1,000 clips. `JobResult.summary_stats` was most of it.
- After: 124 MB peak at 4,831 clips, 12 MB of HTML, 5.6 s locally.

## Changes
- `batch_rows` defers `Job.config`, `stage_seconds` and `JobResult.summary_stats`.
- `runsummary.share_steps`: `_batch_runs` reads in chunks; runs share identical step dicts and one Pipeline. These runs are read-only.
- Batch page stops reloading itself every 15 s past `AUTO_REFRESH_CLIPS` (500).
- `aggregate.batch_video_ids`: the clips in a batch from steps alone (`run_on_videos`, `_add_batch`).
- Launch in turns: an AJAX launch starts `LAUNCH_CHUNK` (300) clips and answers `more` and `batch_id`. The hub resends the same form with `add_to`; clips in the batch are skipped, so resending is safe.
- Reconcile: `.order_by()` before `.distinct()` (it listed one user 50 times); `reconcile_user_runs` rechecks oldest first.

## Not done
- A non-AJAX launch (batch page re-run form) isn't chunked.

## Paging, Cancel and Re-run (same day, user: "we need pagination")
- Above `BIG_BATCH` (500) the page renders `CLIPS_PER_PAGE` (100) rows and sends `#clip-index`: every clip's filter values (`aggregate.row_attrs`, also the source of each row's `data-f-*`).
- `clip_filter.js` builds stand-ins from the index, so filter, sort, re-run ids, review metrics and Update work as before. Only the page on screen is fetched (`pipelines:batch_rows`, `_clip_rows.html`). Page is in the URL (`?page=`).
- Selection keeps picks in a Set, so they survive pages; "Select all shown" takes every filtered clip.
- The clip viewer steps within the page on screen.
- No auto-refresh above `BIG_BATCH`.
- **Cancel:** one UPDATE cancels every waiting job, the few on the GPU get markers, then runs close `TURN_CLIPS` (300) a request. Waiting jobs were never sent, so they need no marker.
- **Re-run:** `TURN_CLIPS` a request, moving forward by pk (`after`), so a clip that fails again at once can't loop.
- The share page still renders every row.

## Clip rows and the trips page (same day, user)
- A clip's row on the batch page opens its results page, where events are reviewed: the whole row (`data-href`, script in batch.html), the still and the time as links. "crops & CSVs" is gone and so is the inline viewer, for clips. A photo batch keeps its viewer. Fix tubes stays on the results page.
- The period page (`analysis:period_review`) is now "Review trips" only: event review, its card and Add missed event are gone from it; marks still jump to their moment.
- Trip review has speed buttons (0.25×–16×), remembered per browser (`pr-speed`), back to 1× on leaving review.

## Trips: precision, recall and the Trips module (same day, user)
- User's idea: fully reviewed clips hold every real exit and entry, so true trips can be rebuilt from them and the pipeline's trips scored against them.
- `period.trip_quality`: pairs from detected events vs pairs from reviewed events (accepted + added, not rejected), over stretches of fully reviewed clips, per place and day. A trip matches when both ends are the same event. A stretch stops at a clip not fully reviewed.
- These replace the trips stats on the batch page ("Suggestion right" is gone from the panel). Trip calls don't feed it.
- `suggest_trips`: one to one; an exit pairs only with the next exit-or-entry at its tube, when that's an entry; same day; marker first.
- `analyze.trips` module in the builder: shortest (seconds) and longest (minutes). Read live from the pipeline (`period.trip_rules`), so changes need no re-run or Update.
