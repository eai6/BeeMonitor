# 84 · One review page, a day at a time, with its batch's metrics

Status: **built** (2026-11-16).

## The ask

> "I reviewed those clips in another batch yet their results are here in this
> new batch. I don't need metrics for every run. I just needed it for the CV
> eval. These review and metrics things should be isolated to the review page.
> which is now crashing every time now."
> 1. Metrics only on the review page; the review page is for both events and
>    trips. 2. One day at a time. 3. (a) Reviews made in this batch only.

## Audit (what is there)

- **Batch page** carries the metrics: "Metrics for" site picks, Events review
  (precision/recall, Count in metrics Exits/Entries/Stays) and Trips review
  (`batch.html:354-411`, `_review_metrics.html`, `batch_review_metrics`,
  `batch_review_actions`, `batchstats.totals` → `review.batch_review_summary`).
- **Reviews cross batches.** No review model has a batch; metric and listing
  queries go through `review.shared_job_runs` (every run sharing the GPU job,
  any batch) and `listing._reviewers` / `batchstats.fingerprint`
  (`run__gpu_job_id__in`). Display (`event_summaries`, `current_calls`,
  `added_rows`) is this run only. So a batch's metrics count another batch's
  reviews, and can disagree with what the player shows.
- **Review page** (`analysis/period_views.py PeriodReviewView`,
  `period_review.html`, `static/js/period_review.js`) is trips only; events are
  reviewed on each clip's page. Opened from a batch it is `gather_batch`: every
  clip of one unit/site in the batch, no date limit (~3,000 clips, ~27,000
  events here), all events built into one page. That is what runs the web out
  of memory (3 workers SIGKILLed 17:16–17:18 UTC); more memory only moves the
  limit, the page grows with the batch.

## The design

**One review page per batch: one place, one day.**
`/analysis/review/?batch=…&site_id=…&day=YYYY-MM-DD` (or `device=`). Header:
place, day with previous/next, the day's progress. Opens on the first day with
something left to review. A day is 60–140 clips: bounded whatever the batch.
Trips never cross midnight (no flights at night), and are paired within a day
already (`suggest_trips` groups by tube and date).

**Events and trips on it.** The day's events listed with Yes/No and Add missed,
and Mark clip reviewed, through the endpoints the clip page uses
(`EventReviewView`, `AddMissedEventView`, `ClipReviewedView`); the trips as now.
The clip page keeps its own event review in the player.

**Metrics on it, only there.** A Metrics panel: Events (precision/recall, the
Count in metrics Exits/Entries/Stays choice) and Trips (precision/recall, by
type), for the whole batch, from the totals worked out in the background
(`batchstats`), so opening a day reads no other day. Removed from the batch
page: the panels, the "Metrics for" site picks, their JS and endpoint.

**A batch counts its own reviews.** Metric, reviewer and fingerprint queries
take runs of the same batch only (`shared_job_runs` and `listing._reviewers`
scoped by `batch_id`). The 30 clips reviewed in another batch count there, not
here. Display is unchanged (already this run only).

**Batch page:** one **Review** link per place (was "Review trips"); the clip
list's reviewed column stays, scoped to this batch.

## Built as

- `period.gather_batch(..., day=)`: one day; `days` with clips and reviewed per
  day; opens on the first day with a clip not marked reviewed.
- `_place` reads a batch's clips from its runs' steps (`batch_video_ids`), not
  every run whole (that alone held ~6,000 contexts in memory).
- `period_views._metrics`: a big batch's from `batchstats.stored`, a small
  one's now. `_review_metrics.html` included on the review page; Count in
  metrics posts `next` and comes back to it.
- `review.shared_job_runs`, `listing._reviewers`, `batchstats.fingerprint`
  scoped by batch (`batchstats.VERSION` 3).
- Page: Events / Trips switch, the day's events in turn (Yes/No, Play ±2 s,
  Next unreviewed, Mark clip reviewed, Add at this frame), playback controls
  shared. Batch page: metrics, site picks above them and
  `batch_review_metrics` removed; "Review trips" is **Review**.

## Not changing

Review models (no migration), the clip page, CSV downloads' columns (they
follow the same scoping), trip pairing rules.
