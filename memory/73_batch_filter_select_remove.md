# 73 — Batch: filter by site and review, metrics follow the filter, select, remove clips

**Status:** built (2026-11-01).

## Asked (user, 2026-11-01)
- Filter a batch by site "and stuff".
- The review metrics should follow the filter.
- Remove clips from a batch, as Add clips adds them.
- (From memory/72) update only some clips.

## Built
- **Filters** (`_clip_filter.html`, `static/js/clip_filter.js`):
  - a **Site** menu (two or more sites) and a **Review** menu (reviewed / not reviewed);
  - rows carry `data-f-site` (`leftout.video_site`: the chosen site, else the file-name site), `data-f-reviewed` (`review.clip_touched`) and `data-f-run`;
  - `aggregate.batch_rows` adds `site` and `reviewed` and `select_related("site")`; `batch_sites(rows)`;
  - the site is also in the search text;
  - the filter dispatches `clipfilter:change` with the visible rows.
- **Metrics follow the filter:**
  - the review section moved to `_review_metrics.html`;
  - when filtered, the page POSTs the shown run ids to `batch_review_metrics` (anyone who can open the batch, via `_batch_runs`), which renders the same template with `scope` ("For the N clips the filter shows…");
  - clearing the filter restores the whole-batch HTML;
  - `initReviewMetrics(root)` re-wires the Count-in-metrics form and the cut-off slider.
  - Only the review metrics follow the filter; outcome, GPU time and downloads stay batch-wide.
- **Select** (launcher only):
  - a Select button shows tick boxes on finished and failed rows;
  - the bar offers "Select all shown (N)", Clear, **Update these** (posts `which=picked` to `batch_apply_groups`; disabled when the pipeline is up to date) and **Remove from batch** (with a confirm);
  - the Update card keeps "All clips" and "Clips reviewed"; its tick option from memory/72 was replaced by this.
- **Remove:** `batch_remove_clips`:
  - finished or failed runs get `batch_id=None`, `removed_from_batch=<batch>` (migration 0021);
  - running ones are refused ("cancel first");
  - results, reviews and votes stay on the run;
  - removed runs are hidden from the Runs page's single runs;
  - removing the last clip goes to Runs.
- **Put back:** Add clips (`run_on_videos` with `add_to`) re-attaches removed runs of those clips (`_return_removed`) instead of launching: no GPU, reviews back.
- **Migration 0018** now defers `removed_from_batch`: it uses the live model, which broke fresh databases.
- **Tests:**
  - `tests/test_batch_select.py`: site and review attributes, metrics subset and access, remove and put back, running stays, only the launcher, update these;
  - a jsdom drive of the rendered page checked the site and review filters, the metrics fetch and swap, select all shown, the remove post, and clearing.

## Effects worth knowing
- Event-classifier training (`eventml`) only reads reviews on runs in a batch, so a removed clip's reviews leave the training data until it's put back.
