# 74 — Shared review calls; Reviewed by filter

**Status:** built (2026-11-02).

## Asked (user, relaying a reviewer, 2026-11-02)
1. Filter a batch's clips by who reviewed them, so one reviewer doesn't re-review the other's clips.
2. Every reviewer on a batch may review, change and remove the others' reviews, e.g. remove an event someone else added that was already detected.

## Built
- **One call per event and per trip.**
  - `review.set_event_vote` and `period.set_trip_vote` replace any other reviewer's call on the same event or exit, across the runs sharing the GPU job (`shared_job_runs`). Taking a call back removes it for everyone.
  - `review.current_calls(run, user)` gives `{key: (verdict, by)}`, with "you" for the viewer. The player's event rows gain `e[10]` = by; the period page's events gain `called_by`; trips' `mine` gains `by`.
  - The player shows "by <name>" next to another reviewer's call and on the review card.
  - Older votes that disagree still read "disputed" (current call None) until someone calls the event again, which clears the others.
- **Added events:** `review.remove_added` lets anyone who can review the run remove one (`can_review`). The overlay's `removable` flag follows.
- **Reviewed by filter:**
  - `review.reviewers_by_run(runs)` counts event calls, added events, clip-reviewed marks and trip calls;
  - `aggregate.batch_rows` adds `reviewers`, and `batch_reviewers(rows)` lists the names;
  - rows carry `data-f-by` (space-separated usernames);
  - the menu offers Anyone, each name ("You (name)" for the viewer) and "Not by <name>" (`!name`);
  - the review metrics follow it, like the other filters (memory/73).
- **Not changed:** Mark clip reviewed stays per person (`ClipReview`).
- **Tests:** test_event_review (change, take back, legacy disputed), test_missed_events (any reviewer removes), test_period_review (trip call changed), test_batch_select (`data-f-by`, menu), test_track_overlay. A jsdom check covered the by / not-by filter.
