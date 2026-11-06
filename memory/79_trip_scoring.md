# 79 — Trip scoring: one list, scored against true trips

**Status:** built (2026-11-05). Replaces the Before and After lists and votes on both (memory/74, 78).

**Why:** the user asked to simplify. Reviewing trips of false events let one exit get two Yes votes.

## Rule
- Reviewers answer only the trips of the reviewed events: rejected out, added in. Yes answers are the true trips.
- `period.trip_scores` pairs the detected events and the reviewed events over every clip of a unit or site, then keeps trips with both ends in clips marked reviewed (`ClipReview`).
- Right: a pipeline trip that is a true trip. Wrong: any other pipeline trip, by a No or because its events didn't survive review. Missed: a true trip the pipeline didn't make.
- A listed trip with no answer, a dispute or "can't tell" waits and counts for nothing.
- Precision = right / (right + wrong). Recall = right / (right + missed).

## Limits
- Metrics read the slim copy, which has no markers, so the scored pipeline trips ignore marker pairing. The page's list uses markers. A clip set with marked bees could score lower than it should.
- `TripReview.before_review` and `relinked` are no longer read by metrics.

## Trips CSV (2026-11-06)
- One file replaces the Reviewed trips CSV, which is gone from the batch, the share link and Review trips. This drops memory/62's rule that only reviewed pairs are exported: the user asked for trips with their review, as the events CSV.
- `period.trip_rows(runs, reviewed, calls)` and `trip_scores` read the same `_trip_sets`, so the file gives the metrics: keep `clip_reviewed` = true; right = pipeline + accepted; wrong = pipeline + rejected, event_rejected or pairing_changed; missed = review + accepted; the rest wait.
- Offered with a Trips node (`period.has_trips`) on the batch, the share link and the clip widget. A clip's file pairs inside that clip only.
- Calls on trips that no longer exist are in no export. Pairing has no markers, so a call on a marker pair isn't attached to a row.
- Events CSVs gained `clip_reviewed`.
