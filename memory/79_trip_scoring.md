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
