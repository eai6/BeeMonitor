# 78 — Trip types

**Status:** built (2026-11-05). Design canvas: https://claude.ai/artifact/Pv4HiYgBbgwRvhs6c5RAXX

**Goal:** bees sometimes leave one tube and go straight into another. Those exits and entries aren't foraging. We can't always detect this reliably, so nothing is discarded. Each trip gets a type, and the analysis decides what counts as a foraging trip.

## Audit
- `period.suggest_trips` pairs per tube and never looks across tubes. It is called only by the Review trips page.
- Event keys are `action|reference|track|frame`. Slim run summaries keep action, reference, track and frame, so the type can be worked out without loading whole runs.
- Track numbers belong to one clip. A hop across two clips can't be seen.
- Added events can have a blank track.
- A track is lost after the user's 2 s setting, so a stay in a tube over 2 s splits the bee into two tracks.
- `set_trip_vote` keeps one call per trip, whoever made it (memory/74). Metrics are in `review.batch_review_summary`, the CSV in `period.trip_rows` (also the share link's trips CSV, through `sharing.batch_csv`).

## Types
- **Direct** (`direct`): no hop seen at either end. The default. It means "no hop seen", not proof that the bee went outside.
- **Tube hop** (`tube_hop`): the trip's exit, or its entry, belongs to a hop.
- New types can be added later: one more choice, the same fields.

## The rule
- Per clip and per track, take the detected exit and enter events in frame order.
- An exit whose next event on that track is an enter is a hop. Both event keys are marked.
- This includes a return to the same tube (the bee never left view). The user accepted this recommendation.
- A trip is a Tube hop when its exit key or entry key is marked.
- Pairing, the trip list, and the Before and After lists don't change. No event is discarded.
- Only detected events are read. Rejected events still count, and an added event isn't linked; a reviewer can set the type by hand.

## Where it is stored
- The automatic type is never stored. One function, `period.hop_keys(rows)`, works it out from a run's event rows whenever it's needed, so older reviews get a type and a re-track updates it.
- `TripReview.trip_type`: blank by default, set only when a reviewer changes the type. Migration 0026 adds it. Blank means automatic.
- The reviewer's choice wins everywhere. On a trip with no call yet, it's saved with the Yes or No. On a called trip, changing it saves straight away.

## Page (Review trips)
- A **Show** filter: All, Direct, Tube hop, with counts. Progress and Previous/Next follow it.
- A **Trip type** control on each trip, with the reason (for example "track 14 left tube 5 and went into tube 12") or "changed by" a reviewer.
- `TripReviewView` takes `trip_type`. `period_views` adds `trip_type` and `type_reason` to each trip.

## Metrics
- `batch_review_summary` adds `by_type`, the same right, wrong, missed, precision and recall per type. The All row stays as it is.
- Hop keys are read from slim runs, and only for runs that have trip calls.
- The batch page shows All, Direct and Tube hop rows.

## CSV
- Two new columns in `TRIP_FIELDS`: `trip_type` and `trip_type_set_by` (`auto` or `reviewer`).

## Cost
- Web only: no GPU, no re-run, no Update, no effect on the big run.
- One small migration: a blank column on `TripReview`.

## Not covered
- Hops split across tracks, or across clips: they stay Direct unless a reviewer changes them.
- An ID switch in the tracker could mark a false Tube hop. A reviewer can change it.

## Tests
- `hop_keys`: hop to another tube, same tube, split tracks, two clips, blank track, exit with no entry, entry with no exit.
- A trip is a Tube hop by its exit alone, or its entry alone.
- The reviewer's type wins in the page, metrics and CSV.
- Metrics by type add up to the All row, slim and whole alike.
- The batch page still never selects `context`.

## Docs
- `docs/platform/runs.md`: Trips section and the CSV table.
- Trips node description, if it needs to say what the types mean.
