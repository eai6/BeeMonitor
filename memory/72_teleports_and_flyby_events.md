# 72 — Fast-bee teleports, next-tube switches, fly-by entries; faster web re-track

**Status:** built (2026-11-01). Not yet pushed or deployed. The tracker changes need a GPU image bump for new GPU runs; Update this batch applies them on the web without one.

## What the user saw (2026-11-01)
1. **A fast bee's search circle catches the next bee.** A bee flies off fast and its track is lost. Its "follows its speed" reach is a circle, speed × frames lost × 1.5. The next bee out of a tube behind it (track 61 → tube 41), or one flying in, takes the id.
2. **A switch to the next tube.** Track 29 went into tube 32, and a bee came out of tube 33 and got id 29. "Stays at that tube" widens the tube box by one bee width, and tubes on this hotel are about a bee width apart, so tube 33's entrance is inside tube 32's widened box.
3. **A fly-by counted as an entry.** A bee exits 42 and flies down past 52; its last detection is in tube 52's box, so ENTER 52. This comes from the counting rule (last detection in a tube = entry), not from tracking.
4. **Re-tracking on the web is slow.**

## Fixes
1. **Heading cone** (`BeeTracker._ahead`; "speed" mode only):
   - beyond `LOST_BASE` (2 widths) from its last detection, a lost track that was moving (≥ 1 px/frame) only takes detections within `LOST_CONE` = 45° of its heading;
   - `Track.loss_velocity` is stored with `loss_speed`;
   - it applies in matching and in reviving (through `_allowed`).
2. **Nearest tube** (`_allowed`): the detection must be within the padded box of the track's tube **and** no other reference's box may be nearer to it (`_box_distance`).
3. **Reviving dead tracks obeys both settings.** Pipelines with a BeeTrack MOT node already send `max_resurrection_seconds = 0`, so this only matters for runs without one.
4. **Settle at the tube** (Events node `settle_seconds`, "At the tube at least (seconds)", default 0 = off; try 0.2):
   - `ops.compute_events(settle_frames=)`: an exit needs the track's first unbroken run at the tube to last ≥ N frames, and an entry needs its last run to;
   - `at_edge` events are exempt;
   - recount only, no GPU.
5. **Rules version** (`retrack.run.LOST_RULES` = 2, `versioned()`):
   - `settings_of` adds `lost_rules` when speed or stay is on;
   - GPU-made tracks are taken as older rules, so a batch with those settings is offered "Lost-track rules: updated" in Update this batch (`regroup.pipeline_changes`) and re-tracked once;
   - after the image bump, new GPU runs get one needless web re-track on Update (harmless);
   - `run_calls` strips the key before building BeeTracker.
6. **Faster re-track:**
   - Measured: a synthetic 5-min clip, 4 bees, every frame, replays in 5.5 s on an M-series Mac. The tracker's small numpy calls dominate. It was left alone, so worker and web keep identical numerics.
   - The real cost was scheduling: `recount.sweep` ran inside the reconcile tick (≤ 90 s of work, then a 120 s sleep), so less than half the time. It now has its own thread (`reconcile._recount_loop`), which sweeps back to back while there's work (1 s gap) and checks every 20 s when idle.
   - There is one per gunicorn worker (2), and runs are claimed atomically.
   - **Instance:** App Runner is 1 vCPU / 2 GB. With 2 workers, both sweeps share one core, and page requests too. Going to 2 vCPU / 4 GB (`instance-cpu` / `instance-memory` in `infra/aws`) would roughly double re-track throughput with no code change; the user decides.

## Tests
- `test_retrack.LostSearchTests`:
  - fast bee looked for ahead, not behind;
  - the next tube is another bee;
  - revival follows speed and stays at its tube;
  - each fails on the old tracker.
- `RetrackRunTests.test_tracks_made_under_older_lost_track_rules_are_offered_again`.
- `test_event_rule.SettleAtTheTube`.

## Update only some clips (asked and built 2026-11-01)
- **The Update card's choice:** "All clips (N)", "Clips reviewed (N)", or "Clips I tick in the list".
  - Ticks are checkboxes on finished clip rows, joined to the form with `form="update-form"`;
  - they're shown only when "tick" is chosen;
  - the JS refuses an empty tick list.
- **`regroup.apply_pipeline(only=)`:** `batch_apply_groups` reads `which` / `run`. "Reviewed" uses the new `review.clip_touched(counts)` (a vote, an added event, or Mark clip reviewed), the same test as the review summary.
- **Clips left out** keep their counts and stay listed as behind; the card says batch totals mix the two.
- **Tests:** `test_insect_groups.ApplyToBatchTests` (ticked, reviewed, nothing ticked).

**Feedback:**
- 2026-11-01: precision and recall went up with settle 0.2, but quick take-offs lost their exits (EXIT 35, track 10). Split into **Entry** (`entry_settle_seconds`) and **Exit** (`exit_settle_seconds`) on the Events node; `executors._settle_frames`. The old `settle_seconds` is ignored (no hidden setting). Suggested: entry 0.2, exit 0–0.1.
