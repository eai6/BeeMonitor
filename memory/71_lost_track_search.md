# 71 — Lost tracks jump to other bees (teleports)

**Status:** built (2026-11-01). Off by default; turn it on per pipeline and judge it with Update this batch.

## The problem (user, 2026-11-01)
A bee enters tube 50; its track goes lost; a different bee appears at tube 36, and the lost track takes it, so the entry and the exit become one "teleporting" track, with wrong events.

## Why (audit)
- **BeeTrack's `_gate`:** a track may claim a detection within 3 bee widths, plus **1 width per lost frame**, up to `match_distance_multiplier` (8 widths). After about 5 lost frames, a lost track reaches 8 widths (~400 px), several tubes, in any direction. The cost is distance × (1 − IoU).
- **Speed is ignored.** The gate grows the same for a bee that slowed down to enter a tube (it should stay put) as for one flying fast. There is no shrink-at-rest; the user remembered one, but it isn't in the code.

## The fix: three BeeTrack settings, off by default (today's behaviour unchanged)
1. **Lost-track search: "grows each frame" (today) | "follows its speed".**
   - With "follows its speed", a lost track's reach is `max(2 widths, last speed × frames lost × 1.5)`, capped by the max jump.
   - A bee that slowed to a stop at a tube keeps a small reach; one lost mid-flight keeps a wide one.
   - The speed is the track's velocity when it was lost (before the per-frame decay).
   - A track seen last frame keeps today's 3-width gate.
2. **"A track lost at a tube stays at that tube": off | on.**
   - If a lost track's last detected centre is inside a reference (a tube), it can only be matched to a detection within that tube's box widened by one bee width. Anything else is a new bee.
   - Uses the references in pixels: the worker passes its layout's nests; the web replay passes the run's references.
3. **"Max jump (bee widths)":** `match_distance_multiplier`, 8 as now, now settable.

## Where
- `src/beemonitor/tracking/mot/bee_tracker.py`:
  - new `BeeTracker(lost_search=, stay_in_reference=, references=)`;
  - `Track.loss_speed`;
  - `_gate` and `_allowed`.
  - The web's vendored copy is kept identical (test).
- **Worker:**
  - `BeeTracking` takes `references` and the new params;
  - `video_analyzer` passes `nests['nests']` boxes and the new settings;
  - `config` defaults.

  Needs a GPU image bump for new GPU runs.
- **Web:**
  - registry `TRACKER_FIELDS` (`beetrack_lost_search`, `beetrack_stay_in_reference`, `beetrack_match_distance_multiplier`); defaults aren't sent, so existing cache keys are unchanged;
  - `retrack/replay.tracker_params` accepts them;
  - `retrack/run` passes the run's references in pixels;
  - `REPLAYABLE` includes them, so **Update this batch** re-tracks with them on the web, with no GPU.
- **How to judge it:** on BeeMonitor_Manuscript, turn them on, Update this batch, and compare precision and recall against the existing reviews (memory/70); votes follow their events (memory/69).

## Built
- **BeeTracker** (source and vendored copy kept identical):
  - `Track.loss_speed`, set in `predict` on the frame a track first goes unmatched;
  - `_gate` "speed" mode: `LOST_BASE` 2 widths, `LOST_SLACK` 1.5;
  - `_reference_of` and `_allowed`, which pad the tube box by one bee width.
- **Worker:**
  - `BeeTracking` takes `lost_search`, `stay_in_reference` and `references`; `config` has defaults; `match_distance_multiplier` now comes from config (8 as before);
  - `video_analyzer` passes `nests['nests']` boxes.
- **Web:** three MOT fields; `retrack.replay.tracker_params` and `run_calls(…, references)`; `retrack.run._reference_boxes` gives the run's Events references in pixels; `REPLAYABLE` includes the new fields.
- **Tests** (`test_retrack.LostSearchTests`):
  - the default reproduces the teleport;
  - "speed" and "stay at tube" each prevent it;
  - a fast bee lost mid-flight keeps its id;
  - defaults aren't sent;
  - a web re-track uses the run's tubes.
- **A test that was already failing:** `src/beemonitor/tests/test_bioclip_env.py::…full_checkpoint_loading` fails locally on the clean tree too (environment).

**Feedback:**
