# 43 · Selectable, tunable MOT trackers in the pipeline

Status: **plan** (2026-10-05) — awaiting decisions below.

## Ask
The pipeline's **MOT — Track Objects** step offers only BeeTrack. Expose the
standard trackers (ByteTrack, BoT-SORT, …) so a user can pick one and tune it.

## Audit
- `track.mot` already has a `tracker` select (one choice, `beetrack`). The value
  rides the job config (`executors._pipeline_tracker`) but `_spawn_gpu_job`
  never forwards it and the worker ignores it — "inert on the worker today".
- `BeeTracking` always builds `mot.bee_tracker.BeeTracker` (Kalman + Hungarian
  + resurrection). Everything downstream reads that object's surface:
  `tracker.update(rows, frame) -> list`, `tracker.get_active_tracks()`,
  `tracker.tracks[*].{id, is_confirmed, time_since_update, last_bbox, history,
  taxon, last_confidence, last_source, bee_id…}` — crops (memory/42), species,
  tracking CSV, events.
- `mot/ultralytics_tracker.py` (`UltralyticsTracker`) exists but is wired to
  nothing, and calls `model.track()` — it runs YOLO itself and ignores the
  pipeline's Detect step, so it can't follow SAM 3 or a custom model. Keep the
  idea, not the approach.
- Ultralytics 8.4.5 (already in the GPU image) ships `BYTETracker` and
  `BOTSORT` as standalone classes: `update(results, img)` where `results` only
  needs `.conf / .xyxy / .xywh / .cls` + indexing — i.e. **any detector's boxes**.
- **BoxMOT** 25.0 (pip `boxmot`, AGPL-3.0 like us): ByteTrack, BoT-SORT,
  OC-SORT, DeepOCSORT, StrongSORT, BoostTrack, HybridSORT, SFSORT, OccluBoost;
  `tracker.update(dets (N,6) [x1,y1,x2,y2,conf,cls], frame) -> (M,8)`. ReID
  trackers (DeepOCSORT, StrongSORT, BoostTrack, HybridSORT, BoT-SORT+ReID)
  download ReID weights at runtime — the worker is offline, so they'd have to
  be baked in, and the stock ReID models are trained on people/vehicles.
  Requires numpy ≥ 2.2 (check against the image's pins).

## Design
One adapter, `mot/external.py: ExternalTracker`, that wraps any of these and
presents BeeTracker's surface (so crops, species, CSV, events need no change):
- feeds each frame's detection rows (from whatever Detect produced) as boxes,
- maps returned tracks to track views with `id`, `last_bbox`, `history`,
  `time_since_update` (0 when matched this frame), `is_confirmed`, taxon from
  the detection class, confidence.
- `BeeTracking` builds BeeTracker or ExternalTracker from
  `config.tracking.tracker` + `tracker_params`.
- Motion gating (two-mode) is unchanged: frames skipped as idle aren't fed to
  any tracker; lookback frames are.

Pipeline: `track.mot` → **Tracking algorithm** select, then that algorithm's
settings (`show_if`). Executor puts `tracker` + `tracker_params` in the hashed
job config (changing them re-runs the GPU job); spawn forwards; worker passes
them to the analyzer. BeeTrack's own knobs (max age, min hits, match distance,
resurrection) get exposed too, so every choice is tunable.

```
┌ MOT — Track Objects ───────────────────────┐
│ detections → tracks                        │
│ Tracking algorithm  [ ByteTrack        ▾ ] │
│ High-score threshold        [ 0.25 ]       │
│ Low-score threshold         [ 0.10 ]       │
│ New-track threshold         [ 0.25 ]       │
│ Keep lost tracks (frames)   [ 30   ]       │
│ Match threshold             [ 0.80 ]       │
└────────────────────────────────────────────┘
```

## Open questions
1. Which trackers: Ultralytics' two (no new dependency), + BoxMOT's motion-only
   ones (OC-SORT, SFSORT, BoT-SORT w/o ReID), or + the ReID ones too?
2. Settings: a curated few per tracker (as above), or every native parameter?
