# 68 — Events from the whole trajectory (an option)

**Status:** built (2026-10-31). Requested because exiting bees were counted as entries. The detector lost a bee at the tube mouth, so its last *detection* was in the tube, while the tracker's predicted positions showed it flying off.

## Decided by the user (2026-10-31)
- **Keep the Events node simple:** one setting, no direction heuristics. I had proposed following the prediction only when the bee moves away from the tube, because a bee entering a tube is also predicted onwards past the mouth for up to "Keep a lost track" seconds. The user chose the plain option.

## Built
- **Events node, "Track start and end from":**
  - **Detections only** (`detected`, the default and the memory/60 rule);
  - **Whole trajectory** (`all`): predicted rows count for where a track starts and ends.
- `ops.compute_events(..., detected_only=)`. The Events output records `track_rows`.
- **Caveat to watch:** with the whole trajectory, real entries whose prediction runs past the mouth can stop counting as entries. The reviewed events on a batch show whether it helps.
- **Applying it to a batch that already ran:** `regroup.apply_pipeline` now copies every analyzer's config (matched by step id) as well as Identify Species. The batch page has **Recount with the pipeline's settings**, or **Apply the pipeline's settings to this batch** on the Insect groups card. No GPU.
