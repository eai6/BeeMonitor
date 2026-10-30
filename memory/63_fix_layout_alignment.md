# 63 — Fix the ROI and references for a clip or a range of days

**Status:** plan + design (2026-10-30), not built. Design canvas: https://claude.ai/artifact/3LY7eueuVYe5sDZHiHdpLS (boards "Editor", "Check alignment", "After saving"). Requested from job 19929 (`mendels_2024-05-23_18_20_01`): the hotel shifted on some days, so the drawn tubes sit beside the real holes and events land on the wrong tube.

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## What exists (audit, 2026-10-30)

- **Layouts:** a unit layout is versioned (`DeviceLayoutVersion`) and takes effect from when the unit received it (`applied_at`). Site and single-clip layouts are `videos.Layout`, which belongs to a site **or** a clip (memory/52). Lookup order: `devices.layouts.layout_for_video` checks the clip, then the site, then the unit version in use when the clip was recorded.
- **Editor:** `/videos/<pk>/layout/` (clip) and `/videos/sites/<pk>/layout/` (site), drawn on a full-resolution frame (`/videos/<pk>/frame/?t=`). Its tools are rectangle, polygon, move, delete, rename and history. There's no way to move **everything at once**, which is what a shifted camera needs.
  - Saving from a clip goes to its site or to the clip only.
  - A **unit clip** with no site can only save to the clip. There's no way to say "these days".
- **Results page** (`/analysis/<job>/results/`): it shows the regions on the video but doesn't link to the editor.
- **Applying a change:**
  - References are used after tracking, on the web. `recount.recount_run` re-resolves the reference steps from the current layout and recounts events and interactions from the saved tracks, with **no GPU**. The overlay, CSVs, batch table and clip counts all read the recounted outputs.
  - The ROI is different: the GPU **masks tracking to the ROI** (`cloud/wrapper/pipeline.py` `_build_manual_nests`). A moved ROI only becomes exact after tracking is re-run.
- **Reviews:** `EventReview.event_key = action|reference|track_id|frame`. Recounting against moved tubes can change an event's tube, so its vote no longer matches anything. `AddedEvent.reference` is the tube id the reviewer picked.

## The plan

### 1. Find the days that drifted: alignment check
For a unit or a site, the **Check alignment** page shows one frame per recorded day, each with the layout that day's clips use drawn over it. A shifted day stands out at a glance. Each tile shows:
- the date;
- the layout source ("unit v3", "days 21–24 May", "this clip");
- the number of clips and of analysed clips;
- a **Fix** button that opens the editor on that day.

*Proposed:* the frame is taken 2 s into the day's first clip. **Show every clip** expands a day into its clips.

### 2. Fix it: editor additions
- **Move all:** drag the whole layout (ROI and every reference together), or nudge it with the arrow keys (1 px, or 10 px with Shift). A toggle, "references only", leaves the ROI where it is.
- **Scale all**, from the layout's centre, for a camera that moved closer or further away.
- **Ghost:** the layout before your change stays faintly drawn, so you can see how far it moved.
- **Opening it:** it opens from the results page ("ROI & references · from unit mendels v3 · **Fix alignment**") on the current frame of that clip, from the alignment check, and from the clip page.

### 3. Who it applies to
When saving, choose one of:
- **This clip only.** This already exists.
- **These days:** from–to, in the unit's or site's local time. It defaults to the clip's day. It shows "applies to 46 clips (31 analysed)".
- **From this day on:** the same, with no end date.

A days layout is for **analysis only**. It's never sent to the unit (the unit keeps its own drawn layout).

**Data:** `videos.Layout` gains `device` (FK, null), `starts_on` and `ends_on` (dates, null). Each layout belongs to exactly one of a clip, a site or a unit. Only site and unit layouts can carry days.

**New lookup order:**
1. the clip's own layout;
2. a days layout for its unit or site that covers the clip's local date (when several overlap, the most recently saved wins, and the editor warns about the overlap);
3. the site;
4. the unit version.

The results page names the source, for example "days 21–24 May (v2)".

### 4. Update the results
After saving, the editor asks: **Recount N analysed clips now**.
- If **only references moved**, the recount runs in the background sweep with no GPU, and finishes in seconds to minutes. The clip page, overlay, CSVs and batch page all move together.
- If **the ROI moved too**, the recount still uses the new tubes, but tracking stays limited to the old ROI. The editor says so and offers **Re-run tracking for N clips (≈ X GPU-hours)**. That's a normal batch, so the user sees what it costs before it starts.
- Every recounted run records which layout version it used (`run.context` reference step plus `layout_version`), so a run says which drawing it was counted against.

### 5. Reviews survive a recount
*Proposed:*
- A vote whose event is still there (same track, action and frame, now at a different tube) **moves with it**. The match is on track, action and frame, with the old tube kept in `review_notes` as "tube 41 → 51 after layout fix".
- A vote whose event is gone is kept but marked **outdated**. It's left out of the CSV and metrics and listed under "N votes on events that changed".
- An added event keeps the tube the reviewer picked, because a person chose it.

## Built first (2026-10-30): the user's "what I need now"
- **Editor:** "Move all" mode (drag moves every reference object; "with the ROI" also moves the ROI), arrow-key nudges (1 px, Shift 10 px) for the selection or for everything, and a dashed white outline of the layout as loaded once it changes. In the shared `layout_toolbar.html` and `layout_canvas_script.html`, so the unit editor has it too.
- **Results page:** "Fix tube positions" opens the clip editor on the current frame, with a link back.
- **Recount:**
  - After saving a clip or site layout, the editor says how many finished runs used the old one and offers **Recount events** (`videos:layout_recount`, `site_layout_recount`).
  - `recount.recount_clips` recounts up to 8 runs at once. More are handed to the sweep (`counting_rules=0`).
  - If the ROI moved, the editor says tracking used the old ROI.
- **Votes don't follow a moved event** (decided while building, replacing §5's proposal). A vote judged the tube, so it stays on the old event key, shows as no longer detected, and the recount reports how many need a fresh look.
- **Not built:** days layouts, the alignment check, re-run tracking.
- **Tests:** `apps/pipelines/tests/test_layout_recount.py`. jsdom drive of Move all → save → recount.

## Build order
1. `Layout.device`, `starts_on`, `ends_on`, the new lookup and the source label. Saving "these days" or "from this day on" from the clip editor. Tests for the lookup (overlaps, time zone, the day boundary).
2. Move all, scale all and ghost in the shared editor (`layout_canvas_script.html`). A "Fix alignment" link on the results page.
3. Recount after saving (queue the affected runs, reusing `recount.recount_run`), ROI-moved warning, re-run tracking offer. Reviews moving with their events.
4. Alignment check page (unit and site), with day tiles and clips per day.

## Questions
1. Should days be **whole days** in local time (planned), or also allow a time range within a day (for a camera bumped at 2 pm)?
2. **Recount automatically on save** (planned, shown as a progress line), or only when the user presses the button?
3. Should a vote **move with its event** to the new tube (planned), or should every vote on a recounted clip be marked outdated and need re-review?

**Feedback:**
