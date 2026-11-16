# 83 · References a model finds, arranged by the pipeline (no day checks)

Status: **built** (2026-11-16), not yet committed. Replaces the numbering half
of memory/82 (template matching, day flags, the References page).

## The ask

> "there are only suppose to be 60 tubes … from top left to bottom right. 10
> for each row. 6 rows."
> "the daily check was meant to be a solution in place of the model detection.
> Doing daily check for many days from multiple sites is difficult … This whole
> reference page should go away."
> "when we use a model for reference detection. the user should also specify
> the number of rows and columns to be detected at the site."
> "No hand-drawn layout should not replace model. The reference should obey the
> pipeline. if set to model then model it should be. if set to device drawing
> then we use drawing."
> "nest tube is just the reference for my batch or study. Other studies may
> have other references or even context that is not grid like."
> All tubes are grids for now, but won't always be; numbering by rows, top
> left first, is what was done by hand.

## Why memory/82's numbering went wrong

Detected boxes were matched to the site's drawn layout (shift + zoom, Hungarian
pairing). Anything that didn't pair became a flag: a template tube unmatched
was "kept", a detection unmatched was "new" (numbered 61, 62, 63 … past a
60-tube hotel, see the 3 false boxes along the bottom board). Every flag made
a day "to look at", and days to look at needed a person: the same daily check
the model was meant to remove.

The analysis package already numbered tubes without a template:
`src/beemonitor/detection/nest_detector.py` (`NestDetector`) clusters rows by
y, fills gaps by row spacing, numbers row by row, and checks 6 × 10 ± 3. Its
weak points: 10 per row hard-coded (`rows_10`, `< 10`, `row_idx * 10`),
10-pixel thresholds, extras never dropped (a row of 13 just fails the check),
columns not aligned across rows.

## The design

**The pipeline decides; nothing mixes.** The Reference node's source is either
the saved layout (drawing: used exactly as drawn) or *found by a model*. In
model mode a clip's own drawing no longer wins (`drawn_wins` is gone), and no
drawing is compared against.

**Arranging is a setting of the node, apart from finding.** For a model:
- **Grid** — `rows` × `columns`, numbered row by row from the top left
  (1..columns on the top row). The default, 6 × 10.
- **As found** — every box the model finds, in reading order, no fixed count:
  for references that aren't a grid (flowers, feeders, entrances…).
More arrangements can be added beside these without touching the rest.

**One grid function, in the analysis package**
(`src/beemonitor/references/arrange.py`, numpy only; the web image already has
`src/` on its path). `grid(boxes, rows, columns)`:
1. the row direction (tilt) from neighbouring boxes; rows are boxes within half
   a tube's height of each other along it; the `rows` fullest rows kept;
2. the spacing between tubes; each box's column within its row; rows aligned
   to each other, and the `columns` consecutive columns holding the most boxes
   kept — boxes outside, or a second box at one cell (the worse fit), dropped;
3. a cell no box filled is placed on its row's line at its column (the row's
   own fit, or the other rows' column when the row has few boxes);
4. fails (with a reason) on fewer than half the cells found, fewer rows than
   asked for, or fewer columns seen than asked for: never a guess at which
   edge is missing.

**No person checks days.** A clip whose grid fails borrows the grid of the
nearest clip in time at the same site (same model, class and sampling; within
3 days) that fits. Otherwise it is *not placed*: counted with no tubes rather
than wrong ones. The ROI stays the saved one (as in memory/82): changing it
would change what was tracked.

**What the batch page shows instead of the References page:** one read-only
line, "Numbered by model — 6,212 placed · 18 from a nearby clip · 3 not placed
· 40 waiting", and a live "Recounting: N clips to go" bar. Renumbering is
**Update this batch** (no separate button): a run numbered the old way shows as
"Reference · Tubes" there, and with "stays at that tube" on, as tubes to track
again. To try a few clips first: filter the clip list (or Select), then
Update "Clips the filter shows"; no new buttons. The card was cut down: no replay check button, no stored (stale) "N still to
do" count, no long explanation.

## Removed

- References page, its day page, actions and progress URL
  (`references_views.py`, `batch_references*.html`), the batch page's button.
- `TubeDayCheck` (migration drops the table).
- `tubes.match` / `_fit`, flags, `template_key`, `drawn_wins`;
  `references.days` / `current_for` / `_detected_row` / `_kind`.
- `devices.layouts.shared_layout_for_video` (only the matcher read it).

Kept: `TubeDetection` (what a model found, unnumbered: arranging is redone on
every recount, so changing rows/columns needs no new detection), the ONNX/CPU
detection, the recount worker, `references.counted` (listing, now with the
arrangement result), the layout editor's `?tubes_of=`.

## Running batches

Changing the node (arrangement, rows, columns) is a reference change: Update
re-extracts from the saved tracks; no new detection, no GPU (the tubes
fingerprint re-tracks only when "stays at that tube" is on, as before).

## Not done

- `NestDetector` (GPU image) still has its own row logic; switching it to
  `arrange.grid` needs a GPU image bump.
- Rows × columns per site (one batch, hotels of different shapes): the node's
  setting only for now; all current hotels are 6 × 10.
