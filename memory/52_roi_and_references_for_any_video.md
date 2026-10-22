# 52 — ROI and reference objects for any video (sites and single clips)

**Status:** steps 1–3 + the sampled-reference fix built 2026-10-22 (not committed yet); "add to a training project" not built. Design canvas: https://claude.ai/artifact/9kjvmu9SWhZQ47CzB2eAB1 (boards "ROI & references editor", "Sites page, clip page, results"). v2: no model suggestions in the editor (§4).

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## The problem

A ROI (where to look) and reference objects (what visits are counted against: nest tubes, nest holes, numbered flags) can only be drawn **per unit**. Clips uploaded by hand have no unit, so they get neither. Example: ground-nesting footage with yellow numbered flags (A, C, 1) beside nest holes: no ROI, no references, so no per-nest visits.

Worse, with no references drawn the GPU worker always runs the **bee-hotel nest model** (`cloud/wrapper/pipeline.py:779` `_fill_nests`), which invents hotel tubes on footage that has none.

## What exists (audit, 2026-10-22)

- Layout = `roi_override` box, `roi_polygon`, `nest_layout` (`[{id, box, points?}]`, all 0..1), stored on **Device** with versions (`DeviceLayoutVersion`). One reader: `devices/layouts.py:59 layout_for_video`, empty when the clip has no unit. Every consumer goes through it (legacy jobs, pipeline `reference.layout`, frame sampling).
- **Site** (`videos/models.py:24`) has name, lat/lon, notes: no geometry.
- **Editor**: `devices/templates/devices/roi_editor.html` — canvas over a still, rectangle/polygon, auto-numbering, move/edit, history. Device-bound: draws on the latest heartbeat photo, saves to the device.
- A drawn **ROI crops tracking on the GPU**; references are used afterwards, locally. So a ROI change re-runs the GPU, a reference change doesn't.
- References "from detections" exist only as a side path (a Detect node wired as a reference; ids by order, not stable across clips, never saved or confirmed).

## The plan

### 1. A layout can belong to a site or to one clip
- Same shape as a unit's: ROI (box or polygon) + reference objects. Each reference gains a **name** (e.g. "A", "C", "Hole 3") so results read "Flag C: 14 visits" rather than "Reference 3".
- **Site**: shared by every clip recorded there. Versioned like a unit's, so a clip analysed in May keeps May's layout.
- **One clip**: overrides the site for that clip (the camera moved, a one-off upload).
- *Proposed:* layouts stored on a new `Layout` table (owner = site **or** video **or** device), replacing the three device fields over time; devices keep working unchanged meanwhile.

### 2. Which layout a clip uses
**Clip → its site → its unit → none.** One resolver replaces `layout_for_video`; everything that reads layouts gets sites and clips for free. The results page says where the layout came from ("Site: North bank, version 2").

When a clip has a layout from any source, the worker **does not** run the hotel nest model. It runs only when the user asks for it (a Detect node, or the unit is a bee hotel with no drawn tubes, as today).

### 3. Drawing
- The unit editor, generalised: opens from a **clip** ("Draw ROI & references") or a **site** ("Edit layout"), on a full-resolution frame of a clip (scrub to pick a clear frame), never the rotated thumbnail.
- Same tools as the unit editor: ROI rectangle/polygon, reference rectangle/polygon, move, delete, history. Plus a name per reference. No model in the loop: drawing is instant.
- Saving from a clip asks: **this clip only**, or **every clip at this site** (if it has one; a clip with no site can create one here).

### 4. References from detections — pipelines only
- **The editor has no "find references"** (decided 2026-10-22): SAM 3 is serverless and slow to start, so waiting on it while drawing is a poor experience. References are drawn by hand, as for units.
- Detected references stay where they already are: a **Detect node used as a pipeline's reference source** (e.g. a model trained to detect flags or nest holes), run as part of the analysis. Fixes there: read sampled detections (today a sampled Detect gives no references — a bug), merge boxes seen on several frames into one object, and stable ids by position.
- *Proposed:* a button on a layout, "Add these boxes to a training project", so drawn references can become training data for such a model.

### 5. Where people find it
- Clip page: "ROI & references: none · Draw" / "from site North bank · Edit".
- Upload: pick a site → its layout applies (already picks sites).
- **Sites page** (new, small): list of sites with clip count, layout status, Edit.

## Build order
1. `Layout` model + resolver (clip → site → unit), worker nest-model fallback off when a layout exists, layout source on results.
2. Video frame endpoint (full resolution, unrotated, any time) + generalised editor (clip / site), names on references.
3. Sites page + clip-page entry points.
4. Pipeline Detect-as-reference fixes (sampled frames, merge, stable ids) + "add to a training project".

## Decided (2026-10-22)
- Saving from a clip with a site goes to the whole site by default; "this clip only" is the other choice. A clip whose site is only a typed name gets a real site of that name (and its sibling clips join it).
- Only new runs use a changed layout. (No "re-run with this layout" button yet.)

## Built
- `videos.Layout` (site **or** clip, versioned, `save_new` skips duplicates). `devices.layouts.layout_for_video` → clip, site, unit; returns `source` / `source_label`. Device-only gates removed in `_video_job_config`, `resolve_reference` / `roi.nest_layout`, frame sampling.
- `nest_fallback=False` in the job config whenever the layout came from a site or clip → payload → `inference.py` → `CloudPipeline._fill_nests(fallback=False)`: no bee-hotel model. **Needs a new GPU image to take effect on the worker** (until then a site/clip layout with references is unaffected, since drawn references already skip the model; a ROI-only layout still gets model tubes).
- Reference names: editor **Rename**, `parse_layout_body` keeps `name` (units too), `ops.roi_references` labels by name.
- Editor shared by unit/site/clip: `templates/components/layout_toolbar.html`, `layout_canvas_script.html`, `layout_history_script.html`. Clip editor `/videos/<pk>/layout/`, site editor `/videos/sites/<pk>/layout/`, Sites page `/videos/sites/manage/` (link on Processing), clip-page card. Frames: `/videos/<pk>/frame/?t=` (ffmpeg on a presigned URL, full resolution, unrotated).
- Pipeline "Reference — Saved Layout" reads the same lookup (label updated). A sampled Detect wired as reference now yields references: distinct objects across frames, ids by position, normalised by the clip's measured size.

**Feedback:**
