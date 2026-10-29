# 61 — Events and interactions on the video; the rendered video matches the player

**Status:** built 2026-10-29 (not committed). Decided: chip windows 1.5 s; clock offered as an off-by-default render layer. Also built: the toggle is "ROI and reference objects" (the pipeline's ROI drawn too), and full window keeps the overlay (the stage goes full window; the video's own fullscreen is off).
Canvas: https://claude.ai/artifact/E7PjdMMJCNtbimZ6P5Vy7K
Related: [[46_live_tracking_overlay]], [[60_activity_extraction_audit]].

## Why
- The user wants to see events and interactions drawn on the video, not only listed.
- A rendered video ("Render video" on the results page) looks different from the player overlay.

## Audit
- **Player** (`static/js/track_overlay.js`). It draws boxes per frame from the stored payload (`analysis/overlay.py`, VERSION 3: boxes only) and the ROI and references served live by `?part=events`, with these layers:
  - boxes, species labels, 2 s trails, lost positions dashed;
  - colour per track by golden-angle hue: `hsl(id*137.508, 85%, 58%)`.

  Events appear only as a per-track list in the side panel, and interactions not at all.
- **Rendered video** (`sagemaker_backend/inference.py` `_annotate_video`, started by `GenerateAnnotatedVideoView`). It shares nothing with the player:
  - colours come from 5 fixed colours picked by Python's `hash()`, which changes between runs, so a track's colour isn't even stable;
  - it draws the worker's `nest_bboxes`/`hotel_bbox`, not the pipeline's references;
  - its label is the raw detector `taxon`, not the voted species;
  - no lost/dashed state, no trails, no events, no interactions;
  - it ignores the unit's 180° mount (the player rotates);
  - it writes `mp4v`, which browsers won't play; the video proxy re-encodes it.

## Plan

**1. Events and interactions on the player** (`track_overlay.js`, `overlay_events`)
- `?part=events` also returns interactions as `[start_frame, end_frame, track_id, with, relation, min_distance_bl]`, from the run's saved rows (no recount).
- Two new layers, both on by default: **Events** and **Interactions**. Drawing follows the Legend artboard:
  - **exit:** a chip "EXIT 3" under the insect from the event frame for 1.5 s; the reference glows blue;
  - **enter:** the chip for the 1.5 s before the event (the insect is gone after it); amber glow;
  - **stay:** the chip for the whole stay; violet glow;
  - ` · at edge` is added when flagged;
  - **in a reference:** while the spell lasts, the reference is drawn solid in the insect's colour, and its label adds `in 5 · 2.3 s`;
  - **near:** a line between the two insects with `near 1.4 bl` while the spell lasts.
- **Timeline marks** for events across the whole clip (exit blue, enter amber, stay as a violet span); click one to seek.
- The side panel lists the events near the current time; click to seek.

**2. One drawing spec, two renderers** (the rendered video looks like the player)
- **The spec** — a module (`analysis/overlay_spec.py`) — defines:
  - the colour function;
  - box, label, trail, lost, ROI, reference, event and interaction styles, sizes and timings;
  - the label text.

  The JS reads the same constants from the payload (`payload.style`), so there is one source for both.
- **Render:** the web builds a full render payload, uploads it as JSON, and the GPU task draws from it. Nothing is re-derived on the worker, so it can't disagree with the player. The payload holds:
  - the stored boxes;
  - track identities;
  - the live events, interactions, ROI and references;
  - the layers on when Render was pressed;
  - `rotate_180`;
  - fps.
- **Worker** `_annotate_video`: a port of the canvas drawing (OpenCV), same geometry and timings, 180° when the unit is mounted upside down. Encoding becomes H.264 via ffmpeg (already installed in the GPU image), so it plays in the browser without the proxy re-encode.
- A parity test: draw frame N of a fixture with the Python renderer and check colours and box positions against the spec. The JS uses the same spec constants (a test asserts the JSON the page gets equals the module).
- The Render dialog: layer checkboxes (prefilled from the player), cost in GPU-hours, then Render.

**3. Not now:** burning a clock into the video (offered as an off-by-default layer in the dialog; build only if wanted).

## Questions
1. Chip timing: exit shown for 1.5 s after the event, enter for 1.5 s before it. OK, or a different window?
2. Should the rendered video include the clock layer (clip time + timestamp) as an option?

## Build notes (2026-10-29)
- `apps/analysis/overlay_spec.py`: `STYLE` (every colour, size and timing), `LAYERS`, `track_rgb`, `layers_from`. Its docstring is the drawing contract both renderers follow.
- `?part=events` (`views.overlay_live`) returns events, interactions, regions (ROI + references) and `style`; the JS replaces its built-in style with it.
- Render: `views.render_spec(job, layers)` = the stored boxes + the live part + layers + `rotate_180` + `clip_start`, gzip'd to `processed/<user>/<annot_id>/render_spec.json.gz`. The GPU payload carries only `render_spec_path` (no tracking CSV, no worker nest boxes).
- GPU: `sagemaker_backend/render_overlay.py` (pure drawing, tested against the web's real `STYLE`), H.264 through ffmpeg (falls back to mp4v with a warning), text drawn with DejaVu Sans Mono sprites so "·" and accented names match the player.
- The render dialog prefills its layers from the player; unticked = off (the form sends `layers`).
