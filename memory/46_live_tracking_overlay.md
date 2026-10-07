# 46 · Live tracking overlay on the results page

Status: **built** (2026-10-07) — results page + batch clip viewer; — canvas https://claude.ai/artifact/N32sM8BV645zQAXpw2DSmr

## Ask
"Play the original video and overlay the trackings in real time in the web."
For checking tracking quality (fragmentation, id swaps) without waiting on a
GPU render. The rendered annotated video stays, on request, for downloads.

## Audit
- Results page (`analysis/results.html`) has an Original Video `<video>` from a
  presigned S3 URL, rotated with CSS when `device.rotate_180`, and an
  Annotated Video panel: a "Generate annotated video" button that sends an
  `annotate_video` GPU task (streams boxes from tracking CSV), served via
  `analysis:video_proxy` once rendered.
- Tracking CSV (worker file): `frame,track_id,x1,y1,x2,y2,cx,cy,confidence,
  source,taxon,…,mode`, original-pixel coords, one row per confirmed track per
  frame — including frames where the track is lost and its box is the last
  seen one (`time_since_update > 0` isn't in the CSV; a lost row repeats the
  previous bbox with a moving cx/cy).
  ~100k rows / 16 MB for a 10-min clip.
- Per-track species/marker: `job_tracks()` → `tracks_data` (TRACK_FIELDS).
- Events per track: computed primitives (`primitives_for_job`).
- Crops per track: `track_crops` + `_crop_viewer.html`.
- fps: `fps_with_source(summary_stats, video)`.
- Since 3afb01f the page renders only 500 rows per table; the overlay must not
  put rows in the HTML either.

## Design
- **Panel** "Video with tracks" replaces the two video panels: one player,
  overlay `<canvas>` sized to the video's box, same CSS rotation as the video
  (so boxes turn with the frames). "Download annotated video" link renders
  on request (existing GPU task), then serves it.
- **Data**: new endpoint `analysis:tracks_overlay` → compact JSON
  `{fps, width, height, tracks:{id:{species,conf,color}}, frames:{frame:[[id,x1,y1,x2,y2,lost],…]}}`,
  ints, gzip (~0.6–1 MB for 10 min). `lost` = bbox identical to the
  previous row's (the tracker's coast). Cached per result (S3 or cache).
  Fetched after the page loads; nothing in the HTML.
- **Sync**: `video.requestVideoFrameCallback` → `frame = round(mediaTime*fps)`;
  fallback `timeupdate` + rAF. Scale = displayed size / video's natural size.
- **Drawing**: box + label `id · species`; color by id; lost = dashed and
  faded with "lost N s"; trail = last 2 s of centroids; optional regions
  (hotel ROI, nests) from summary_stats.
- **Controls**: play/pause, ±1 frame (1/fps seek), speed 0.25–2×, toggles
  (boxes, species, trails, lost tracks, regions), keys space / ← → scoped to
  the player (not global). "On screen now" list.
- **Follow a track**: click a box or row → others dimmed, its trail drawn
  whole, timeline marks where it's on screen (click to jump), side panel:
  species + votes, first/last seen, on-screen time, marker, its events,
  crops (opens the crop viewer).

## Open questions
1. Replace both video panels with the one player, or keep the original
   video panel too? (Proposed: replace.)
2. Also on the batch page's clip viewer (`#cv-…`)? (Proposed: later.)
3. Chunked runs: tracking CSV is stitched — fine. Photos: n/a.

## Plan
1. Endpoint + payload builder (`apps/analysis/overlay.py`) with a test on a
   small CSV (lost detection, species join, ints).
2. `analysis/_track_overlay.html` partial + JS (vanilla, no build step),
   included on results.html in place of the video panels.
3. Follow-a-track side panel (events, crops) reusing existing data.
4. Tests (endpoint 200/404 for stranger, payload shape); stub presigning (CI).
