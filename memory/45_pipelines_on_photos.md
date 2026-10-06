# 45 · Pipelines on photos

Status: **plan** (2026-10-06) — supersedes memory/40 part 2's design (written
before photos moved into the videos table). Awaiting decisions, then a canvas.

## Ask
"Pipelines should run on photos too" — e.g. count insects in an image with
Detect alone, or name the species of each insect in a photo.

## Audit (what changed since memory/40 part 2)
- Photos are `Video` rows with `kind="photo"` (`Video.everything`; clips-only
  default manager). So a run, a GPU `Job` and the StepResult cache can key on
  a photo exactly as on a clip — no nullable `Job.video`, no stills M2M.
- Units already upload photos: a 5-photo burst before each clip (64 MP
  OwlSight / 12 MP OAK) and optional periodic photos. They show under
  Videos → Photos. Nothing can analyse them yet.
- `Video.objects` (clips only) is what `run_on_videos`, `launch_batch`,
  schedules and the hub's run controls use, so photos are excluded everywhere
  by construction; the Photos tab has no run controls.
- Species by crop vote exists (memory/42: BeeMachine / BioCLIP over crops).
  On a photo each detection is its own crop → one call per insect.
- Detection on 64 MP: YOLO at 640 px shrinks a ~60 px bee to ~4 px; SAM 3
  resizes to ~1K. Needs **tiling** (~1280 px tiles, ~20 % overlap, merge with
  NMS): ~40–60 tiles per 64 MP photo, ~8 for 12 MP. Decoded 64 MP ≈ 190 MB.

## Design
- **Input block** `input.photo` (beside `input.video`): the run's photo.
  Valid downstream: Detect (forced "this image", tiled), Reference (drawn
  regions), Analyze → Detection count (total / per class / per region),
  Identity → species. Tracking, events, interactions, marker and
  colony-activity need motion and are refused after a photo input
  (validation message says why).
- **GPU task `detect_photo`**: download, tile, YOLO or SAM 3 per tile, merge,
  save a padded crop per detection (+ a 1280 px overlay preview); species
  vote reuses `track_vote` with each detection as its own track. Returns
  `detections:[{box, class, conf, species?, crop_key}]`.
- **Running**: Videos → Photos gets the same select + Run pipeline controls
  (pipelines whose input is a photo). One run per photo, batched like clips;
  schedules gain "new photos".
- **Results**: per photo — the image with boxes, count per class, species per
  detection, CSV; per batch — counts per photo over time (taken_at), species
  totals, combined CSV.
- **Uploads** (memory/44 page): also accept JPEG/PNG/TIFF/HEIC photos from any
  camera, time from EXIF DateTimeOriginal → name → batch time → upload.

## Open questions
1. Photos from any camera (upload page) too, or unit photos only for now?
2. One pipeline for both (input block switches clip/photo), or separate
   photo pipelines?
3. Tiling automatic above a size, or a setting on Detect?
