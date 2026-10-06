# 45 · Pipelines on photos

Status: **built** (2026-10-06) — canvas https://claude.ai/artifact/8mJzQDZbGeXvvWDwqZgUy4 . Needs the GPU image (task detect_photo, pillow-heif) deployed — supersedes memory/40 part 2's design (written
before photos moved into the videos table).

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

## Decisions (2026-10-06)
1. Photos from units AND any camera (upload page: JPEG/PNG/TIFF/HEIC, time
   from EXIF DateTimeOriginal → name → batch time → upload time).
2. Separate input blocks: a pipeline starts from Video Input or Photo Input;
   the editor offers only blocks that fit.
3. Tiling automatic whenever the photo is larger than the detector's input.

## Built
- Worker: `_detect_photo` (task "detect_photo"), shared `_frame_detector`
  (YOLO/SAM 3), `_read_photo` (OpenCV, else Pillow + pillow-heif);
  `beemonitor/detection/tiling.py` (1280 px tiles, 20 % overlap; merge drops a
  box mostly inside a bigger same-class box): 64 MP → 63 tiles, 12 MP → 12.
  Padded crop per insect + 1600 px preview in the processed bucket; species on
  the crops via CloudPipeline._species_classifier. Result → summary_stats.photo.
- Platform: `input.photo` block (output "photo"); detect.objects and
  reference.layout accept photo; `registry.photo_errors` refuses NEEDS_VIDEO
  blocks and mixed inputs; `pipeline_input_kind`. Engine binds photos into
  input.photo. `build_photo_detection_config`; submit_gpu_step uses
  Video.everything; spawn forwards task/classes, never chunks a photo.
  Detection count / Identify species read summary_stats.photo
  (`photo_detection_count`, `photo_species`).
- Videos → Photos: run controls (photo pipelines only), run_on_videos kind=photo
  (photos without a parent clip); schedules use photos for photo pipelines.
- Results: run page `photo_view` (boxes over the preview, crops, species);
  batch page photo summary (per photo over time, species totals) + photos CSV.
- Uploads: JPEG/PNG/TIFF/HEIC → kind=photo; EXIF DateTimeOriginal read in the
  browser (JPEG/TIFF; HEIC falls to name → batch → upload time).
