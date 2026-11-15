# 82 · Tube references from a detection model

Status: **built** (2026-11-15), not yet committed. Design canvas: https://claude.ai/artifact/PtktntnxpGJUyuUdeY9HuN

## The ask

> "I think I just need to solve this problem the right way, by building a nest
> tube detection model for the reference for analysis … if I change from saved
> reference to detected reference with a yolo model, how will the pipeline
> update run. Will it work with the current pipeline update system."
> "for yolo models their classes options should be a dropdown … from the labels
> of the model."

Fixing 6,306 clips' tube boxes by hand, a day at a time (memory/81), is the
problem. A trained tube model finds them per clip instead; a hand-drawn layout
stays the way to correct a day the model gets wrong.

## Audit (what is there today)

**Detected references already exist, half-way.**
- A Detect node set to *Sampled frames* and wired to an analyzer's reference
  port becomes the reference (`executors.py:225 detected_reference`,
  `:259 _sampled_references`). It runs on the GPU as the worker's
  `pre_annotate` task (`executors.py:1501 build_sampled_detection_config`):
  the whole clip is downloaded for ~20 frames.
- Its objects are numbered by position, top to bottom then left to right
  (`ops.py:1062`). One missed tube renumbers every tube after it.
- `references.counted` skips `source == "detected"` (`references.py:31`), so
  the References page can't tell such a clip is up to date.

**Tube numbers must mean the same tube on every day.** Keyed on them:
per-tube batch totals (`aggregate.py:657`), trips pairing an exit with the
next entry by `(reference, day)` (`period.py:340`), hops (`period.py:390`),
review votes (`review.py:25`, `retrack/votes.py`), AddedEvent/TripReview
`reference`. Names ("Flag C") come from the layout.

**Classes.**
- Trained models know their classes: `CustomModel.classes` = the project's
  classes cut to the job's subset, in the dataset's order
  (`training/views.py:688`). Uploaded models: typed by hand, often empty.
  Nothing reads `model.names` from a model file.
- The Detect node's class is free text (`registry.py:403`, `label`). The
  worker matches by `r.names[cls].lower()` (`sagemaker_backend/inference.py:509`),
  so a typo finds nothing, silently.
- The labelling tool already maps a model's classes with dropdowns
  (`annotations/_labeler_fields.html:21`, memory/57).
- The sampled path loads `custom_bee_model_path` only; a nest-type model goes
  into `custom_nest_model_path` and is never used there (`executors.py:1444`).
  The built-in nest model matches by class id (hotel 0, tube 1;
  `src/beemonitor/core/config.py:214`), its names unknown to the web.

**The update system** (`regroup.py`).
- `pipeline_changes` calls any added or changed GPU node a GPU change
  (`regroup.py:348`); *Update this batch* then re-runs those clips' whole GPU
  job, tracking included (`rerun_on_gpu`). Switching references to a Detect
  node would re-track all 6,306 clips on the GPU for nothing: tube boxes are
  not in what tracking uses (`engine.py:53 NOT_IN_TRACKING`), only the ROI is.
- `applied_steps` copies analyzers, MOT's web-replayable settings and Identify
  Species, not a reference node's settings (`regroup.py:137`).
- Recount re-resolves reference steps (`recount.py:83`) and re-reads a Detect
  node's saved output, so once detections are stored, recounts work.

**Gap found on the way (affects today's layout fixes too).** With MOT's
"stays at that tube" on, tracking uses the tubes (`retrack/run.py:254`,
`bee_tracker.py:540`). But `needs_retrack` compares MOT settings only
(`retrack/run.py:157`): fixing a day's tubes recounts it without re-tracking,
so its tracks still follow the old tubes.

**CPU inference on the web image.** No torch/ultralytics/onnxruntime in
`base.lock`; `opencv-python-headless 5.0` (with its DNN module) and `scipy
1.16` are. `frames.frame_jpeg` grabs one full-resolution frame with ffmpeg
from a presigned URL (no download of the clip). The recount Lambda
(memory/81) runs the web image.

## Decisions

1. **Tubes are detected on CPU, in the recount Lambda, not on the GPU.**
   ~20 frames a clip, fetched by seeking (ffmpeg on a presigned URL), run
   through the model exported to ONNX, with OpenCV's DNN module. No new image,
   no GPU-hours, no whole-clip download. The GPU `pre_annotate` path stays as
   it is for other sampled Detect uses.
2. **A detection is stored per clip, not per run** (`TubeDetection`): the same
   clip, model, class and sampling give the same tubes in every pipeline and
   batch, so it's worked out once.
3. **Numbers come from the site's saved layout** (the numbering template):
   detected tubes are matched to it. Tubes the model misses keep the
   template's box moved by the camera shift; tubes it finds that the template
   lacks get new numbers. Either flags the day.
4. **A day drawn by hand wins over the detection** (a clip's own Layout),
   default on, switchable on the node. The References page can drop a day's
   drawing to go back to the detection.
5. **The ROI stays the saved one.** Tracking uses it on the GPU; detecting it
   is a later step (only if the model has a hotel class).
6. **Changing the tube source is a web change.** *Update this batch* queues
   detect → match → re-track if needed → recount on the Lambda. No GPU.
7. **Classes are a dropdown of the chosen model's classes** everywhere a YOLO
   class is picked; free text only for a model with no known classes (and
   for SAM 3, where the text is the prompt).

## Phases

### 0 · Re-track when the tubes change (small, first)
- `retrack.run.settings_of` / `tracked_with`: when `stay_in_reference` is on,
  include the references' `geometry_key` (`references.geometry_key`); the
  re-track records it in `retracked.settings`. The GPU's tracks carry the key
  of the layout sent to the job (`config.nest_layout`).
- A run tracked before this has no key: treat as "tracked with the layout it
  counted with" (`references.counted`), so no wave of re-tracks.
- Tests: fixing a day's tubes with stay-at-tube on re-tracks; off, recounts only.

### 1 · Classes from the model (the dropdown)
- `CustomModel.classes` is the source. Built-in bee model:
  `BEEMONITOR_DEFAULT_BEE_MODEL_CLASSES`. Uploaded with none: filled from the
  ONNX file's metadata once exported (phase 2), else free text.
- Detect node: the model picker (`object_model`) lists bee, nest and custom
  models; the class field becomes a select of that model's classes, refreshed
  when the model changes (editor JS reads a `{model pk: classes}` map the
  view already builds the model list from, `pipelines/views.py:145`). A saved
  label that isn't one of the model's classes shows as "⚠ not a class of this
  model".
- Sampled Detect resolves its model from `object_model` whatever its type
  (fixes the nest-model slot gap).
- Training upload form: the classes field prefilled from the file when known.

### 2 · Detecting tubes on CPU
- **Spike first (half a day):** export the tube model to ONNX; on 3 clips,
  compare OpenCV DNN boxes with ultralytics'. Decode both heads: v8/v11
  (`1×(4+nc)×N`, needs NMS: `cv2.dnn.NMSBoxes`) and YOLO26 (end-to-end,
  `1×300×6`). Time per frame on 1 vCPU. Fallback if OpenCV can't run it:
  onnxruntime in `base.lock` (~15 MB wheel).
- **Export:** at the end of training (the training container has
  ultralytics) write `best.onnx` next to `best.pt` → `CustomModel.onnx_key`,
  and fill `classes` from its metadata when empty. Existing models: an
  `export_onnx` task on the GPU worker, one tiny job a model (GPU image
  change: push, then bump the tag separately).
- **`apps/pipelines/tubes.py`:** `detect(video, model, label, settings)`:
  frames at even times (skip the first/last 5 %), letterbox to the model's
  size, forward, decode, keep the class, cluster across frames
  (`ops.count_distinct_objects`), keep objects seen in ≥ half the frames
  (a bee sitting on a tube shouldn't hide it, a passing bee isn't a tube).
- **`TubeDetection`** (videos app): `video`, `model`, `label`, `settings_key`
  (frames, confidence, model version), `boxes` (0..1, with confidence and
  frames seen), `frames`, `detected_at`, `error`. Unique on
  (video, model, label, settings_key).

### 3 · Matching to the site's numbers
- Template: the site's current Layout (ids and names). A clip's own layout is
  never a template.
- Shift: translation + uniform scale from template to detections, by
  iterating mutual-nearest pairs and taking their median offset (cameras get
  bumped; memory/63's "Move all" is this, by hand).
- Pairing: `scipy.optimize.linear_sum_assignment` on centre distance after
  the shift, gated at half the median tube width.
- Output layout: matched → template id + name, detected box; template tube
  not found → template box moved by the shift, `kept: true`; detection not in
  the template → next free id, `new: true`. Stored on the `TubeDetection` as
  `layout` with `matched / kept / new / shift`.
- Flag the day when any tube is new, more than 10 % are kept, or the shift is
  more than one tube width (the camera moved: worth a look).
- No template (a site with no saved layout): number by position, and offer
  "Use this as the site's numbering" on the References page.

### 4 · The Reference node
- `reference.layout` gets **Tubes from**: *Saved layout* (today) | *Detected*,
  and with Detected: model (dropdown), class (dropdown of its classes),
  frames (default 20), confidence, and **A day drawn by hand wins** (on).
- `resolve_reference` with Detected: the clip's own Layout when it has one and
  the switch is on; else its `TubeDetection.layout`; ROI from the saved layout
  either way. Output `source: "detected"`, `source_label: "tube model v3"`.
- A new run: the reference step is a CPU step that waits for its detection —
  the engine parks it (as it parks a GPU step) and the Lambda's finish calls
  `advance_run`. The GPU step waits for it only when MOT uses stay-at-tube;
  otherwise they run side by side.
- `references.counted` counts detected references (their key and source).

### 5 · Updating a batch
- `pipeline_changes`: a change to the reference node's tube source or its
  settings is a web change, "Tubes: saved layout → detected (tube model v3,
  "nest tube") — no GPU-hours". `applied_steps` copies the reference node's
  settings.
- *Update this batch* → `tubes.queue(runs)`: SQS messages `{"tubes": [pk…]}`
  on the recount queue; the Lambda detects (when not stored), matches, then
  `recount_claimed` (re-tracks first if phase 0 says so). Progress is the
  References page's bar. Web fallback as for recounts (memory/81).
- Rough cost: 6,306 clips × ~20 frames; maybe 10–20 s a clip on a 2 GB
  Lambda → ~20–35 Lambda-hours, a few dollars, ~3–4 h at 8 at a time.
  **Estimate; the spike measures it.**

### 6 · References page
- Each day: **Detected** (24 of 24 matched), **Check** (1 new, 2 kept, or
  moved), **Drawn by hand**, **Waiting**. Filter *Needs a look*.
- A day's check: the frame with the detected boxes, the template's ghost,
  new and kept tubes marked; **Looks right** (marks the day checked),
  **Fix** (editor opens with the detected layout carried, as `?from=`),
  **Use the detection** (drops the day's own drawing), **Make this the site's
  numbering**.

## Open questions
- Which trained model is the tube model, and its class name (the dropdown
  will show it).
- Should a clip's own drawing win by default? Some were saved by mistake
  (memory/81); the References page lists them either way.
- Does the model detect the hotel too (ROI from detection later)?

## Order
0 → 1 → 2 (spike decides OpenCV vs onnxruntime) → 3 → 4 → 5 → 6.

## Spike (2026-11-15)

Built-in `models/nest_detection.pt` is YOLOv12m, classes `{0: hotel, 1: nest}`,
imgsz 640. Exported to ONNX (opset 17, onnxslim) and run with
opencv-python-headless 5.0.0.93 (the web lock's): on three 1280x720 mendels
clips the same tube count as ultralytics (59, 58, 58), every box matched at
IoU >= 0.95; 0.8 s a frame on an M-series laptop CPU (ultralytics/torch 0.15 s).
So no onnxruntime. Default frames a clip lowered from 20 to 8.

## Built

- **Phase 0.** `retrack.run.settings_of` adds `tubes` (a fingerprint of the
  Events step's tube boxes, `tubes_key`) when "stays at that tube" is on;
  `tracked_with` takes it from the re-track's settings, else from the GPU
  job's `config.nest_layout`; an old re-track without it counts as current (no
  wave). `recount_run` resolves the reference steps before deciding to
  re-track. `retrack_reason` -> "Tubes it stays at" on the changes card.
- **Phase 1.** `apps/training/model_classes.py`: `yolo_models(user)` (built-in
  bee "", built-in nest `builtin:nest` -> `v1/nest_detection.pt` with
  `["hotel", "nest"]`, then usable models with `CustomModel.classes`).
  Editor field types `yolo_model` and `model_class` (`model_field`,
  `prefer`, `text_when`): a plain dropdown of the model's own classes, nothing
  else (the user's call, 2026-11-15: "I just want to see the labels for that
  model as a drop down"). A saved name the model lacks is replaced by its
  class quietly; text only for SAM 3 and unknown classes.
  Detect's `object_model` can be the built-in nest model.
- **Phase 2.** `apps/pipelines/tubes.py`: `_load` (ONNX + `.onnx.json` from
  the models bucket, cached in /tmp), `boxes_in` (v8/v11/v12 head with NMS,
  and end-to-end heads), `objects` (cluster across frames, seen on >= half),
  `detect` -> `videos.TubeDetection` (unique per clip, model key, class,
  settings; errors retried after 15 min). ONNX export: `train.py
  _export_onnx` after best.pt (best effort; onnx + onnxslim in
  requirements.train.txt), `scripts/export_onnx.py` for existing ones.
- **Phase 3.** `tubes.match`: commonest template->detected offsets on a
  half-tube grid, each refined with a least-squares shift + zoom and
  Hungarian pairing (scipy), the one pairing most kept; pairs within half a
  tube; missed -> template box moved (`kept`), extra -> next ids (`new`);
  flag new / kept > 10 % / moved > 1 tube or zoom > 10 % / no_template /
  none_found. 60 tubes in ~9 ms. Only tubes whose centre is in the ROI.
- **Phase 4.** `reference.layout` (now "Reference — Tubes and ROI"): source
  `detected`, `tube_model` (default built-in nest), `tube_class`,
  `tube_frames` (8), `tube_confidence` (0.4), `drawn_wins` (on).
  `tubes.layout_for` -> `{hotel_roi (saved), nest_layout, source: detected,
  tubes: {state: detected|drawn|pending|error, matched, kept, new, shift,
  scale, flag, template_key, marks, detection}}`. Detection runs only under
  `recount.DETECTING` (set by `recount_run`); a fresh run gets `pending` and
  is queued on completion (`engine._finalize_if_terminal`, on_commit).
- **Phase 5.** `applied_steps` copies the reference node's config;
  `apply_pipeline` and `recount_clips` never recount a detecting run inline
  (`recount.detects_tubes`), and queue in one send. `PER_MESSAGE` 20 -> 5.
  Lambda 3072 MB, 2 GB /tmp.
- **Phase 6.** `references.days`: `kind` detected / look / waiting / hand /
  saved (+ checked from `pipelines.TubeDayCheck`), `worst`, `flag_key`,
  stale when a drawing appeared or the site's numbering changed
  (`current_for` `shared_key`). Day page `batch_references_day` (frame, marks,
  template ghost, clips to look at); actions `checked`, `numbering` (saves
  the run's tubes as the site layout and queues the place's stale runs).
  Layout editor `?tubes_of=<run>` opens with a run's detected tubes.
- Tests: `test_detected_tubes.py` (23), 4 in `test_retrack.py`; jsdom check
  of the editor picker. Full suite 1,942 OK.

## To go live
1. Push (CI deploys the web and the worker; the GPU/training images rebuild
   because sagemaker_backend/ changed: no tag bump needed, inference unchanged).
2. `pulumi up` in infra/aws (queue + Lambda from memory/81, now 3 GB).
   Without the queue, the web's sweep would detect on its one vCPU.
3. `scripts/export_onnx.py --bucket beemonitor-dev-models-<account>` once
   (needs `pip install onnx onnxslim`), for the built-in models and models
   trained before.
4. Pipeline: Reference -> Source "Tubes found by a model", model + class;
   batch: Update this batch. Then References -> To look at.
