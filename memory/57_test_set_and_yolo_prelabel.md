# 57 — Left-out days for evaluation; pre-label with a chosen YOLO model

**Status:** built 2026-10-24 (not yet committed). `apps/annotations/leftout.py`, `labelers.py`, `_labeler_fields.html`, migration 0014, `scripts/score_detector.py`; GPU: `inference._sample_label_batch` YOLO branch (needs a new image + image-tag bump). Export gained `?reviewed=1`. Plan v1 (Test set tab, leaderboard, eval-only GPU scoring) dropped: "we don't need new UI stuff".
Related: [[25_auto_finetuning_design]], [[38_sampling_on_sagemaker]], [[55_events_are_entries]].

## Why
Re-analysing the 2024 footage (mendels 16 Apr – 7 Jun; natalies 2 May – 7 Jun) with a new detector for the paper revision. Reviewers ask for (R2 #1, R1 #4, #10):
- standalone detector metrics (mAP50, mAP50-95, P, R) on held-out data, with the train/test split stated;
- end-to-end pipeline (event) metrics on held-out videos, every manual event in the denominator;
- a statement that training and evaluation data are disjoint.

So one set of **left-out days** serves both evaluations: nothing from them is trained on (detector here; the paper's Random Forest in its notebooks).

## The left-out days (decided)
Whole days, per site, 12 site-days, spread through each site's season and its weather (Open-Meteo, 09–19 h; to confirm against station KPABELLE49):

| site | days |
|---|---|
| mendels | 2024-04-21, 04-30, 05-08, 05-17, 05-23, 06-05 |
| natalies | 2024-05-04, 05-08, 05-17, 05-23, 05-28, 06-05 |

- 30 Apr, 8 May, 23 May hold the paper's 11 manually annotated mendels videos.
- Each site has a cold/dark day (21 Apr 7.5 °C · 4 May 12 °C), all-day rain (5 Jun), overcast drizzle (17 May), showery and sunny warm days.
- All are full recording days (62–73 real clips). Avoided: partial days; natalies 15 May (none); mendels 2–6 May (mostly empty files).

## Data facts found
- **Empty clips:** mendels 2–6 May the unit created files with no video (22, 2, 1, 1, 15 real of 62). Imported as clips, all < 1 MB.
- **Natalies night recording** until 1 May (clips 00:00–23:50); good footage from 2 May. Keep those clips out of every project.
- **Recording hours by filename:** mendels 08:55–19:10, natalies 08:01–20:00 (user remembers natalies as ~8–17; unresolved, affects the ecology, not this plan).

## Audit
- Pre-labeling is SAM 3 only at the request layer: `_labeler_opts()` returns `("sam3","")` (`annotations/views.py:41`, since d260e40). The task model, spawn and the GPU `_frame_detector` still run YOLO/custom models (`custom_model_key` → `custom_bee_model_path`); classes are matched by lower-cased name against the project's, unknown names dropped (`sagemaker_backend/inference.py:495-511`).
- Combined sample + pre-label (`sampling_remote._invoke`, `inference.py _sample_label_batch`) always uses SAM 3; `sample_label_clip(path, detect_fn, …)` is detector-agnostic. The SAM 3 endpoint runs the same image as tracking (ultralytics included).
- No split / held-out concept. `_build_training_payload` (`training/views.py:137`) takes every non-placeholder frame; train.py val = last 20% by `<title>_f<frame>`.
- `ExportProjectView` writes a YOLO dataset. Event rows (enter/exit per reference) are saved per run and exported as CSV.

## Plan

### 1. Leave out days, per project (Settings tab, one text field)
- Lines like `mendels: 2024-04-21, 2024-04-30`; a line with no site applies to all sites. Site matched by Site name, else `site_name`, case-insensitive. Bad dates / unknown sites are listed back on save.
- Clips on those days (local date as on the Clips page) get no sampling or pre-labeling in that project, and no frames from them reach a training job from that project, even if annotated.
- Each training job stores the left-out days and its training clip list in `metrics.dataset` (for the methods section).
- Sampling and pre-labeling skip clips under 1 MB (the empty files) instead of failing.

### 2. Pre-label with a chosen YOLO model (existing forms, one select)
- Labeler: SAM 3 (default) · built-in model · each `CustomModel.usable(user)` model, classes shown. `_labeler_opts` validates the pk → `labeler="yolo"`, `custom_model_key`.
- Class mapping rows (model class → project class or skip), e.g. `bee` → `insect`. Defaults: same name; else the project's only class; else skip. Refused only if all skip. Sent as `class_map`; the GPU renames before its name filter, in both paths.
- `_sample_label_batch` builds a YOLO `detect_fn` for `detector_kind="yolo"`; runs on the SAM 3 endpoint. YOLO confidence 0.25. GPU image change → push, then bump the image tag separately.

### 3. Detector evaluation (no UI)
- A separate **test project** holding only left-out-day clips; frames sampled across each day's hours, pre-labeled, then every frame checked by a person.
- `scripts/score_detector.py <weights.pt> <exported test project>` → mAP50, mAP50-95, P, R overall, per site and per day. Run once per model (built-in, new). < this already shows on the model training page>

### 4. Pipeline (event) evaluation on the left-out days — DEFERRED (user, 2026-10-24: manual event annotation comes later, no UI for it now)
- Run the full pipeline with the new detector on every clip of the 12 site-days.
- Manual events (enter/exit, tube, time) for the evaluation clips, as a CSV.
- `scripts/score_events.py <predicted events.csv> <manual events.csv>` → one-to-one matching within ±2 s at the same tube (R1 minor #6); unmatched manual events stay false negatives (R1 #4); P, R, F1 per clip, day, site and pooled.

## Tests
- Left-out days: parsing (sites, bare dates, blank lines, bad dates reported); sampling skips them; training payload drops their frames; `metrics.dataset` lists them; < 1 MB clips skipped.
- Labeler: other users' private models rejected; payload carries key + `class_map`; mapping defaults; all-skip refused; GPU rename `bee` → `insect`, skipped classes dropped.
- score_detector: per-site/day grouping from titles.

## Deferred
- Manual event annotation and `score_events.py`: later, once the manual events exist (format then). <I will do this manually later after analysis.>
