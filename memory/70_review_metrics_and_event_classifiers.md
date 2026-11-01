# 70 — Review metrics by action; training and evaluating event classifiers from reviews

**Status:** built (2026-10-31) with the proposed answers. Design canvas: https://claude.ai/artifact/1u4xnddRrfLxZsC9DERFAo (boards "Batch page: which actions the review metrics count", "Training › Event classifiers").

> Write under any **Feedback:** line. *Proposed* means a starting point, not a decision.

## What the user asked (2026-10-31)
- **Metrics:** count only exits and entries in the review metrics (what foraging needs). Stays are still reviewed, just left out of the numbers, and it's configurable.
- **Later:** train a classifier that also scores stays, apply it, and see the new metrics.
- **Plan for:** using the review data to **train and evaluate** event classifiers.

## What exists (audit, 2026-10-31)
- **The scorer** (`apps/pipelines/ml_scoring.py`):
  - The Events node's "Score events with the ML classifier" adds `ml_confidence` to exits and entries only.
  - The model is `models/event_classifier_model.pkl`: a scikit-learn random forest over 20 trajectory features of a track relative to a reference, in pixels and frames (`trajectory_features`, copied from the worker's `EventProcessor`).
  - Trained outside the app, for the paper, on hand-annotated videos.
  - Nothing is dropped; the reader picks a cut-off.
- **Reviews** (memory/62):
  - `EventReview`: verdict accepted or rejected per user, with action, reference, track and frame.
  - `AddedEvent`: missed events, with action, reference and frame; the track is often blank.
  - `TripReview`.
  - Votes follow their events across recounts and re-tracks (memory/69).
  - `review.batch_review_summary` gives precision = accepted / (accepted + rejected) and recall = accepted / (accepted + added), over every action.
- **Tracks:** each run's tracking table (re-tracked or the GPU's) gives a reviewed event's trajectory, so its features can be computed now, on the tracks the reviewer saw.
- **Training app** (`apps/training`): detection models only. `CustomModel` has `model_type` nest_detection / bee_tracking / custom, `storage_key`, `metrics` and publishing.
- **Held-out data:** the left-out days (memory/57: 12 site-days of mendels and natalies) are kept out of every training set, for the paper's evaluation.

## The plan

### 1. Which actions the metrics count (now)
- **Setting:** the Events review card on the batch page gets **Count in metrics: Exits ✓ Entries ✓ Stays ☐**. It's saved on the batch (`Batch.review_actions`, so invited members see the same numbers). A small migration; the `Batch` row is created on first save.
- **What follows it:** "N of M detected events reviewed", precision and recall (added missed events included only if their action is chosen), and the per-clip review counts. The card says "Exits and entries only (stays left out)".
- **Unchanged:** reviewing (stays are still reviewed), the CSVs (every event and every vote) and trip metrics.
- **Defaults:** this batch: exits and entries. New batches: *proposed* all three, as today.
- **Scored subsets:** when an Events step scores events, the card also shows precision and recall for **events at or above a confidence cut-off** (the cut-off is a slider) next to all events, so a classifier's effect is visible on reviewed data.

### 2. A training set from reviews
- **One row per reviewed detected event:**
  - the 20 trajectory features plus the stay features (§4), computed from the run's current tracks and that run's reference;
  - the label: accepted = 1, rejected = 0;
  - action, site/unit, day, clip, batch and reviewer.
- **Disputed events** (reviewers disagree) are left out and counted.
- **Missed events (`AddedEvent`)** can't train a scorer of detected events (nothing was detected to score), but they count in the **recall** of every evaluation.
- **Built from chosen batches** on a new **Event classifiers** page (Training › Event classifiers):
  - a summary of what's there: reviewed events by action, site and day, and the share accepted;
  - a download as CSV (features + labels), for notebooks and the paper.

### 3. Splitting without leakage
- **Never split by row:** events from one clip share bees, light and layout.
- *Proposed:* the **left-out days** are the test set, always (memory/57). Training and validation use the other days, with **grouped cross-validation by day**.
- A day with no reviews is reported, not silently skipped.
- The page states the split: days, clips and events per side.

### 4. Training
- **One model per kind of event:**
  - **exit/enter**: the same 20 features as now, so it's comparable with the paper's model;
  - **stay**: its own features: duration, dwell share, movement and spread inside the reference, distance to its centre, entry and leave speeds, body length.

  *Proposed* as two models, because the features of a stay differ from those of an exit or entry.
- **Where it runs:** scikit-learn random forest (the same family as now; gradient boosting as an option), on the web in the background. The training set is thousands of rows, so it takes seconds; no GPU.
- **What's stored:** a `CustomModel` with `model_type="event_classifier"`:
  - the joblib file in S3;
  - feature names and the actions it scores;
  - the training batches and days, counts per class;
  - its metrics.

  Publishing works as for detectors.

### 5. Evaluating
On the held-out days, for each model and for the current built-in one:
- **Per event, by action:** ROC-AUC, precision, recall and F1 at a chosen cut-off; a precision–recall curve; calibration (are 0.8s right 80% of the time).
- **For the pipeline:** if events below the cut-off were dropped, what precision and recall become against the reviews, with missed events in recall. These are the numbers the paper needs ("every manual event in the denominator", memory/57).
- **A comparison table:** built-in model vs each trained one, same test set.

### 6. Applying a model
- **Events node:**
  - "Score events with" becomes **Off / Built-in (paper) / <your models>**;
  - an optional **Leave out events scored below** with a cut-off. Off by default: scores are a column, the reader decides (memory/60).
- **Applying it:** **Update this batch** applies it in place, with no GPU (memory/68, 69). The metrics card then shows the new numbers, which is the "apply and see the new metrics" loop.
- **Stays:** scored only by a model that covers stays; until then their `ml_confidence` stays blank and the card says why.

## Build order
1. **Metrics by action:** the batch setting, the card, summaries and tests. *Can ship first, on its own.*
2. **Training set:** feature extraction for reviewed events (exit/enter and stay), the page's summary and CSV download.
3. **Training:** left-out-day split, grouped CV, background training, storing the model.
4. **Evaluation:** per-event and pipeline metrics, comparison table.
5. **Applying:** the Events node's model choice and optional cut-off; recount via Update this batch.

## Questions
1. **Metrics default for new batches:** all three actions (proposed), or exits and entries?
2. **Test set:** always the left-out days (proposed), or chosen per training?
3. **Models:** separate exit/enter and stay models (proposed), or one model with the action as a feature?
4. **Training:** in the app (proposed), or only the CSV export for notebooks?

**Feedback:**

## Built (2026-10-31)
- **Metrics by action:**
  - `Batch.review_actions` (migration 0019; empty = all); `batches.review_actions` / `set_review_actions`; `pipelines:batch_review_actions` (launcher only).
  - `review.batch_review_counts(runs, actions)` and `batch_review_summary(..., actions)` count only those actions, missed events included. The card keeps showing when a filter is set.
  - The card has **Count in metrics** toggles, a "… only — … left out" note, and the **cut-off panel**: scored reviewed events, computed in the browser.
  - `review.review_votes(runs)` is factored out for reuse.
- **Scoring** (`ml_scoring`):
  - `track_paths` (boxes, or centres only), `ref_boxes`, `stay_features` / `STAY_FEATURES`, `features_for`;
  - `load("on" | "model:<id>")`;
  - `score(rows, …, spec, stay_spec)`.
  - The Events node has "Score exits and entries with" (Off / Built-in / your models), "Score stays with" and "Leave out events scored below" (`ml_min_confidence`, blank = keep all). These are `event_model` fields in the editor (`EVENT_MODELS`).
- **Training** (`apps/pipelines/eventml.py`):
  - `dataset()`: one row per reviewed detected event on its run's current tracks; disputed events left out; missed events kept for recall.
  - `train()`: left-out days are the test set; grouped-by-day CV for validation; random forest (balanced) or gradient boosting.
  - `evaluate()`: AUC; P/R/F1 at 0.5; pipeline P/R with missed events; calibration bins.
  - `builtin_report`.
  - `train_and_store` runs in a background thread. The model is stored as joblib in the processed bucket under `event_models/…`, as a `CustomModel(model_type="event_classifier", classes=actions)` (training migration 0015).
- **Page:** **Training › Event classifiers** (`training:event_classifiers`):
  - batches with reviews; left-out days (defaulting to the user's projects');
  - Show the data; Download CSV;
  - Train (exit/enter, stay; algorithm);
  - a comparison table with the built-in model, plus calibration.
- **Tests:** `test_review_metrics.py` (5), `test_eventml.py` (7).

