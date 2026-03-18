# BeeMonitor Phase 2 — Implementation Plan

## Overview

Three major features built in 6 stages. Each stage is self-contained, testable, and builds on the previous.

---

## STAGE 11: Job Configuration System

**Objective:** Give users control over analysis settings, prevent duplicate analysis, show cost estimates.

**Deliverables:**

### 11a. Analysis Config Model + UI
- Add `AnalysisConfig` model or extend Job.config with structured fields:
  - `two_mode_tracking`: bool (default True)
  - `confidence_threshold`: float (default 0.25)
  - `ml_threshold`: float (default 0.6)
  - `detection_mode`: choice (yolo)
  - `visualize`: bool (default True, generates annotated video)
  - `gpu_tier`: choice (T4/L4/A10G, default A10G)
- Config presets: "Fast" (T4, no viz, 0.3 confidence), "Standard" (A10G, viz, 0.25), "Quality" (A10G, viz, 0.2)
- Config panel on Videos page: expandable settings before "Analyze Selected"
- Cost estimate: show "Estimated cost: ~$X.XX per video, ~$XX total for N videos"

### 11b. Deduplication System
- Compute `config_hash` = SHA256(video_id + sorted config JSON)
- Store `config_hash` on Job model
- Before spawning: check if completed Job exists with same hash
- Batch submission: report "N new, M already analyzed, skipping M"
- User can force re-analysis with "Re-analyze" checkbox

### 11c. GPU Tier Selection on Modal
- Update Modal app.py: separate functions per GPU tier, or parameterize
- Pass GPU tier from Django → Modal
- Track actual GPU-seconds in Job model (returned from Modal)

**Files to create/modify:**
```
beemonitor_web/
├── apps/analysis/
│   ├── models.py          — Add config_hash field, gpu_tier
│   ├── migrations/         — New migration
│   ├── views.py           — Update BatchJobView with config panel + dedup
│   └── templates/
│       └── analysis/config_panel.html  — Reusable config form component
├── apps/videos/
│   └── templates/videos/list.html     — Add config panel before analyze
cloud/
└── modal_app/app.py       — Parameterize GPU tier
```

**Validation:**
- Submit same video twice with same config → second skipped
- Submit same video with different config → both process
- Cost estimate matches actual Modal bill
- T4 job costs less than A10G job

---

## STAGE 12: User Account & Profile System

**Objective:** User profiles with organization info, tier management, and usage tracking foundation.

**Deliverables:**
- Extend `UserProfile` model:
  - `tier`: free / standard / pro / enterprise (already exists)
  - `organization`: CharField (already exists)
  - `max_concurrent_jobs`: int (default 10 for free)
  - `monthly_credit_cents`: int (default 3000 = $30 for free)
  - `used_credit_cents`: int (resets monthly)
  - `credit_reset_date`: DateField
- Profile settings page: edit organization, view tier, see usage
- Admin panel: manage user tiers, allocate credits
- Usage tracking: record GPU-seconds per job, convert to cents
- Monthly reset cron (management command)

**Files:**
```
beemonitor_web/apps/accounts/
├── models.py              — Extend UserProfile with quota fields
├── migrations/
├── views.py               — Profile page, usage dashboard
├── templates/accounts/
│   ├── profile.html       — Edit profile
│   └── usage.html         — Usage charts and credit balance
├── management/commands/
│   └── reset_monthly_credits.py
```

**Validation:**
- Free user sees $30 credit balance
- After job completes, credit balance decreases
- Admin can change user tier
- Monthly reset restores credits

---

## STAGE 13: Quota Enforcement & Cost Tracking

**Objective:** Block job submission when quota exceeded, track real costs, show usage analytics.

**Deliverables:**

### 13a. Cost Tracking
- Modal returns actual execution_seconds in result
- Store `execution_seconds` and `compute_cost_cents` on Job model
- GPU tier cost rates stored in config: T4=$0.000164/s, A10G=$0.000306/s, etc.
- After job completes, calculate: cost = execution_seconds × rate

### 13b. Quota Enforcement
- Before job submission: check `used_credit_cents + estimated_cost < monthly_credit_cents`
- If exceeded: show "Quota exceeded. Used $X of $Y this month. Upgrade or wait for reset."
- Concurrent job limit: check active processing jobs < max_concurrent_jobs
- Batch submission: estimate total cost, warn if it would exceed quota

### 13c. Usage Dashboard
- `/accounts/usage/` page with:
  - Credit balance bar (used/remaining)
  - Cost breakdown by day (Chart.js bar chart)
  - Cost breakdown by site
  - Top videos by cost
  - Projected monthly spend
- Usage data in admin panel

**Files:**
```
beemonitor_web/apps/accounts/
├── models.py              — Add compute_cost tracking
├── views.py               — Usage dashboard view
├── templates/accounts/usage.html
apps/analysis/
├── models.py              — Add execution_seconds, compute_cost_cents
├── views.py               — Quota check before submission
├── migrations/
cloud/modal_app/app.py     — Return execution time in results
```

**Validation:**
- Free user blocked after $30 spent
- Cost shown per job in analysis list
- Usage dashboard shows accurate spend
- Admin can increase user credits

---

## STAGE 14: Video Annotation Interface

**Objective:** Let users annotate video frames for training custom YOLO models.

**Deliverables:**
- Annotation UI: draw bounding boxes on video frames
  - Frame navigator: step through video frames
  - Bounding box tool: draw, resize, delete boxes
  - Class labels: configurable (bee, wasp, nest, etc.)
  - Export: YOLO format (txt files with normalized coordinates)
- Annotation storage:
  - `Annotation` model: video, frame_number, boxes (JSONField)
  - `AnnotationProject` model: name, classes, user, videos
- Frame extraction: extract frames from video on demand (Azure or Modal)
- Import: support uploading existing YOLO annotations (zip of txt + images)

**Files:**
```
beemonitor_web/apps/annotations/          — New Django app
├── __init__.py
├── apps.py
├── models.py              — AnnotationProject, Annotation
├── views.py               — Annotation editor, frame extractor
├── urls.py
├── forms.py
├── templates/annotations/
│   ├── editor.html        — Canvas-based annotation UI
│   ├── project_list.html
│   └── project_detail.html
├── static/js/
│   └── annotator.js       — Bounding box drawing on canvas
├── migrations/
```

**Validation:**
- User creates project, adds videos, annotates frames
- Bounding boxes saved to DB in YOLO format
- Export produces valid YOLO training dataset (images/ + labels/)
- Import existing annotations works

---

## STAGE 15: YOLO Fine-Tuning Pipeline

**Objective:** Train custom YOLO models on user annotations using Modal GPUs.

**Deliverables:**

### 15a. Training Job System
- `TrainingJob` model: user, project, base_model, epochs, status, metrics
- Training config: base model (yolov8n/s/m, yolo11n/s, yolo26), epochs, image size, batch size
- Submit training job → Modal GPU function
- Modal function:
  1. Download annotations from Azure
  2. Prepare YOLO dataset (train/val split)
  3. Run `model.train()` via ultralytics
  4. Upload trained weights (.pt) to Azure
  5. Return metrics (mAP, precision, recall, loss curves)

### 15b. Modal Training Function
```python
@app.function(gpu="A10G", timeout=14400)  # 4 hours
def train_yolo_model(
    project_id, user_id, base_model, epochs, imgsz, batch_size,
    dataset_azure_path, output_azure_path
) -> dict:
    from ultralytics import YOLO
    model = YOLO(base_model)
    results = model.train(data=dataset_yaml, epochs=epochs, imgsz=imgsz, batch=batch_size)
    # Upload weights + metrics to Azure
    return {"model_path": ..., "metrics": ...}
```

### 15c. Custom Model Management
- `CustomModel` model: user, name, model_path (Azure), base_model, metrics, created_at
- Model list page: view trained models with metrics
- Select custom model when submitting analysis jobs
- Model comparison: run same video with default vs custom model

**Files:**
```
beemonitor_web/apps/training/             — New Django app
├── models.py              — TrainingJob, CustomModel
├── views.py               — Submit training, view models, metrics
├── urls.py
├── templates/training/
│   ├── list.html          — Training jobs
│   ├── new.html           — Submit training job
│   ├── models.html        — Custom models list
│   └── model_detail.html  — Metrics, loss curves
cloud/modal_app/app.py     — train_yolo_model function
cloud/wrapper/training_pipeline.py  — Prepare dataset, run training
```

**Validation:**
- User annotates 50 frames → submits training → model trained in ~30min
- Trained model appears in model list with mAP score
- User selects custom model → analysis uses it instead of default
- Custom model produces better results on user's specific bee species

---

## STAGE 16: Integration & Polish

**Objective:** Connect all Phase 2 features, polish UX, add monitoring.

**Deliverables:**
- Config panel integrated with quota display ("$X.XX remaining")
- Custom model selector in config panel
- Training page linked from main nav
- Annotation editor linked from video detail page
- Email notifications: job complete, training complete, quota warning
- Error recovery: retry failed jobs, resume interrupted training
- API endpoints for all new features (DRF)
- Mobile app updates for new features

**Files:** Various updates across all apps.

---

## Dependency Graph

```
Stage 11: Job Config + Dedup
    ↓
Stage 12: User Profiles + Tiers
    ↓
Stage 13: Quota Enforcement + Cost Tracking
    ↓ (parallel tracks below)
Stage 14: Annotation Interface ──→ Stage 15: YOLO Training Pipeline
    ↓                                    ↓
                Stage 16: Integration & Polish
```

Stages 14-15 can start in parallel with Stage 13 but depend on Stage 11 for config system.

---

## Estimated Effort

| Stage | Scope | Est. Days |
|-------|-------|-----------|
| 11 | Job Config + Dedup | 2-3 |
| 12 | User Profiles + Tiers | 2-3 |
| 13 | Quota + Cost Tracking | 3-4 |
| 14 | Annotation Interface | 5-7 |
| 15 | YOLO Training Pipeline | 4-5 |
| 16 | Integration + Polish | 3-4 |
| **Total** | | **19-26 days** |

---

## Key Design Decisions

1. **Config hash for deduplication** — SHA256 of video_id + config prevents re-analysis waste
2. **Credit system in cents** — avoids float precision issues, 1 cent = smallest unit
3. **GPU tier as user choice** — let researchers trade cost vs speed
4. **YOLO training via ultralytics** — same library used in core BeeMonitor, consistent API
5. **Annotations stored as JSON** — flexible, no need for separate annotation DB
6. **Custom models in Azure Blob** — same storage as everything else, linked to user account
