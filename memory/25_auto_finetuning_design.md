# BeeMonitor — Auto Fine-Tuning for Bee Detection (SAM 3 + DINOv3 → YOLO)

**Status:** Plan for review (drafted 2026-07-03)
**Author:** Drafted with Claude Code (Fable 5), 2026-07-03
**Problem:** This year's videos show no annotated output / no results — the bee YOLO
likely **doesn't generalise to the new domain** (new site/camera/lighting/species).
**Goal:** An **auto fine-tuning loop** that detects domain shift, auto-labels the
failing footage with a domain-robust foundation model (**SAM 3**), selects the most
useful frames (**DINOv3**), fine-tunes the YOLO detector on BeeMonitor's *existing*
SageMaker training pipeline, evaluates, and auto-promotes the improved model — so the
system adapts itself instead of silently producing nothing.

Related: [[23_pipeline_builder_port_design]] (the run pipeline), `apps/training`
(TrainingJob → SageMaker → CustomModel), `apps/annotations` (labels + YOLO export +
AI pre-annotation), `sagemaker_backend/train.py` (the YOLO training container),
EcoMorph (`ecomorph/segmentation/sam3.py`, a working SAM 3 integration to mirror).

---

## 0. Diagnose first (do before building anything)

"No results this year" is *probably* domain shift, but confirm it's the **model**,
not config or the endpoint (which was broken/rolling until ~2026-07-03 and only just
got the ROI-fix image):

1. **Re-run 2–3 of this year's clips now** on the recovered tracking endpoint. If they
   still yield ~0 detections / 0 tracks → not an endpoint/config problem.
2. **Spot-check with SAM 3**: run SAM 3 with prompt `"bee"` on a couple of frames from
   those clips. If **SAM 3 finds bees where the YOLO finds none**, that's the smoking
   gun — and it directly validates this whole design (SAM 3 becomes the labeler).

Only proceed if (1)+(2) confirm domain shift.

---

## 1. Key insight — BeeMonitor already has the training "back half"

This is mostly wiring a *front half* onto infrastructure that exists:

| Already built | File |
|---|---|
| Training job → SageMaker `create_training_job` → poll → `CustomModel` | `apps/training/views.py` `_spawn_training_job`, `PollTrainingJobsView` |
| YOLO training container (ultralytics, builds dataset, uploads `best.pt`) | `sagemaker_backend/train.py` |
| Labels: `AnnotationProject` / `Annotation.boxes` + `to_yolo_format` | `apps/annotations/models.py` |
| **AI pre-annotation** (samples frames, runs a model, writes boxes) | `apps/annotations/views.py` `PreAnnotateView` |
| YOLO dataset export (images/ + labels/ + data.yaml) | `apps/annotations/views.py` `ExportProjectView` |
| Custom-model swap into inference (`custom_bee_model_path`) | `cloud/wrapper/pipeline.py`, model selection in `apps/analysis` |
| Serverless foundation-model endpoint pattern (BioCLIP) | `infra/aws-sagemaker` (separate ECR + endpoint) |

**The gap is only:** a domain-robust *labeler*, smart *frame selection*, and the
*closed loop*. Today's `PreAnnotateView` runs the **same failing YOLO** as the labeler —
so it can't bootstrap the very videos that break. Swapping **SAM 3** in as the labeler
is the crux.

---

## 2. The two foundation models (and why each)

- **SAM 3 — the labeler.** Meta's SAM 3 (Nov 2025; SAM 3.1 Mar 2026) does *Promptable
  Concept Segmentation*: a **text prompt** ("bee") or image exemplar → it detects,
  segments, and tracks **every instance** in images/video, open-vocabulary, ~2× prior
  accuracy. It's domain-agnostic, so it labels the new-domain footage the current YOLO
  misses. EcoMorph already runs it (`SAM3Segmenter("facebook/sam3").segment(img,
  "bee", threshold) → [Detection{mask, score, bbox, area_px}]`) — we mirror that. Masks
  → bounding boxes = YOLO labels. ([SAM 3](https://ai.meta.com/blog/segment-anything-model-3/),
  [facebookresearch/sam3](https://github.com/facebookresearch/sam3),
  [Ultralytics SAM 3](https://docs.ultralytics.com/models/sam-3))
- **DINOv3 — the selector/verifier.** Meta's DINOv3 (2025; 7B params / 1.7B images) is
  the SOTA self-supervised backbone; its embeddings are ideal for **active learning**:
  (a) detect domain shift (this-year frames cluster away from the training set → the
  auto-trigger), (b) pick a diverse, non-redundant, representative subset to label +
  train (don't label 10k near-duplicates), (c) mine hard frames (YOLO↔SAM 3 disagree),
  (d) sanity-check SAM 3 labels by embedding consistency. It's *not* a detector — a
  feature extractor. ([DINOv3](https://www.lightly.ai/blog/dinov3))

**The paradigm = distillation.** A big, slow, expensive zero-shot model (SAM 3) teaches
a small, fast, cheap real-time model (YOLO). SAM 3 runs only at *label* time (bounded
cost); YOLO handles the millions of inference frames. This is exactly Ultralytics'
`auto_annotate` / Autodistill pattern. ([auto_annotate](https://docs.ultralytics.com/reference/data/annotator))

---

## 3. The loop

```
 (trigger) low-detection / domain-shifted batch
      │  DINOv3: this-year frames far from training distribution
      ▼
 sample frames  ──DINOv3──►  select diverse + hard subset (dedup, cover domain)
      │
      ▼
 SAM 3 auto-label  (prompt "bee"/"wasp"; masks → YOLO boxes)  ──►  Annotation rows
      │
      ▼
 (optional) 1-click human verify in the existing annotation editor
      │
      ▼
 YOLO dataset export ──►  SageMaker training (existing train.py) ──►  CustomModel best.pt
      │
      ▼
 evaluate on a held-out verified test set (mAP)  ──►  auto-promote if it beats current
      │                                                 (regression guardrail)
      ▼
 set CustomModel active for that site/device  ──►  re-run the affected videos
```

Each arrow is an existing component or a thin addition. The only genuinely new compute
is the **SAM 3 endpoint** and **DINOv3 embeddings**.

---

## 4. What to build (mapped to existing infra)

1. **SAM 3 auto-label endpoint.** A new GPU SageMaker endpoint (mirror BioCLIP's
   separate-ECR pattern + the tracking endpoint's **scale-to-zero**), container based on
   EcoMorph's `SAM3Segmenter` (HF `facebook/sam3`). Input: frames + prompt(s) + threshold.
   Output: per-frame boxes+scores (+masks). Used *only* at label time → bounded cost.
   *Note:* `facebook/sam3` is **gated** — needs an HF account, license acceptance, and a
   token (a one-time setup step, like the gist token).
2. **Repurpose `PreAnnotateView`** to call SAM 3 instead of the YOLO (or add a
   `labeler=sam3` option). Masks→bbox→`Annotation` rows land in the existing schema, so
   the editor, export, and training all work unchanged.
3. **DINOv3 frame selection + drift detection.** A small step (same container or a light
   endpoint) that embeds candidate frames → cluster/select the subset to label, and
   computes the domain-shift score that *triggers* the loop.
4. **Auto-loop orchestrator.** A scheduled job (cron / the existing async pool) that:
   finds underperforming batches → selects → labels → (optional review gate) → kicks the
   existing `TrainingJob` → on success evaluates → auto-promotes `CustomModel.is_active`
   for the domain → re-runs the affected videos. Reuses `_spawn_training_job` + polling.
5. **Per-domain model selection.** Extend `CustomModel` with an optional site/device
   scope so the right fine-tuned model is auto-picked per video (today it's a manual UI
   pick). Promotion writes this.

---

## 5. Phasing (each independently useful)

- **P0 — Diagnose** (§0). Confirm it's domain shift. Cheap; do first.
- **P1 — SAM 3 labeler.** Deploy the SAM 3 endpoint; make it the pre-annotation engine.
  *Immediately valuable on its own:* it labels the failing videos and feeds a human-review
  pass — even before any auto-loop.
- **P2 — DINOv3 selection + drift trigger.** Active-frame selection, dedup, and the
  domain-shift detector that decides *when* to retrain.
- **P3 — Closed auto-loop.** Trigger → label → (review) → train → evaluate → auto-promote
  (with a regression guardrail) → re-run. Scheduled.
- **P4 — Per-site models + monitoring.** Domain-scoped `CustomModel`s, a drift/health
  dashboard (detections-per-frame, model-in-use per site, retrain history).

---

## 6. Open questions / decisions (for your review)

- **Human-in-the-loop:** fully auto vs a required 1-click verify pass. *Recommend*
  auto-label + **optional** review to start (quality up, effort low), tighten later.
- **SAM 3 licensing/gating:** OK to accept the `facebook/sam3` license + store an HF
  token? (one-time setup). Any research-use constraints for your deployment?
- **Detector version:** keep YOLOv8 (current `train.py`), or move to YOLO11/26?
- **Boxes vs masks:** SAM 3 gives instance masks; the tracker needs **boxes**
  (mask→bbox). Train YOLO-detect (recommended) or YOLO-seg if masks are wanted downstream.
- **Auto-promote guardrail:** need a per-domain held-out eval set + a promotion threshold
  (e.g. +X mAP on the new domain, no regression on the old) + rollback. Where does the
  eval set come from — a slice of verified SAM 3 labels?
- **Classes/prompts:** align SAM 3 prompts with the label classes (`bee`, `wasp`, `nest`);
  multi-taxa is fine (SAM 3 is open-vocabulary).
- **Trigger definition:** "underperforming" = detections/frame below a threshold, or a
  DINOv3 domain-distance cutoff? (Probably both.)
- **Cost:** SAM 3 GPU only at label time, DINOv3 cheap, training bounded, all
  scale-to-zero — consistent with [[project_aws_cost_audit]] /
  [[feedback_cloud_cost_conscious]]. Confirm the added SAM 3 endpoint is acceptable.

---

## 6b. P1 status — CODE COMPLETE (2026-07-03), pending deploy

Decisions taken: labeler = **SAM 3**; posture = **auto-label + optional review**.

Shipped (commits d1e9119 / 5e2a2a1 / 3b642fd):
- **P1a** `sagemaker_backend/sam3/` (Dockerfile.sam3, serve, inference.py) — GPU BYOC,
  loads facebook/sam3, two modes: `pre_annotate` (video → YOLO-seed boxes, same
  contract as the tracking worker) + `images` (ad-hoc base64).
- **P1b** infra: ECR `beemonitor-sm-dev-sam3` + GPU async **scale-to-zero** endpoint
  gated on `deploy-sam3`; Django invoke policy; CI `build-push-sam3` (bakes gated
  weights via the `HF_TOKEN` secret).
- **P1c** web: `SAGEMAKER_SAM3_ENDPOINT_NAME`; `PreAnnotate[All]View` `labeler=sam3`
  path; a **Labeler** dropdown (YOLO / SAM 3) on the annotations page. Auto-labels →
  `Annotation` rows → existing editor (review) → YOLO export → existing training.

### Deploy runbook (needs the SAM 3 gate first)
1. Accept the license at huggingface.co/facebook/sam3; create an HF token.
2. Add repo GitHub secret **`HF_TOKEN`** → the `build-push-sam3` CI job builds+bakes
   the image (tagged with the commit sha).
3. `infra/aws-sagemaker`: `pulumi config set deploy-sam3 true` +
   `pulumi config set beemonitor-sagemaker:sam3-image-tag <sha>` → `pulumi up`
   (creates the GPU endpoint `beemonitor-sm-dev-sam3`, scale-to-zero).
4. Set App Runner env **`SAGEMAKER_SAM3_ENDPOINT_NAME=beemonitor-sm-dev-sam3`**.
5. Test: annotations page → Labeler = **SAM 3** → Pre-annotate → boxes should appear
   on this year's frames where YOLO found none. Review, export, train (existing flow).

Remaining phases: **P2** DINOv3 frame-selection + drift trigger · **P3** closed
auto-loop (train → eval → auto-promote) · **P4** per-site models + monitoring.

---

## 7. Recommendation

Do **P0** immediately (it's ~an hour and tells us if this is even the right problem).
Then **P1 (SAM 3 labeler)** is the highest-leverage build — it fixes the current pain
(labels for the failing videos) and is a prerequisite for everything else. Treat the
full auto-loop (P3) as the goal but earn it after P1/P2 prove the labeler + selection.
DINOv3 is the "when + which frames" brain; SAM 3 is the "how to label" engine; the
existing `apps/training` pipeline is the "how to fine-tune" that already works.
