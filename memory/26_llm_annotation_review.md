# BeeMonitor — LLM Vision Review of Annotations

**Status:** Plan (approved decisions, 2026-07-04) — ready to implement.
**Goal:** A vision LLM vets a pre-annotated frame — **removes false positives** using
the existing boxes + confidences as context — to cut the detector's over-detection on
bee-hotel scenes. Feeds the review-status system (`review_source="llm"`, blue ✓ badge).

## Decisions
- **Filter-only.** The LLM decides which *existing* boxes are real insects (keep) vs
  false positives (remove). It does **not** invent coordinates (LLMs are unreliable at
  precise boxes). This directly targets "detects a lot of things that aren't bees."
- **Models:** batch = **claude-haiku-4-5** (`ASSISTANT_FAST_MODEL`, cheap bulk);
  per-frame in the editor = **claude-sonnet-4-6** (`ASSISTANT_MODEL`, accurate).
- **Scope:** batch runs over **unreviewed frames only**, capped (~200/run); never
  re-charges or overrides a **human**-reviewed frame.

## How it works (the reliability trick)
Draw the existing boxes on the frame **numbered**, send that overlay image, and ask the
model which *numbers* are real — so it references boxes by index, never emits pixels.

`apps/annotations/llm_review.py`:
- `available()` → `bool(settings.ANTHROPIC_API_KEY)`.
- `_frame_jpeg(ann)` → bytes of `frame_image_path` (processed bucket, via `presigned_get`).
- `_numbered_overlay(jpeg, boxes)` → Pillow draws each box + its index.
- `review_boxes(ann, boxes, model)` → Claude **tool-use** (`report_review` → `{keep:[int], notes}`);
  returns `[boxes[i] for i in keep]`. Falls back to the input boxes on any error
  (never destructive without a clear signal).
- `review_annotation(id, model, persist)` → filter + optionally save as `llm`-reviewed;
  skips `review_source="human"`.

## Triggers
- **Per-frame** — editor "AI Review" button: POST current boxes → Sonnet filters →
  returns kept boxes → canvas updates (not persisted; you Enter to save = human ✓).
- **Batch** — detail "LLM Review" → daemon thread over unreviewed frames with Haiku →
  persists each as `llm`-reviewed. Redirect + "refresh to see blue ✓".

## Contract (structured tool output)
```json
{ "keep": [indices of real insects], "notes": "3 removed on nest holes" }
```
New boxes = `[boxes[i] for i in keep]`. No adds (filter-only).

## Cost & safety
- Vision cost per frame → batch caps + "N skipped" logging; per-video scoping later.
- Idempotent; never overrides human review; charges credits like pre-annotate.
- Prereq: **`ANTHROPIC_API_KEY` set in prod** (also gates the setup assistant).

## Phasing
- **B1** per-frame AI Review (Sonnet) · **B2** batch LLM Review (Haiku, unreviewed) ·
  **B3** (later) optional "suggest missed bees" as confirm-only additions.

Related: [[25_auto_finetuning_design]] (the auto-label → review → train loop this
strengthens); review-status model shipped in commit e869f92.
