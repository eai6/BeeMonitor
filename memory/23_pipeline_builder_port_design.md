# BeeMonitor — Visual Pipeline Builder via the pan-APA Workshop Port

**Status:** Design for review (drafted 2026-07-02)
**Author:** Drafted with Claude Code (Fable 5), 2026-07-02
**Goal:** Turn BeeMonitor's "Processing" area into a **Scratch-style, drag-and-drop
pipeline builder** where a user composes AI analyses over device videos/images to
extract **abstract ecological data** — foraging trips, flower/ROI visitation, colony
activity, species ID — without writing code. Concretizes the vision in
[[19_processing_workflow_builder_design]] by **porting an existing, working builder**
(the `pan-APA-AI-Workshop-2026` Django app) rather than building one from scratch.

Related: [[19_processing_workflow_builder_design]] (the original proposal — Phase 0
shipped), [[15_monitoring_agent_design]] (BioCLIP perception), `apps/analysis`
(foraging/interactions + the async Job/poll machinery), `src/beemonitor`
(detection/tracking primitives).

---

## 1. Key finding — the two systems are the same architecture

The Workshop app (`scratch/pan-APA-AI-Workshop-2026`, a Django project) already
implements the exact workflow model proposed in doc 19. BeeMonitor has the *engines*
and no builder; the Workshop has the *builder* and only thin engines. Porting is a
merge, not a rewrite.

| Concern | Workshop (`pan-APA`) | BeeMonitor today |
|---|---|---|
| Block/step definition | `pipeline/registry.py` `BLOCK_REGISTRY` (type → metadata) | *none* — fixed recipes in `ProcessingHubView` |
| Pipeline storage | `pipeline/models.py` `Pipeline.steps` = JSON `[{id, block_type, config}]` | *none* — one `Job` per analysis |
| Run record | `PipelineRun` (status, `input_data`, `step_results`) | `analysis.Job` + `JobResult` |
| Executor dispatch | `pipeline/engine.py` `EXECUTOR_MAP[type] → module.func` | hardcoded sequence in the SageMaker handler |
| The ML itself | thin API calls (Gemini/Whisper/YOLO) | **`BaseDetector`/`BaseTracking` + SageMaker GPU jobs** (the real value) |
| Builder UI | `pipeline/templates/pipeline/editor.html` — Scratch snap-blocks + HTMX | *none* |
| Bonus | `pipeline/notebook_generator.py` → Colab `.ipynb` | — |

The Workshop's `Pipeline.steps` JSON + registry + `EXECUTOR_MAP` **is** the typed-node
workflow model doc 19 proposed — already prototyped in the other repo.

### Workshop block schema (the abstraction we adopt)
Each block in `BLOCK_REGISTRY` declares: `display_name, description, category, icon,
input_type, output_type, backend ("local"|"api"), config_fields[]` where each config
field is `{name, label, field_type (text|textarea|number|select|file), required,
default, choices}`. Blocks are meant to "snap" when one's `output_type` matches the
next's `input_type` — the Scratch mechanic. (Workshop declares types but does **not**
enforce them; BeeMonitor must — see §3.)

---

## 2. What ports as-is vs. the real work

**Ports nearly verbatim** (copy the `pipeline` app into BeeMonitor):
`Pipeline`/`PipelineRun`/`PipelineAsset` models, `registry.py` schema +
`get_categories()`, `engine.py` dispatch (`get_executor`, `execute_step`),
`editor.html` + the HTMX add/reorder/configure endpoints, `notebook_generator.py`
(reuse for the STEM export angle), `sanitize.py` (strip secrets from errors).

**The 4 genuine gaps** — Workshop was built for cheap synchronous API calls; ecology
is expensive async GPU:

1. **Richer artifact types** (§3). Scalar `text/image/audio/data` → add
   `Video · Frames · ROI/NestLayout · Detections · Tracks · Events · Observations · Table`.
2. **Enforced port validation.** Workshop treats types as metadata only. Here a wrong
   connection burns a GPU job — validate on save **and** before Run.
3. **Multi-input steps** (§5). Workshop is strictly linear (`out[N] → in[N+1]`).
   Foraging needs two upstreams (`NestLayout` **and** `Tracks` → `ForagingTrips`).
   Fix: pass the whole accumulated **context** and let a step declare
   `inputs: {port: upstream_step_id}`.
4. **Async execution + cost** (§6). GPU steps are submit-to-SageMaker → poll, not
   blocking calls; and a graph can silently fan out cost. Reuse `Job` config-hash
   dedup for caching and show a cost estimate before Run.

---

## 3. Artifact-type system + port validation

Extend the scalar vocabulary with ecological artifact types that flow on edges as
**references** (S3 keys / model PKs), not inline blobs:

`Video, Frames/ImageSet, ROI (incl. NestLayout), Detections, Tracks, Events
(foraging trips / interactions), Observations (taxa), Table (rows for export/plot)`
plus the existing `text/image/data/any/none`.

Validation rule (enforced): edge is legal iff `upstream.output_type` is assignable to
`downstream.input_type` (`any` matches all; exact match otherwise). Run this on
step-connect (UI feedback), on save, and as a pre-flight before submitting jobs.

---

## 4. The BeeMonitor block palette (new `ecology` category)

Each block wraps code that **already exists** — this is orchestration, not new ML
(exceptions flagged):

| Block | in → out | Wraps (entry point) |
|---|---|---|
| `input.video` | – → `Video` | `apps/videos`; ProcessingHub video picker (`analysis/views.py`) |
| `roi.nest_layout` | – → `ROI` | **already built:** `Device.nest_layout`/`roi_override`, `devices.ROIEditorView` (`/devices/<id>/roi/`) |
| `roi.draw` | `Video` → `ROI` | ad-hoc region draw (reuse the ROI editor canvas) |
| `detect.nest` | `Video` → `ROI` | `src/beemonitor/detection/nest_detector.py` `NestDetector.detect_nests` |
| `detect.bee` | `Video` → `Detections` | `src/beemonitor/detection/yolo_detector.py` `YOLODetector` |
| `track.bee` | `Detections`/`Video` → `Tracks` | `src/beemonitor/tracking` `BaseTracking.process_video` + SageMaker video endpoint |
| `analyze.foraging_trips` | `Tracks`+`ROI` → `Events` | `apps/analysis` + `src/beemonitor/processing/event_processor.py` |
| `analyze.visitation` | `Tracks`+`ROI` → `Table` | **new (small):** count tracks entering ROI |
| `analyze.colony_activity` | `Video`/`Tracks` → `Table` | **new:** occupancy / motion-over-time metric |
| `identify.taxon` | `Frames`/`Tracks` → `Observations` | `apps/monitor/pipeline.py` `classify_frame` (BioCLIP) |
| `identify.marker` | `Tracks` → `Tracks+id` | **new ML:** per-trajectory QR / color-tag decode |
| `filter.roi / .confidence / .taxon / .time` | `X` → `X` | thin filters |
| `output.table / .chart / .summary / .dataset` | `*` → – | `analytics.py`, CSV export, LLM NL summary, `annotations` YOLO export |

Only two blocks are net-new ML: `analyze.colony_activity` and the advanced
`identify.marker`. Everything else is a wrapper over existing code.

---

## 5. Pipeline data model (extends the Workshop model)

```jsonc
// Pipeline.steps  (JSON)
[
  { "id": "a1b2", "block_type": "input.video", "config": {"video_id": 42} },
  { "id": "c3d4", "block_type": "track.bee",    "config": {"conf": 0.4},
    "inputs": {"video": "a1b2"} },                       // ← multi-input via named ports
  { "id": "e5f6", "block_type": "analyze.foraging_trips",
    "inputs": {"tracks": "c3d4", "rois": "b2c3"} }
]
```
- `inputs: {port: upstream_step_id}` is the only addition to the Workshop schema.
  Absent `inputs` ⇒ implicit "previous step" (keeps linear pipelines simple).
- `PipelineRun` gains `context` (JSON: `step_id → artifact ref`) and **per-step
  status** (`pending|running|waiting_gpu|done|failed`). GPU steps spawn a real
  `analysis.Job` tagged `(run_id, step_id)` — reuse that model wholesale.

---

## 6. Execution — resumable state machine (decided: option B)

No new job runtime, no Celery. The builder rides the existing async + poll machinery
that already drives the Processing hub (`analysis.views.PollJobsView`,
`_poll_sagemaker_results`, `invoke_endpoint_async`).

`advance_run(run)` (called by the poller when a tagged `Job` finishes):
1. write the finished step's output into `run.context`;
2. execute any now-ready **local** steps inline (input/roi/filter/output — instant);
3. submit `Job`s for any now-ready **GPU** steps
   — a step is *ready* when every `inputs:{}` upstream id is `done`;
4. if nothing remains → `completed`.

Consequences:
- **Readiness = all inputs done** ⇒ the *same* scheduler runs Phase-1 linear pipelines
  **and** the Phase-2 DAG canvas. The executor is never rewritten for the visual graph.
- **No held threads; survives restarts** — run state lives in the DB.
- **Caching:** `Job`'s existing config-hash dedup means re-running a pipeline skips
  unchanged steps for free.
- **Cost control:** compute a pre-Run estimate (count GPU steps × tier) and show it
  before the run starts ([[project_aws_cost_audit]], [[feedback_cloud_cost_conscious]]).

---

## 7. The three seed pipelines (as step graphs)

```
Foraging trips:   input.video → detect.nest ─┐   (or roi.nest_layout)
                              → track.bee ────┴→ analyze.foraging_trips → output.table
                                       └(advanced)→ identify.taxon + identify.marker (per track)

Flower/ROI visits: input.video → roi.draw(flower) → track.bee
                              → analyze.visitation → identify.taxon → output.chart

Colony activity:   input.video(in-nest) → track.bee/analyze.colony_activity → output.table
```
Same palette recombined — the point of the abstraction. Ship each as a saved
template (`Pipeline.is_template`).

---

## 8. Roadmap (extends doc 19 phases)

| Phase | Deliverable |
|---|---|
| **0 — done** | One-click recipes (Processing hub, shipped) |
| **1 — Port + linear ecology pipeline** | Copy `pipeline` app → `apps/pipelines`; add `ecology` palette (§4); artifact types + **enforced** port validation; multi-input `inputs:{}` + context; `advance_run` on `PollJobsView`; cost estimate before Run. Ship **Foraging-trips** as flagship template. |
| **1.5 — Visitation + Colony** ✅ | `analyze.visitation` + `analyze.colony_activity` implemented as **local** post-processors over the tracking CSV (`ops.py`): schema-tolerant loader (`track_id/frame/cx/cy`, pixel→0..1 normalise), ROI∩tracks visit/dwell counting, and time-binned occupancy/detection series. Colony template now includes a `track.bee` step. |
| **2 — Visual DAG canvas** ✅ | **Drawflow** canvas (CDN, no build step) replaces the linear editor: palette drag-drop, named typed ports (multi-input via `get_input_ports`), inline node config, save→`graph_to_steps`. `Pipeline.graph` stores the raw layout; `steps` stays the source of truth (canvas rebuilt from `steps` + saved positions). Scheduler + executors unchanged. Linear-editor HTMX endpoints removed. |
| **3 — Education + advanced ID** | (a) `identify.marker` ✅ — real per-track individual ID (colour/QR/number) as a **local** step over the tracking CSV's `bee_id`/`bee_id_method`/`bee_id_confidence` columns (`ops.marker_identities`); the track job sets `identify_bees` when a marker step is downstream. New "Individual bee IDs" template. (b) ✅ **Notebook export** (`notebook.py`): a pipeline → runnable Colab `.ipynb` — per-block markdown (explanation + config) + code cells (real pandas logic for analyze/output, guided YOLO/tracking/BioCLIP scaffolds), pip installs derived from block types. `export_notebook` view + "⬇ Notebook" button. (c) ✅ **STEM lesson packs** (`lessons.py`): 4 explainable lessons (objectives + sections + guided questions) tied to the seed templates; `/pipelines/lessons/` list + detail; "Start this lesson" clones the template onto the canvas. "Lessons" nav link. |

**Status 2026-07-02: Phases 0–3 all shipped.** The full memory/23 design is implemented
end-to-end (builder → engine → analytics → visual DAG canvas → advanced ID → notebook
export → lessons), committed on `main`.

**Processing-hub integration (2026-07-02).** The `/analysis/processing/` hub's fixed
"What to run" recipe (tracking/species toggles + model selectors + ROI toggle) is
**replaced by a pipeline picker**: choose one of your pipelines/templates and run it on
the filtered/selected videos. `pipelines.run_on_videos` creates one `PipelineRun` per
video via `engine.steps_with_video` (injects each video into the `input.video` step) and
returns the same AJAX JSON contract as `analysis.BatchJobView`, so the hub's existing
per-video live status + `analysis:poll` work unchanged (each GPU step spawns a tagged
`analysis.Job` for that video). Model + ROI choices now live inside the pipeline (the
`roi.nest_layout` block reads the video's device layout per-run).

**Outputs moved to run history (2026-07-03).** The hub's static "Extracted ecological
data" CSV block (foraging/interactions/events/tracking/nest/species) is removed —
those are now **outputs of a pipeline run**. New **Run history** (`pipelines.run_list`
→ `/pipelines/runs/`, "Runs" nav link) lists every `PipelineRun` for the user (pipeline,
input video, status, progress); the run detail surfaces per-step outputs with a per-step
`⬇ CSV` download (`run_output_csv`, serialises table rows) and a "results →" link to the
underlying `analysis.Job` CSVs for GPU steps. Run views now authorise by `run.user`
(not template ownership) so hub-launched template runs are viewable.

**Re-run with step caching (2026-07-03).** A `PipelineRun` already persists frozen
`steps`, `step_status`, and per-step `context` (outputs). Added a **`StepResult`** cache
(`user, cache_key, block_type, output`): GPU steps are keyed by a hash of their
*effective* job config (video + resolved ROI/nest layout + flags via
`build_detect_and_track_config`). On advance, a matching key reuses the cached output
**instantly, with no SageMaker job**; only *successful* outputs are cached, so a failed
GPU step (e.g. `SAGEMAKER_ENDPOINT_NAME` not set) re-runs next time. Local steps always
recompute (cheap, and they read live data like device ROI) — a changed ROI flows into the
GPU key and forces a re-run. **`rerun`** view/button starts a fresh run of the *current*
pipeline on the old run's video; unchanged steps reuse cache. Run detail shows a `cached`
badge per reused step. Hub notice removed.

---

## 9. Where it lives / port mechanics
- New Django app **`apps/pipelines`** (mirrors the Workshop `pipeline` app). Keep the
  `runner/` sandbox out for now (BeeMonitor doesn't run arbitrary user code).
- Port files: `models.py`, `registry.py`, `engine.py`, `templates/pipeline/editor.html`
  + HTMX views, `notebook_generator.py`, `sanitize.py`. Replace/extend `registry.py`
  with the ecology palette and swap the Workshop executors for BeeMonitor ones that
  submit `Job`s.
- **Cleanup:** the review copy lives at `scratch/pan-APA-AI-Workshop-2026/` (untracked)
  plus an empty `.scratch-review/`. Gitignore or delete both so the Workshop repo is
  never committed into BeeMonitor. Copy only the specific files into `apps/pipelines`.

---

## 10. Risks / open questions
- **Artifact storage** — settle S3-key vs model-PK conventions for `Tracks`/`Events`
  refs on edges before coding the executors.
- **Phase-2 canvas lib** — React Flow vs Rete.js vs Drawflow vs custom (doc 19 §7).
  Affects how "Scratch-like" vs "node-graph" it feels. Edward's call.
- **`identify.marker`** — per-trajectory QR/color-tag ID is genuinely new ML; scope it
  as its own effort (re-ID + marker decode), not part of Phase 1.
- **Metering** — Workshop counts API calls; BeeMonitor uses credits + scale-to-zero.
  Reconcile the quota model with `accounts` credits.
- **Notebook export for GPU steps** — Workshop notebooks assume Colab-runnable code;
  BeeMonitor GPU steps map to SageMaker. For STEM export, generate CPU/Colab-friendly
  equivalents or clearly mark server-only steps.

---

## 11. Recommendation
Port in Phase 1 as a **linear** builder over the ecology palette, with enforced port
types and the `advance_run` scheduler on the existing poller; ship Foraging-trips as
the first template. The visual DAG canvas (Phase 2) reuses the same scheduler, so it's
a UI project, not an engine rewrite. Treat `identify.marker` and STEM lesson packs as
Phase 3.
