# BeeMonitor — Processing Hub → Visual Workflow Builder

**Status:** Proposal for review (IA pivot started 2026-06-15; builder is future, phased)
**Author:** Drafted with Claude Code, 2026-06-15
**Update (2026-07-02):** Phase 0 (Processing hub of one-click recipes) has shipped.
The concrete plan for Phases 1–3 is now [[23_pipeline_builder_port_design]] — port the
working `pan-APA-AI-Workshop-2026` Scratch-style builder (same Django/steps-JSON/registry
architecture) and back its blocks with the existing SageMaker jobs. Read doc 23 for the
block palette, artifact-type system, and the resumable-state-machine executor.
**Goal:** Turn the "Processing" page from an analytics view into a **visual,
drag-and-drop workflow builder** (Roboflow Workflows / Scratch style) where a user
composes AI analysis pipelines over uploaded videos/images to extract **abstract
ecological data** — foraging trips, interactions, species ID, counts — without
writing code. Serves the **STEM-education** angle (visual, explainable, lesson-able)
while making the platform genuinely flexible and adaptable.

Related: [[15_monitoring_agent_design]] (BioCLIP perception), [[17_bee_confirmation_design]]
(on-device filter), `src/beemonitor` (the YOLO MOT tracking package), the
`analysis` app (foraging trips + interactions already exist).

---

## 1. IA pivot (done 2026-06-15, commit da9d904)
- **Landing (`/`) → device list** (the operational home).
- Top nav: **"Processing"** (→ `analysis:analytics`) replaces "Set up"; the brand
  points to the device list. Adding a device lives on the Devices page.
- Next: the "Processing" entry grows from the analytics page into the hub below.

---

## 2. We already have the building blocks ("nodes")
The builder is mostly **wrapping primitives that already exist** — this is an
orchestration + UI layer, not new ML:

| Capability | Where it lives today |
|---|---|
| Video ingest (uploads, device clips) | `apps/videos`, `apps/api/uploads` |
| YOLO **bee tracking** (MOT) | `src/beemonitor/tracking` + SageMaker video endpoint |
| **Nest / hotel** detection | `models/nest_detection.pt`, `src/beemonitor/detection` |
| **Foraging-trip** detection | `apps/analysis` (foraging_trip fields, daily summary) |
| **Interaction** detection (target↔target) | `apps/analysis` (interaction fields) — *data already produced* |
| **BioCLIP** taxonomic ID (crops/images) | `apps/monitor/pipeline` + BioCLIP endpoint |
| On-device **bee confirmation** | `hardware/motion/confirm.py` (tags clips) |
| Analytics / charts / CSV export | `apps/analysis/analytics.py`, views |

So a "tracking" node = the existing MOT job; a "foraging trips" node = the existing
analysis; an "identify species" node = BioCLIP; etc. The novelty is letting users
**compose** them on a canvas.

---

## 3. The model — a workflow is a typed DAG of nodes
- A **workflow** = a JSON graph: `{nodes:[{id,type,params}], edges:[{from,to,port}]}`.
- Each **node** = one processing step with typed input/output **ports**. Edges pass
  typed artifacts between nodes.
- **Artifact types** flowing on edges: `Video`, `ImageSet/Crops`, `Detections`,
  `Tracks`, `Events` (foraging trips / interactions), `Observations` (taxa),
  `Table` (rows for export/plot).

### Node palette (initial, fixed set)
- **Inputs:** Video (pick uploaded / device clips by site/date), Image set.
- **Detect:** Bee detector, Nest detector, (custom model) → `Detections`.
- **Track:** MOT tracker (`Detections`→`Tracks`).
- **Analyze:** Foraging trips (`Tracks`→`Events`), Interactions (`Tracks`→`Events`),
  Counts/Visitation (`Tracks/Events`→`Table`).
- **Identify:** BioCLIP species ID (`Crops`→`Observations`), region-prior constrain.
- **Filter:** ROI (hotel), confidence, species/taxon, time window.
- **Output:** Table/CSV, Chart, Ecological summary (NL), Dataset export (training).

A node validates that its input port type matches the upstream output — the same
"types must match" idea that makes Scratch blocks snap or not.

---

## 4. Execution
- The graph JSON is saved (`Workflow` model) and **run** as a `WorkflowRun`.
- A backend **executor** topologically sorts the DAG and runs each node, reusing
  the existing async job pattern (`spawn_gpu_job_async` / the bounded pools) and
  the SageMaker endpoints (scale-to-zero — [[15_monitoring_agent_design]] §10).
- Node results **materialize as the existing models** (Detection, Observation,
  ForagingTrip, Interaction) plus per-run artifacts; outputs render as the
  existing charts/CSV. Cost stays controlled by the same scale-to-zero +
  classify-once levers ([[project_aws_cost_audit]], [[feedback_cloud_cost_conscious]]).
- Determinism + caching: a node's output is keyed by (node type, params, input
  artifact hash) so re-running a workflow re-uses unchanged steps (Roboflow does this).

---

## 5. Phasing (each independently shippable; UI complexity grows last)
- **Phase 0 — Processing hub (no graph).** Make "Processing" a page that lists your
  videos and offers the existing analyses as **one-click recipes** (Track ▸ Foraging
  trips ▸ Interactions ▸ Identify species). Wraps current jobs; surfaces results.
  *This is the smallest step and immediately useful.*
- **Phase 1 — linear pipeline.** A guided "input → steps → output" builder (ordered
  list, not a free canvas): pick a video, add steps from the palette, run. Saves a
  (linear) Workflow. Validates port types between consecutive steps.
- **Phase 2 — visual node graph.** The Roboflow/Scratch canvas: drag nodes, connect
  ports, configure, save/load named workflows, run + watch progress, branch/merge.
  (Client: a node-graph lib, e.g. React Flow / Rete.js / Drawflow; server: executor.)
- **Phase 3 — extensible + education.** User-addable nodes (custom models from the
  `training` app), shareable workflow templates, and **STEM lesson packs**
  (pre-built explainable workflows + guided questions). Export ecological datasets.

---

## 6. Enrollment + "Add device" merge (separate IA cleanup)
Today there are two paths: `DeviceCreateView` (manual per-device key) and
`DeviceEnrollmentView` (zero-touch token → golden image). With the golden image,
**these are one thing**. Plan: a single **"Add a device"** page where the primary,
default path is golden-image/token enrollment, with the manual-key option tucked
under "advanced / no golden image." One nav/entry, one mental model. Low-risk,
self-contained — can land before the builder.

---

## 7. Risks / open questions
- **Scope** — the full node-graph builder is large; Phase 0/1 deliver value fast
  and de-risk the executor before the canvas UI. Don't build Phase 2 first.
- **Executor design** — typed ports + artifact passing + caching is the real
  engineering; keep the node set small and the types few to start.
- **Compute cost** — reuse scale-to-zero endpoints + result caching; a workflow
  must not silently fan out expensive GPU jobs (show an estimate before "Run").
- **Client lib choice** (Phase 2) — React Flow vs Rete.js vs Drawflow vs a custom
  canvas; affects how "Scratch-like" vs "node-graph" it feels. Edward's call.
- **Where compute runs** — all server-side (current pattern) vs some in-browser for
  light steps; start server-side.
- **STEM framing** — lesson packs + explainable nodes are the differentiator; design
  them with a real curriculum in mind, not bolted on.

---

## 8. Recommendation
Ship the IA pivot (done), then **Phase 0 (Processing hub of one-click recipes)** as
the next concrete step — it unifies the analyses that already exist into the new
"Processing" home and is small. Treat the visual builder (Phase 2) as the headline
feature to design in detail once Phase 0/1 prove the executor + node model.
