# BeeMonitor — Public Pipeline API (build + run pipelines from Colab)

**Status:** Plan for review (drafted 2026-07-03)
**Author:** Drafted with Claude Code (Fable 5), 2026-07-03
**Goal:** Expose the visual pipeline builder over a **public REST API** (API-key auth)
so a user can, from **Google Colab** (or any client), build a pipeline, upload videos,
run it against the **real SageMaker endpoints**, and pull per-step results as data —
turning the exported notebook (Phase 3b, [[23_pipeline_builder_port_design]]) into
something that *actually runs for real*, not just scaffold code.

Related: [[23_pipeline_builder_port_design]] (the builder + engine this exposes),
`apps/api` (existing DRF API), `apps/accounts` (`APIKey` + billing), `apps/developer`
(API-key issuance UI). Modelled on the **EcoMorph** API pattern
(`~/Documents/GitHub/EcoMorph` — to be reviewed; see §7).

---

## 1. The foundation already exists

BeeMonitor is not starting from scratch — most of the plumbing is in place:

| Need | Already there |
|---|---|
| API-key model | `apps/accounts/models.py` `APIKey` (`bmk_<type>_<random>`, sha256-hashed, revocable) |
| API-key issuance UI | **Developer page** (`apps/developer` → "Create API Key", shown once, revoke) — *this is where each user gets their key* |
| API-key auth | `apps/api/authentication.py` `APIKeyAuthentication` (`Authorization: Bearer bmk_…`) |
| DRF API app | `apps/api` (auth, jobs/sync, uploads, web-uploads, devices) |
| Presigned uploads | `apps/api/web_uploads.py` (presign + complete; already stores `site_name`) |
| Pipeline engine + runs | `apps/pipelines` — `validate_steps`, `graph_to_steps`, `steps_with_video`, `engine.start_run`/`advance_run`, `StepResult` cache, per-step outputs |
| Billing | `accounts` credits — GPU runs already bill the user on the web path |

So this work is **exposing the pipeline layer over the existing API-key auth**, not
building auth, keys, or a run engine.

---

## 2. Colab flow (the user experience)

1. **Developer page → Create API Key** → copy `bmk_…` (per-user, shown once).
2. In Colab: `pip install beemonitor` (or paste a client cell); set `BEEMONITOR_API_KEY`.
3. Build + run:
   ```python
   from beemonitor import BeeMonitor
   bm = BeeMonitor(api_key)                       # or reads BEEMONITOR_API_KEY
   vid = bm.upload_video("clip.mp4", site="Mendel's Garden")
   p   = bm.from_template("Foraging trips")       # or bm.create_pipeline(steps=[...])
   run = bm.run(p, video_ids=[vid.id])            # runs on REAL SageMaker endpoints
   run.wait()                                     # polls status
   trips = run.step_output("f")                   # -> pandas DataFrame
   run.summary()                                  # per-step status + metrics
   ```
4. Real GPU tracking + local analytics execute; the run bills the user's credits like
   the web path. Results come back as data (CSV/JSON → pandas).

---

## 3. API surface (new — under `/api/v1/pipelines/`, API-key authed)

All authed via the existing `APIKeyAuthentication`; scoped to the key's user; reuse the
tier throttle.

**Introspection**
- `GET /api/v1/pipelines/blocks/` — the block registry (`serialize_blocks()`): block
  types, categories, typed input/output ports, config fields. Lets a client build a
  valid graph programmatically.

**Pipelines (CRUD)**
- `GET  /api/v1/pipelines/` — list the user's pipelines (+ templates).
- `POST /api/v1/pipelines/` — create `{title, steps[]}` (validated by `validate_steps`).
- `GET/PUT/DELETE /api/v1/pipelines/{id}/` — retrieve / update steps / delete.
- `POST /api/v1/pipelines/validate/` — validate a `steps[]` graph, return errors, no save.
- `POST /api/v1/pipelines/{id}/clone/` — clone a template into the user's pipelines.

**Runs**
- `POST /api/v1/pipelines/{id}/run/` — body `{video_ids:[...]}` (or a filter). Creates one
  `PipelineRun` per video via `steps_with_video` + `engine.start_run`; returns run ids.
  (Reuses the exact `run_on_videos` logic.)
- `GET  /api/v1/runs/` — list the user's runs.
- `GET  /api/v1/runs/{id}/` — status + per-step status + per-step outputs (sanitized
  context — strip `_`-prefixed keys).
- `GET  /api/v1/runs/{id}/steps/{step_id}.csv` — a step's tabular output (reuse
  `run_output_csv`). JSON variant `…/output` for non-tabular.

**Uploads (API-key)**
- `POST /api/v1/pipelines/uploads/initiate` + `/complete` — presigned video upload keyed
  to the API key's user (reuse `web_uploads` logic; auth = API key instead of session).

**Serialization:** DRF serializers for `Pipeline` (id, title, steps, updated_at) and
`PipelineRun` (id, pipeline, status, step_status, per-step outputs, started/completed).

---

## 4. Notebook export tie-in (the headline)

Update `apps/pipelines/notebook.py` (Phase 3b) so **"⬇ Notebook"** can emit an
**API-driven** Colab that runs the *real* pipeline:
- a setup cell (`pip install beemonitor`, paste key from the Developer page),
- cells that call `bm.run(...)` + fetch per-step outputs,
- optionally keep the open-tool scaffold cells as an "offline / educational" variant.

This is exactly "run the pipeline for real via Colab using keys + real endpoints."

---

## 5. Auth, limits, cost
- Reuse `APIKeyAuthentication`; consider a `pipeline` key-type / scope so a key can be
  limited to pipeline ops (optional; existing key types may suffice).
- Runs bill credits like the web path; enforce the same budget/concurrency checks
  (`UserProfile.has_budget`, `max_concurrent_jobs`).
- Rate-limit uploads + runs (tier throttle already exists).
- GPU stays scale-to-zero ([[23_pipeline_builder_port_design]]); the API doesn't change
  the compute path, just its trigger.

---

## 6. Phasing
- **P1 — read + run.** `blocks/`, pipelines CRUD, `validate/`, `run/`, `runs/{id}` status +
  per-step output (GET), `steps/{id}.csv`. Ship a Colab **client snippet** (single cell).
- **P2 — uploads via API key.** presign/complete keyed to the API key's user.
- **P3 — `pip install beemonitor` client** + `notebook.py` emits API-driven notebooks.
- **P4 — polish.** API docs page (extend `apps/docs`), Developer-page copy for pipeline
  usage, examples.

---

## 7. EcoMorph alignment (open)
The user wants this to mirror EcoMorph's API/client/Colab pattern
(`~/Documents/GitHub/EcoMorph`). That path is outside the repo and TCC-blocked, so it
must be copied in to review (e.g. `rsync … ~/Documents/GitHub/EcoMorph/ scratch/EcoMorph/`).
Once available, reconcile: key-type/scopes, client package shape + method names, run/poll
semantics, error envelope, and the Colab bootstrap cell — then implement P1.

---

## 8. Recommendation
Build **P1** first (introspection + CRUD + run + per-step read) — it's a thin DRF layer
over machinery that already exists and immediately enables a Colab client. Defer the
`pip` package (P3) until the endpoint contract is proven with the single-cell client.
Align the exact shapes to EcoMorph before coding.
