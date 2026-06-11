# BeeMonitor — Taxonomic Monitoring Agent Design

**Status:** Proposal for review (not yet implemented)
**Author:** Drafted with Claude Code, 2026-06-11
**Goal:** Turn raw activity into *taxonomic insight*. On top of the existing
telemetry + activity-clip pipeline, add (1) a perception pipeline that identifies
**what insect moved** — species via BioCLIP — from a few sampled frames per activity,
and (2) a Claude reasoning agent with that perception + the device's telemetry/GPS as
context, so the system can say things like *"2 individuals of* Bombus impatiens *at
site Y, 14:00–15:00."*

**Scope (this round): insect ID via BioCLIP only.** Plant ID (Pl@ntNet) and
plant–pollinator association are **deferred to a later phase** — see §8. Keeping it to
one model means no external API/keys, crop-only frames (cheaper cellular), and a
single SageMaker endpoint.

---

## 1. Motivation

The device already detects *that* something moved (motion-gated recorder) and where
the unit is (GPS in telemetry). What's missing is *what* moved and *what it means*.
Researchers want pollinator identity, counts, and plant associations at a taxonomic
level — not "47 activity clips." This layer closes that gap and is the natural
capstone on the cellular telemetry ([[10_cellular_telemetry_design]]) and dashboard
([[12_device_dashboard_telemetry_v2]]) work.

---

## 2. Locked decisions (this round)

- **Two layers, not one (pipeline + agent).** A deterministic perception pipeline
  auto-classifies each activity's frames (cheap, fast, consistent, searchable); a
  Claude agent layers reasoning, location priors, ambiguity resolution, and
  on-demand queries on top. Don't spend an LLM call per frame.
- **Models on SageMaker.** BioCLIP + an organism detector run as SageMaker
  endpoint(s), reusing the infra/auth pattern already in the repo. Pl@ntNet stays an
  external API.
- **On-device frame pre-filter over cellular.** The recorder picks the 1–3 most
  informative **crops of the mover** per activity and sends them over cellular under a
  daily cap; full video stays WiFi-gated. Best signal-per-byte, fits the
  split-transport model. (Wide context frames are only needed for plant ID — deferred.)

---

## 3. How it fits the existing codebase

Seams confirmed by survey:

- **Recorder** `hardware/main_motion.py`: `_save_telemetry_still()` already captures
  a downscaled JPEG into a `telemetry/` queue dir that the telemetry service ships
  over cellular (keeps last 3). The motion gate already computes blob bboxes — so the
  recorder can **crop around the mover** for free. New activity-frame sampling is a
  sibling of this, writing to an `activity_frames/` queue.
- **Telemetry** `hardware/telemetry.py`: ships the latest queued image to
  `/api/v1/devices/heartbeat`. Frame upload follows the same cellular path + cap.
- **Device-authed ingest** `apps/api/heartbeat.py` (+ `DeviceKeyAuthentication`,
  `MultiPartParser`, csrf-exempt registration in `apps/api/urls.py`): the exact
  pattern for a new `/api/v1/devices/frames` endpoint.
- **SageMaker** `apps/analysis/views.py`: `_invoke_endpoint_async()` + input/output
  S3 buckets, settings `SAGEMAKER_ENDPOINT_NAME/INPUT_BUCKET/OUTPUT_BUCKET`, infra in
  `infra/aws-sagemaker/`. The existing endpoint does video detection+tracking; the
  perception models go on a **separate real-time (sync) endpoint** (small crops, low
  latency) or a second async endpoint.
- **Agent scaffold** `apps/setup/assistant/` (`client.py` cached-prefix read-only
  tool loop with SSE streaming; `tools.py` access-scoped read-only tools; `safety.py`
  redaction): mirror this for the monitoring agent — proposes/explains, doesn't act.
- **Models** `apps/videos/models.py:Video` (device FK, `recorded_at`, `metadata`
  JSON), `apps/devices/models.py:DeviceHeartbeat`. New tables hang off Device/Video.

---

## 4. Architecture / data flow

```
DEVICE (recorder)                         CLOUD
─────────────────                         ─────
activity (clip START)                     ┌─ /api/v1/devices/frames  (device-authed, multipart)
  motion gate → blob bbox                 │     store crop → S3, create ActivityFrame rows
  sample 1–3 mover crops                  │            │
  enqueue → activity_frames/  ───cellular─┼────────────┘
  (daily cap; full video WiFi-gated)      │            ▼  (async, on ingest)
                                          │   PERCEPTION PIPELINE (insects)
                                          │     BioCLIP zero-shot ←── GBIF region taxa prior
                                          │            ▼
                                          │   Detection rows → Observation (dedup per activity, count)
                                          │            │
                                          │            ▼
                                          │   REASONING AGENT (Claude, on-demand + daily summary)
                                          │     tools: query observations, telemetry/GPS,
                                          │            re-identify, location priors
                                          │            ▼
                                          └─  dashboard: species timeline, counts, NL insight
```

---

## 5. Components

### 5.1 Device-side: activity-frame sampling (`hardware/main_motion.py`)
- On a recorded activity, pick the **most informative** frames: the frame(s) at
  **peak motion** (largest/most blobs) for the mover, optionally an entry/exit frame.
- Send tight **crops** of the mover bbox (great for BioCLIP, tiny bytes). Default
  1 crop, configurable up to 3 per activity. (The wider scene frame is only needed for
  the deferred plant-ID phase, so it's omitted for now.)
- Queue to `activity_frames/` with sidecar JSON (activity id, timestamp, bbox,
  motion score). A new uploader path (or telemetry) ships them over cellular.
- **Cellular cap:** `BEEMONITOR_FRAME_DAILY_CAP` (default e.g. 60 frames/day ≈
  ~15 MB/mo at ~250 KB). When capped, keep only the highest-motion activities; log
  what was dropped (no silent truncation). Full frames/video still go over WiFi.

### 5.2 Backend ingest (`apps/api/frames.py` + new model)
- `POST /api/v1/devices/frames` — multipart (image + sidecar JSON), `DeviceKeyAuth`,
  csrf-exempt. Uploads to S3 (`users/<id>/devices/<id>/frames/<yyyy>/<mm>/<dd>/...`)
  and creates an `ActivityFrame`. Fires the perception job (async task / on-demand).

### 5.3 Perception pipeline (`apps/monitor/pipeline/`) — insects only
- **Detector/crop:** MVP reuses the device-supplied bbox crop (no cloud detector);
  v2 adds a SageMaker arthropod detector for re-cropping/validation.
- **Insect ID — BioCLIP** (SageMaker real-time endpoint): zero-shot classify the
  crop against a **location-constrained candidate taxa list** (from GBIF/iNat for the
  device lat/lon) → ranked taxonomic hypotheses with confidence at each rank
  (order→family→genus→species). Location priors are the single biggest accuracy lever.
- **Records:** `Detection` per (frame, taxon, confidence). Per activity, **dedup
  individuals** (same mover across its frames = one `Observation` with a count
  estimate, best taxon, representative crop). Low confidence → flag for agent/human.
- **Plant ID is out of scope this round** (deferred — §8).

### 5.4 Reasoning agent (`apps/monitor/agent/`, mirrors `apps/setup/assistant/`)
- Cached system prefix (role + taxonomy/method rules); read-only tool loop; SSE.
- **Tools:** `query_observations(device, timerange, taxon?)`,
  `get_device_context(device)` (telemetry/GPS/site), `identify_again(frame_id,
  hint?)` (re-run/reprompt perception), `location_taxa(lat,lon)` (GBIF prior),
  `summarize_period(device, range)`.
- **Jobs:** (a) on-demand chat on the device page ("what visited this week?"); (b) a
  scheduled **daily digest** per active device (species seen, counts, new-for-site
  taxa, plant associations, anomalies vs telemetry). Agent *explains/flags* — it does
  not change device state.

### 5.5 Dashboard surfacing (`apps/monitor/` templates, device detail)
- Per-device **species list** + counts, an **observations timeline** (crop thumbnails
  + taxon + confidence), plant-association links, and the agent digest/chat panel.
  Presigned GETs for crops (browser ← S3), consistent with existing image handling.

---

## 6. New data model (sketch)

- **ActivityFrame** — FK Device, optional FK Video; `storage_key`, `kind`
  (crop|wide), `bbox`, `motion_score`, `captured_at`, `frame_meta` JSON.
- **Taxon** — cached taxonomy node: `rank`, `name`, `gbif_id`/`inat_id`, parent FK.
- **Detection** — FK ActivityFrame; `model` (`bioclip` now; field kept so plant ID can
  add `plantnet` later), FK Taxon, `confidence`, `raw` JSON (full ranked output).
- **Observation** — FK Device, FK activity/Video; best `Taxon`, `individual_count`,
  `confidence`, representative ActivityFrame, `observed_at`, GPS snapshot, `status`
  (auto|agent_reviewed|human_confirmed).

---

## 7. Phased execution plan

- **Phase 0 — frames flowing.** Device sampling + cap; `/devices/frames` + ingest;
  `ActivityFrame`; show crops on the device page. *Validate:* frames in S3, capped,
  visible. No ML yet.
- **Phase 1 — perception MVP (insects).** BioCLIP SageMaker endpoint (unconstrained);
  `Detection`/`Observation`; auto-run on ingest. *Validate:* activities get ranked
  insect taxa; spot-check accuracy on known clips.
- **Phase 2 — location priors.** GBIF/iNat region taxa list constrains BioCLIP
  zero-shot; per-activity individual dedup/count. *Validate:* measurable accuracy lift.
- **Phase 3 — reasoning agent.** `apps/monitor/agent` (mirror setup assistant); tools;
  on-demand chat + daily digest. *Validate:* NL insights grounded in real observations.
- **Phase 4 — surfacing + polish.** Species timeline, counts, map, human-confirm
  workflow, export.
- **Phase 5 (future) — plant ID.** Add a wide context frame on-device, Pl@ntNet (or
  iNaturalist CV) on it, `plantnet` detections, and plant–pollinator associations.

Each phase is independently shippable; the dashboard improves at each step.

---

## 8. Cost, risks, open questions

- **Cellular budget** — the daily frame cap is the control; sending **crops** (not
  full frames) is the big saver. Confirm the cap against the ~80 MB/mo target.
- **Field-image accuracy** — motion blur, partial/occluded insects, tiny crops hurt
  BioCLIP. Mitigations: best-frame selection on-device, location priors, confidence
  gating + human-confirm. Set expectations: genus/family often more reliable than
  species.
- **Double-counting individuals** — per-activity dedup is heuristic (one mover per
  activity to start); multi-individual scenes are a later refinement.
- **Plant ID deferred (Phase 5)** — when added: Pl@ntNet free-tier rate caps (cache by
  image hash; iNaturalist CV as a fallback), plus a wide context frame on-device.
  Check ToS for research use. Out of scope now.
- **SageMaker cost/cold-start** — a real-time endpoint bills while up; consider
  serverless inference or batching activities. Reuse the existing async pattern if
  latency isn't critical.
- **Model choice** — BioCLIP (`imageomics/bioclip`) is the default; evaluate BioCLIP-2
  / a fine-tuned head on local taxa if zero-shot underperforms. Edward's call (ML).
- **Privacy/data** — field-site frames are research data; keep the existing
  owner-scoped access model; presigned, expiring URLs only.

---

> **Revision (2026-06-11):** scoped down to **insect ID via BioCLIP only**. Plant ID
> (Pl@ntNet), wide context frames, and plant–pollinator associations moved to Phase 5.

## 9. Relationship to other work

Consumes telemetry/GPS from [[10_cellular_telemetry_design]] /
[[12_device_dashboard_telemetry_v2]]; reuses the Claude tool-loop pattern from
[[13_guided_setup_and_ai_tutor_design]]; independent of provisioning
([[14_golden_image_provisioning_design]]). The agent's read-only,
proposes-never-acts stance matches the setup assistant's safety model.
