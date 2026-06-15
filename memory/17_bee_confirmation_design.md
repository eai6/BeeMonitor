# BeeMonitor — On-Device Bee Confirmation (low-DL motion filter)

**Status:** IMPLEMENTED 2026-06-14 (device + cloud). Device: `motion/confirm.py`,
recorder hooks, config knobs, telemetry count gate via `.unconfirmed` marker,
uploader `bee`-tag carry, `motion_replay --confirm`. Cloud: `/uploads/complete`
stores the `bee` tag on `Video.metadata` (`bee_confirmed` bool + `bee` sub-dict);
`Device.frame_daily_cap` field (migration 0017) pushed in the heartbeat (both the
beat + the fast command poll) and settable in the Django admin; BioCLIP
auto-processing kill-switch `MONITOR_BIOCLIP_AUTOPROCESS`. **Mode is
dashboard-switchable at runtime** (no shell, no restart): `BeeConfirmer.set_mode`
+ a hot-reloaded `bee_confirm_mode.json` (telemetry writes it from the heartbeat;
recorder reloads on the override tick) + `Device.bee_confirm_mode` (migration 0018)
in the admin — so you can flip off/tag/gate per unit remotely. Off-device verified
(unit tests + runtime mode transitions + Django check + `--confirm` against the
real model). **Pending:** optional polished in-page dashboard control (admin works
today); on-Pi shakedown. **Validate per-site with `--confirm` before enabling `gate`** —
see the finding below.
**Author:** Drafted with Claude Code, 2026-06-14

> **Field-fit finding (2026-06-14):** on a sample hive clip, `bee_tracking.pt`
> fired on **~0 movers even at conf 0.05** (movers were tiny edge blobs). The
> confirm/overlap logic is correct (verified), but it means **`gate` mode would
> suppress most activity on footage the model doesn't fire on**. So: run
> `motion_replay --confirm` on real site clips first; likely lower
> `BEEMONITOR_BEE_CONFIRM_CONF` and/or start in `tag` mode; confirm `bee_tracking.pt`
> actually fits the deployment camera/scale before trusting `gate`.
**Goal:** Stop the recorder treating *any* ROI motion (cast shadows, wind-blown
leaves, sun/cloud light shifts) as an activity. Confirm the mover is actually a
**bee** with the YOLO bee model (`models/bee_tracking.pt`) — but at the **edge,
on a Pi 4 CPU, with very few DL runs**.

Related: [[15_monitoring_agent_design]] (cloud BioCLIP species ID — the layer
this feeds), [[10_cellular_telemetry_design]] (crop transport + daily cap),
the `motion/` package refactor that created the `motion/confirm.py` seam, and
the on-Pi reality in `claude-memory/pi-torch-must-be-cpu-wheel.md`.

---

## 1. Problem & framing

The recorder's hot path is **MOG2 motion only** (`motion/gate.py`). MOG2 with
`detectShadows=True` already drops *soft* shadows, but hard shadows, foliage, and
bee-sized non-bee movers still trip the gate → every such trip becomes a counted
**activity** in telemetry and queues mover crops over cellular for **automated
cloud BioCLIP runs** — counts and paid compute spent before anything checks
whether a **bee** was present. (The video clip itself we still want — it uploads
on WiFi either way; the waste is the *automated count + cloud runs*, not the
recording.)

The source `beemonitor` package solves the analogous problem in the cloud with a
three-stage cascade: **cheap motion → YOLO confirms a bee → tracker confirms over
time** (only confirmed tracks become activity). On-device we can't run YOLO every
frame (Pi 4 CPU; `model.predict()` is the expensive call — see the SIGILL note).
So we port only the **YOLO-confirm** stage, made as cheap as possible, and leave
fine species ID to the existing cloud BioCLIP layer.

The result is the **edge analog of the source cascade**, with cost decreasing as
the stage gets more expensive:

| Stage | Where | Cost | Question it answers |
|------|-------|------|---------------------|
| MOG2 blob gate | device, every frame | ~1–2 ms | did something *bee-sized* move? |
| **YOLO confirm (this doc)** | device, few **full frames**/activity | ~1–3 s/frame (Pi 4 CPU) | is the mover a **bee**? |
| BioCLIP | cloud, per uploaded crop | SageMaker | **which species**? |

> **`bee_tracking.pt` is a whole-frame detector**, same as the startup hotel
> detector `nest_detection.pt` (`detect_hotel_roi` → `model.predict(frame_bgr)`
> on the full capture) and same as the source `beemonitor` (`process_frame` runs
> `yolo_detector.detect(frame)` on the **full frame**). It was trained to find
> small bees inside the full scene — feeding it a tight upscaled mover crop would
> be out-of-distribution and *hurt* recall. So confirmation runs on the **full
> main-stream frame**, then ties the detection to the mover with an overlap test.

---

## 2. Core idea — confirm on the frame we already capture, async, confirm-then-coast

The recorder **already** grabs a full main-stream frame per sample point to cut
the mover crop for the cloud pipeline (`motion/recorder.py` sampling block →
`_main_array_to_bgr(cam.capture_array("main"))` → `_mover_crop` → `act_cands`,
gated to one capture per `FRAME_CAPTURE_INTERVAL` s, capped at
`FRAME_MAX_CANDIDATES`). Confirmation **piggybacks entirely on that frame** — it
runs YOLO on the same full `main_bgr`, in-memory, the moment we have it (we still
crop it for the queue afterwards).

> **The only new cost is the YOLO inference itself. Zero additional camera
> captures, zero new sampling cadence.**

Because it's a full-frame run, a bee resting elsewhere in frame must not
"confirm" a shadow's motion — so a detection only counts if a YOLO **bee box
overlaps the MOG2 mover blob** (`gate.last_blobs`). This is the exact pairing the
calibration pass already does (`motion/calibrate.py` → `_bbox_overlap(blob,
yolo_box)`); the confirmer reuses it.

The "very few DL runs" guarantee comes **not from cropping** but from stacking
these count-reducing levers:

1. **Per-activity, not per-frame.** Confirmation is a property of the *activity*,
   so a handful of frames settle it — never the per-frame YOLO the Pi can't afford.
2. **Confirm-then-coast.** The instant one frame yields a bee ≥ threshold over the
   mover, the activity is `confirmed` and **no more YOLO runs for it**. A real-bee
   activity typically costs **exactly one inference**.
3. **Bounded negative budget.** A non-bee activity spends at most
   `confirm_max_runs` (default 3) inferences, then is marked `unconfirmed` and
   stops. A shadow clip costs ≤ 3 inferences, ever.
4. **Async, off the hot path.** A single worker thread + small bounded queue.
   Capture never blocks on inference — preserves the recorder's "a missed
   detection is permanent data loss, favour over-triggering" stance.
5. **CPU-throttled.** `torch.set_num_threads(1–2)` so inference can't starve the
   capture loop on the 4-core Pi.
6. **Native `imgsz` (~640), not shrunk.** The model expects whole-frame framing;
   small bees in a 1080p frame need the resolution — don't downscale `imgsz` to
   "save cost" or recall collapses. Cost is controlled by *count*, not input size.
7. **Idle = $0.** No motion → no frames → no inference. Model loaded **once**,
   kept warm for the process lifetime.

### Per-activity state machine (in the worker)

```
clip opens ─▶ pending
  each sample (full main_bgr + mover blobs)
     ─▶ YOLO(main_bgr, conf≥CONF) → any bee box overlapping a mover blob?
          yes ─▶ confirmed   (record taxon + max-conf; stop running YOLO)
          no  ─▶ negatives += 1
                  negatives ≥ confirm_max_runs ─▶ unconfirmed (stop)
clip closes ─▶ recording is already DONE + remuxed regardless of verdict; the
               verdict only decides where the finished .mp4 lands (deferred routing)
```

`min_confirmations` (default 1; raise to 2 for stricter) = mover-overlapping bee
detections needed to confirm.

### Recording is never gated by YOLO

This is a hard guarantee, not a tuning choice. The capture loop opens / extends /
closes clips **purely on MOG2 motion**, exactly as today — `BeeConfirmer.submit()`
is fire-and-forget onto a bounded queue and returns in microseconds; YOLO runs in
a **separate worker thread**. A 1–3 s inference can never delay a frame, stall a
clip, or drop a fast bee. Two things reinforce this:

- The **pre-roll ring buffer** (`CircularOutput`, `PRE_ROLL`≈3 s) already records
  the seconds *before* MOG2 trips, so the bee's entry isn't lost to trigger (let
  alone YOLO) latency.
- If the confirmer can't keep up (queue full) it **drops the oldest job**, never
  back-pressures the producer. Worst case is a *missing verdict*, never a missing
  frame.

### The video is never gated — only the automated downstream consumers are

**The clip always records to the watched day dir and uploads over WiFi, exactly
as today**, with the verdict written into a per-clip tag (sidecar). What the
verdict gates is the two **automated, resource-consuming** outputs:

- **Telemetry activity count** — an unconfirmed activity is *not counted as a bee
  activity* in the heartbeat (`hardware/telemetry.py`), so dashboards/automation
  aren't inflated by shadows.
- **BioCLIP crop send** — the `activity_frames/` crops (cellular-capped → automated
  cloud BioCLIP runs) are *not queued* for an unconfirmed activity.

The tag also lets the **cloud** skip the expensive automated **video analysis**
(the SageMaker g4dn endpoint — ~90% of the AWS bill, [[project_aws_cost_audit]] /
[[09_aws_migration_plan]]) for unconfirmed clips, *at the user's discretion* — the
video is uploaded and available, just flagged. So nothing is lost; the costly
automated runs are simply not spent on motion that wasn't a bee.

Because these consumers act at/after clip close while a verdict may still be
`pending` (short clip, slow Pi), the count + crop decision is **deferred, not
blocking**:

1. Clip records + remuxes to the watched dir immediately (always). The per-clip
   tag is written `pending` first.
2. When the verdict **resolves**, a callback finalises the tag and — if confirmed
   — increments the counted-activity tally and flushes the held crop bytes to
   `activity_frames/`. Unconfirmed → tag finalised `unconfirmed`, no count, no
   crop send. (Crops are tiny and already in memory in `act_cands`; only their
   *send* waits, not the video.)
3. **Fail-closed timeout (resource-cautious).** If a verdict never resolves within
   `BEE_CONFIRM_VERDICT_TIMEOUT`, the activity is left **uncounted and its crops
   un-sent** — we don't spend the scarce cellular cap / automated cloud runs on an
   unverified activity. This is safe precisely because the **video is already
   uploaded + tagged**: a real bee whose verdict was merely slow is fully
   recoverable in the cloud from the clip (re-derive count / run BioCLIP on the
   video), without having burned cellular on a guess. An **inert** confirmer
   (missing model/torch) → everything counts + sends as today (`off`).

Net: recording *and the video upload* are MOG2-only and never wait; only the
**automated counts + cloud crop runs** wait on YOLO, and on doubt they hold back
(cheap to recover, since the tagged clip is already up).

---

## 3. What the verdict gates — staged policy (the safety dial)

**The video clip always records + uploads, tagged.** The verdict gates only the
two automated, resource-consuming downstream outputs (§2): the telemetry
**activity count** and the **BioCLIP crop send**. `BEEMONITOR_BEE_CONFIRM_MODE`:

| Mode | Activity count (telemetry) | BioCLIP crop send (cellular→cloud) | Video clip | Use |
|------|---------------------------|-----------------------------------|-----------|-----|
| `off` | every clip | every clip | upload (untagged) | today's behaviour |
| `tag` | every clip | every clip | upload **+ tag** | observe only — measure confirm/reject without changing counts |
| **`gate`** (default) | **confirmed only** | **confirmed only** | upload **+ tag** | filter the automated count + cloud runs |

**Decision: ship `gate` as the default.** Unconfirmed motion no longer inflates
the activity count or spends a cellular crop / automated BioCLIP run — but the
**clip itself is still uploaded, tagged `unconfirmed`**, so (a) nothing is lost
and (b) the user/cloud can *separately* decide whether to run the expensive
automated **video analysis** (g4dn) on it. Safe to ship directly because the gate
only ever withholds *cheap-to-recover* automated work (count + a crop send), never
the video, and on a slow/broken confirmer it **fails closed** (don't spend) or
**inert → `off`** (count + send everything). `off` / `tag` stay one env var away.

Validate **before** rollout with offline `motion_replay --confirm` on known clips
(§4.4) — free, no Pi — to eyeball confirm/reject and tune `CONF` /
`MIN_CONFIRMATIONS`, rather than discovering the false-negative rate live.

### The tag that rides with every uploaded clip
A tiny per-clip sidecar (next to the `.mp4`, e.g. `<stem>.bee.json`) carrying
`confirm_status` (`confirmed`|`unconfirmed`|`pending`|`disabled`),
`bee_confidence`, `taxon`, `confirm_runs`. Consumed by:
- **`hardware/telemetry.py`** — counts only `confirmed` activities in the heartbeat.
- **`hardware/uploader.py`** — ships the clip **regardless** (all clips upload),
  and carries the tag so the cloud sets `Video.metadata.bee_confirmed` and can gate
  its own automated video analysis. *(Exact transport — sidecar upload vs filename
  marker vs an upload field — to confirm against `uploader.py` + the video ingest.)*
- The **activity-frames sidecar** (`*_<i>.json`) also gets the same fields, for the
  crops that do get sent.

---

## 4. Components

### 4.1 `motion/confirm.py` (new) — `BeeConfirmer`
- Lazy, **graceful** init: import `ultralytics`/`torch`, load `bee_tracking.pt`
  once, `torch.set_num_threads(N)`. Any failure (missing dep, model not found,
  SIGILL-class load error) → log once, set itself **inert** (acts like `off`).
  **Never crashes recording** — same contract as the hotel-ROI detector.
- Worker thread draining a bounded `queue.Queue` of
  `(activity_uid, bgr_crop, area)` jobs; drops oldest on overflow (favours
  recency, never blocks the producer).
- `submit(uid, main_bgr, blobs)` — enqueue the **full frame** + the mover blobs
  (`gate.last_blobs`, lores coords) (no-op if the activity is already `confirmed`,
  if inert, or if mode is `off`).
- `verdict(uid) -> {status, confidence, taxon, runs}` — thread-safe read.
- `finish(uid)` — free per-activity state at clip close.
- Internals: `_run(main_bgr, blobs)` → `model.predict(main_bgr, conf=CONF,
  imgsz=IMGSZ, verbose=False)` → scale each bee box to lores and keep it only if
  it overlaps a mover blob (`_bbox_overlap`, reused from `motion/calibrate.py`) →
  best mover-overlapping bee conf + label. (Same model + scale handling as the
  calibration pass and `detect_hotel_roi`.)

### 4.2 `motion/recorder.py` (integration — the only call-site changes)
- Construct `BeeConfirmer(mode)` after `_build_gate(roi)` (skip if `off`).
- In the crop-sampling block (where `act_cands.append(...)` happens today): we
  already hold the full `main_bgr` and `gate.last_blobs` there — call
  **`confirmer.submit(act_uid, main_bgr, gate.last_blobs)`** *and* keep the
  existing `_mover_crop` → `act_cands` for the cloud queue. No re-capture. Skip
  submitting once the activity is `confirmed` (confirm-then-coast).
- In `_close_segment`: remux to the **watched day dir as today** (always — the
  video is never gated). Write the per-clip tag sidecar as `pending`. Then, instead
  of unconditionally flushing crops, **register the activity for verdict
  resolution** (hand the held `act_cands` crops to the confirmer/router and call
  `confirmer.finish(act_uid)`).
- **Verdict resolution (deferred, async)** — when an activity's verdict settles,
  the router callback: finalises the per-clip tag (`confirmed`/`unconfirmed`);
  if `gate` + confirmed → flush crops to `activity_frames/` and mark the activity
  counted; if `gate` + unconfirmed → do neither; if `tag` mode → always flush +
  count, only annotate. A `BEE_CONFIRM_VERDICT_TIMEOUT` sweep finalises stuck
  `pending` activities **fail-closed** (uncounted, un-sent) and logs them.
- This pending-verdict→(tag + count + crop-flush) resolver is a small helper in
  `motion/confirm.py` so `recorder.py` stays a thin orchestrator. **No video
  holding dir, no file re-routing** — only the crop bytes + a counter wait.

### 4.3 `motion/config.py` (new knobs, all `BEEMONITOR_*`)
- `BEE_CONFIRM_MODE` = `off|tag|gate` (**default `gate`**)
- `BEE_CONFIRM_MODEL` (default `models/bee_tracking.pt` — reuse `YOLO_MODEL`)
- `BEE_CONFIRM_CONF` (default 0.30 — a notch above the 0.25 detect default to
  bias against confirming noise)
- `BEE_CONFIRM_IMGSZ` (default 640 — the model's native whole-frame size; do not
  shrink, small bees need the resolution)
- `BEE_CONFIRM_MIN_CONFIRMATIONS` (default 1)
- `BEE_CONFIRM_MAX_RUNS` (default 3) — negative budget per activity
- `BEE_CONFIRM_TORCH_THREADS` (default 1)
- `BEE_CONFIRM_QUEUE_MAX` (default 8)
- `BEE_CONFIRM_VERDICT_TIMEOUT` (default e.g. 20 s) — pending activity → fail-closed
  (uncounted, crops un-sent); the tagged video is the recovery path
  (no video holding dir, no quarantine dir — the video always lands in the watched tree)

### 4.4 Downstream consumers + validation surfacing
- **`hardware/telemetry.py`** — the heartbeat activity count reads the per-clip tag
  and counts **`confirmed` only** (in `gate` mode). This is the "don't inflate the
  automated count" requirement. Untagged/`pending`/`disabled` → not counted in
  `gate` (counted in `off`/`tag`). Robust to a missing tag (treat as not-confirmed
  in `gate`).
- **`hardware/uploader.py`** — uploads **all** clips (confirmed or not); just
  carries the tag so the cloud can flag/route. No dir is skipped.
- The 5-min stats line (`recorder.py`) gains `confirmed/unconfirmed/pending`
  counts + total inferences + mean inference ms — the field evidence for tuning.
- `motion_replay.py` gains an optional `--confirm` flag so a sample clip can be
  scored offline (it already runs YOLO via the calibration path), letting Edward
  eyeball confirm/reject decisions before any field rollout.

---

## 5. Cost model (Pi 4 CPU)

- **Real-bee activity:** ~1 inference (confirm-then-coast).
- **Shadow/non-bee activity:** ≤ `BEE_CONFIRM_MAX_RUNS` (3) inferences, then $0.
- Per **full-frame** inference @ `imgsz=640` on the A72: **needs on-Pi
  measurement** — order ~1–3 s each (the calibration pass already calls this
  "slow on a Pi 4"; nest detection pays it once at startup), all async +
  thread-throttled. It never blocks capture.
- ~50 activities/day, mostly bees → **~50–150 full-frame async inferences/day**.
  Low *count* is what makes this affordable, not low per-run cost.
- **If measured latency is too high,** the dials, in order: `MAX_RUNS=1` /
  `MIN_CONFIRMATIONS=1` (one-shot confirm), confirm a *subsample* of activities,
  or — to be validated, since it shifts input scale — confine YOLO to the
  **hotel-ROI sub-region** of the frame (still a real scene region, not a tight
  mover crop) to cut pixels. Native full-frame stays the default. All env-only.

Memory: one resident YOLO model (~5.4 MB weights + torch). `nest_detection.pt` is
loaded transiently for hotel ROI at startup and dropped, so the confirmer's model
is the only persistent one. **Must be the CPU wheel** (SIGILL note) — the same
constraint the existing startup detection + `--calibrate` already satisfy.

---

## 6. Why on-device (not just lean on cloud BioCLIP)

The cloud already IDs species — but only *after* a crop is sent over (capped)
cellular and an **automated BioCLIP run** has billed, and only after a shadow has
been **counted as an activity** in telemetry. The entire point is to **not spend
the cellular cap, the automated cloud runs, or an inflated count on motion that
wasn't a bee** — while still keeping the video (it uploads + is tagged, so a
missed bee is recoverable in the cloud). Confirmation must therefore be at the
edge. It does **not** replace BioCLIP: YOLO answers the binary "is this a bee?"
cheaply on-device; BioCLIP answers "which species?" in the cloud for the crops
that survive. The new `taxon`/`bee_confidence` fields are a *hint* the cloud can
use or override.

It also doesn't replace **calibration** (`motion/calibrate.py`): that tunes the
MOG2 blob-size window so the cheap gate is *bee-sized*; confirmation verifies the
bee-sized mover is *actually a bee*. Layered, not redundant.

---

## 7. Risks / open questions

- **Per-inference latency on the A72** is the gating unknown — measure first
  (offline `--confirm`, then on-Pi). It can never delay capture; it only delays a
  *verdict*, which deferred + fail-open routing absorbs (a held clip times out to
  the watched dir). High latency degrades *filtering coverage*, not data.
- **YOLO false negatives on tiny/blurred/edge-of-frame bees** → a real bee left
  **uncounted with crops un-sent**. But the **clip is still uploaded + tagged
  `unconfirmed`**, so the count/BioCLIP are *recoverable in the cloud from the
  video* — nothing is lost, only an automated run deferred. Mitigated further by
  offline `--confirm` tuning of `CONF`/`MIN_CONFIRMATIONS` *before* rollout; the
  cloud-side `unconfirmed` tag lets a periodic re-check audit the false-negative
  rate. `tag` (count+send everything, just annotate) is the env-only retreat.
- **CPU contention with capture** — throttle torch threads; if frame drops appear,
  lower `imgsz` or `QUEUE_MAX`, or pin the worker to one core.
- **`bee_tracking.pt` class set** — confirm what it emits (`bee`, `wasp`, …) so
  `taxon` is meaningful and "is a bee" can be a class allow-list, not just "any
  box". Quick check at build time.
- **Two YOLO models on disk already exist** (`nest_detection.pt`,
  `bee_tracking.pt`) — no new asset; the confirmer reuses `bee_tracking.pt`.

---

## 8. Phased execution

Target mode is **`gate`** (the default). Build it whole, but gate the field
rollout on *offline* evidence rather than a long dark-tag phase.

- **Phase 0 — build + offline validation.** `motion/confirm.py` (`BeeConfirmer` +
  the pending-verdict resolver), `motion/config.py` knobs, recorder hooks
  (`submit` + deferred count/crop in `_close_segment`), per-clip tag + sidecar
  fields, stats line, and `motion_replay --confirm`. **Validate offline** on known
  sample clips (`--confirm`, no Pi): does YOLO confirm real-bee clips and reject
  shadow clips? Tune `CONF` / `MIN_CONFIRMATIONS`. *Gate before any device runs it.*
- **Phase 1 — on-Pi shakedown.** Deploy to one unit. Measure per-inference
  latency, queue/pending behaviour, frame-drop impact, and the verdict→
  (tag + count + crop-flush) timing incl. fail-closed timeouts. Confirm
  `telemetry.py` counts confirmed-only and the clip + tag still upload.
- **Phase 2 — cloud + fleet.** Cloud reads the tag → sets `Video.metadata.
  bee_confirmed`, gates its own automated **video analysis** (g4dn) on it, surfaces
  `bee_confirmed`/`taxon` on the dashboard, optional periodic re-check of
  `unconfirmed` clips. Roll out to the fleet.

The `off` / `tag` modes remain one env var away if field data shows `gate` is too
aggressive for a site.
