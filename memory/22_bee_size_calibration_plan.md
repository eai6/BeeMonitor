# Plan: Fix & harden on-device bee-size auto-calibration + add transparency

> Status: IMPLEMENTED (commit ed186fc, 2026-06-27). Web half deployed via App Runner.
> Firmware half (config.py/calibrate.py/telemetry.py) reaches FIELDED devices via the
> signed edge-artifact remote update — NOT a golden-image rebuild. The artifact bundles
> the whole hardware/ tree (build-edge-artifact.sh:33) and update.sh restarts
> recorder/uploader/telemetry with health-check + auto-rollback (update.sh:37,259,270).
> Golden image is only for flashing NEW SD cards. Rollout: build+sign artifact (minisign
> key), publish, trigger the fleet-update on the devices page. Constants are env-tunable.

## Context

Users see clips that **end while a bee is still moving in frame**. Root cause
(audited): the on-device auto-calibrator (`hardware/motion/calibrate.py`) learns
a blob-area window `[min_area, max_area]` from the 5th/95th percentile of MOG2
blob areas overlapping YOLO bee boxes. At runtime the recorder rejects any blob
outside that window (`gate.py:93-98`); when **all** of a bee's blobs are rejected,
`motion=False`, and after `POST_ROLL=4s` of "no motion" the clip closes as "idle"
(`recorder.py:375`) even though the bee is visible.

The learned window goes wrong because:
1. **False positives** — calibration trusts YOLO at conf **0.25** (`config.py:163`),
   so shadows/leaves/debris get measured as "bees".
2. **Too few samples** — it commits a window from as few as **12** blob samples
   (`CALIB_MIN_SAMPLES`), so the percentiles are noisy.
3. **`max_area` too small for real bees** — MOG2 fragments a bee into many small
   blobs; the 95th percentile sits near small-fragment size, so a close/fast bee
   that yields one large/merged blob exceeds `max_area` and is rejected. (Runtime
   accepts per-fragment blobs with `min_blobs=1`, so the window must span the full
   fragment↔merged range.)
4. **No sanity clamp** — a bad run fully governs the recorder until the next weekly
   recalibration.

Separately, the user wants **transparency**: the effective calibration window and
the per-clip bee-detection **confidence** are invisible on the dashboard today
(confidence is only in a tooltip; calibration requires SSHing the Pi).

Also requested: raise the on-device **bee-confirmation** confidence so clips aren't
tagged as containing a bee on weak detections.

## Already changed this session (firmware, uncommitted — keep)

- `config.py`: `YOLO_CONF` 0.25→**0.5** (calibration-only; verified not shared with
  the confirmer), `CALIB_MIN_SAMPLES` 12→**30**, `CALIB_TARGET_SAMPLES` 40→**100**,
  `CALIB_MAX_CLIPS` 20→**60**; `BEE_CONFIRM_CONF` 0.30→**0.5**.
- `telemetry.py`: `_read_motion_calibration()` + `m["motion_calibration"]` in the
  heartbeat (reports the learned window + age, read-only).

## Remaining work

### 1. Calibration robustness — pad + clamp the learned window (`calibrate.py`)
After computing `p5/p95` (now from ≥30 cleaner samples), widen and clamp so real
bees across the fragment↔merged size range aren't clipped:
- `min_area = max(AREA_FLOOR, p5 * MIN_PAD)` (e.g. `MIN_PAD=0.6`, `AREA_FLOOR≈8`)
- `max_area = min(MAX_BLOB_AREA_DEFAULT, p95 * MAX_PAD)` (e.g. `MAX_PAD=1.6`)
- guarantee `max_area >= min_area * MIN_SPAN` (e.g. 4×) so the window can't collapse.
- Record both raw `p5/p95` and the final padded/clamped values in `calibration.json`
  (alongside the existing `n_samples`/`n_clips`) for transparency.
- New tunables as `BEEMONITOR_CALIB_*` env consts in `config.py` (match existing
  `_env_float` pattern), so they're overridable without a code change.

### 2. Dashboard: show the learned calibration window (`apps/devices/`)
- `DeviceDetailView.get_context_data` (`views.py:558`) already reads
  `device.heartbeats.first().metrics`; expose
  `ctx["motion_calibration"] = (latest.metrics or {}).get("motion_calibration")`.
- Render it read-only in the **Motion tuning** card (`detail.html:435-475`, which
  already says "blank = use the device's auto-calibration"): show learned
  `min_area`/`max_area`, `n_samples`, and age, with a hint when it's stale or
  derived from few samples. No new endpoint — it rides the existing heartbeat.

### 3. Dashboard: surface per-clip bee-detection confidence (`apps/*/templates`)
The data already exists — uploads store `metadata.bee = {status, confidence, taxon,
runs, mode}` (`apps/api/uploads.py:205-208`); it's only shown in a tooltip today
(`processing.html:153-156`).
- **Video detail** (`apps/videos/templates/videos/detail.html`): add a small
  "Bee confirmation" row near the status block — status, confidence (as %), taxon.
- **Processing list** (`processing.html`): show the confidence inline next to the
  `✓ bee` / `unconfirmed` badge, not just in the title tooltip.
- Pure display; no model/DB change (reads `video.metadata.bee.*`).

### 4. Tests
- Web: extend `apps/devices/tests.py` — a heartbeat carrying `motion_calibration`
  renders on the detail page; a `Video` with `metadata.bee` shows confidence on the
  detail + processing pages (follow the `force_login` + `assertContains` pattern
  already there).
- Firmware: no test harness exists; rely on `py_compile` + a small pure-Python
  check of the pad/clamp math if feasible, otherwise document manual verification.

## Verification
- `python manage.py test apps.devices.tests apps.videos.tests` (web).
- `python -m py_compile hardware/motion/calibrate.py hardware/motion/config.py hardware/telemetry.py`.
- Manual/device (after firmware roll): on a Pi, `cat ~/Desktop/cameraOutput/calibration.json`
  shows raw + padded/clamped window; the device dashboard shows the same window and
  per-clip confidence — no SSH needed thereafter.

## Deploy / rollout
- **Web** (dashboard display): auto-deploys via App Runner on merge to `main`.
- **Firmware** (`config.py`, `calibrate.py`, `telemetry.py`): takes effect only after
  the **golden-image rebuild / signed edge-artifact** update and device check-in
  (the hands-on step from `project_next_session`). Dashboard display degrades
  gracefully for devices on old firmware (no `motion_calibration` key → just hidden).

## Open question for approval
Calibration aggressiveness: the plan **pads + clamps** the learned window (recommended,
directly fixes "ends while moving"). Alternative is thresholds-only (cleaner samples
but the percentile tails can still clip large/merged bee blobs). Pad/clamp constants
above are starting points and env-tunable.
