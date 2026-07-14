# Platform Performance Audit & Fix Plan (v2 — IMPLEMENTED 2026-07-13)

**Symptom:** Site-wide slowness and intermittent `upstream request timeout` (App Runner 504)
since ~2026-07-09, right after the device-page foraging chart shipped (`7a426a5`, `efad597`).

**Status:** v2 design implemented in full on 2026-07-13 (web + hardware). v1's
mechanisms (ThreadPool recompute hook, staleness scan, DB cache table, quantized
cache keys) were dropped after an adversarial review found simpler, safer shapes —
§4 describes what actually shipped. Hardware changes ride the next device artifact.

---

## 1. Environment constraints

| Constraint | Value | Consequence |
|---|---|---|
| App Runner request timeout | **120 s, hard, not configurable** | Any request slower than 2 min → "upstream request timeout". |
| Gunicorn (before) | 2 sync workers, 0 threads, `--timeout 1800` | Two slow requests froze the platform; dead requests pinned workers past the proxy's 504. |
| Django cache | No `CACHES` → LocMemCache | Per-process, wiped per deploy. Kept (deliberately — see §4.6): nothing needs a cross-worker cache anymore. |

## 2. Issues found (all fixed; kept for the record)

1. **CRITICAL — 5s poll recomputed the full activity series** (chart + trips + weather) on every tick (`detail.html` `setInterval(poll, 5000)` → `DeviceStatusView` → `_build_activity_series`).
2. **CRITICAL — trips cache never hit**: cache key embedded `timezone.now()`-derived timestamps, so every poll re-ran the full S3 aggregation.
3. **CRITICAL — trip aggregation downloaded up to 800 events CSVs from S3, serially, in-request** (`_device_forage_trip_times`). One open device tab = both workers pinned = everything else queued past 120s.
4. **HIGH — no shared cache** (LocMem per worker) made the dead cache double-broken.
5. **HIGH — no request headroom**: 2 sync workers.
6. **MEDIUM — batch page repeated the S3 pattern** uncapped (user-triggered).
7. **MEDIUM — `DailyForagingSummary` only computed at container start** (skip-existing), so it was always stale and nothing trusted it.
   **7b. HIGH (correctness) — device chart silently truncated trips to the OLDEST 800 videos** (`order_by video_id` + cap), while entries/exits were uncapped → recent days showed entry/exit peaks with zero trips; batch page and device page disagreed because both recomputed independently.
8. **LOW — weather fetched in-request** (kept: LocMem 1h is fine at 2 fetches/hour/device once polling stopped).
9. **LOW — startup backfills compete with traffic after deploys** (reduced: the entrypoint backfill remains ONE deploy for the trips-JSON migration, then can be removed — see §6).

## 3. Assets built on

- `DailyForagingSummary` per `(user, site_name, device, date)` (`apps/analysis/models.py:113`).
- `aggregate.aggregate_trips(sources, min_sec, max_sec, events=None)` — the single pairing implementation (batch page already used it).
- The background reconciler daemon (`apps/analysis/reconcile.py`, 120s tick) — the only convergence loop; job completion has exactly one success path (`_apply_result_to_job`).

## 4. What shipped (the v2 design)

**4.1 Trips live in the DB.** `DailyForagingSummary.trips` (JSONField) stores the day's
trips paired UNFILTERED (0–86400s) as compact rows `[exit_epoch, duration_sec, nest,
is_cross_video]`, capped 10k/day. Both pairing implementations consume the Exit on
every Entry regardless of bounds, so **bounds are a pure read-time filter on
duration** — proven by test (`apps/analysis/test_foraging.py::BoundsArePureFilterTest`).
Row stats + the downloadable day CSV keep the default 10/7200 bounds (unchanged meaning;
`total_trips != len(trips)` by design). The duplicate `_compute_daily_trips` pairing
was deleted — `compute_daily_trips` now delegates to `apps/analysis/foraging.py`,
which calls `aggregate.aggregate_trips`. **One pairing implementation everywhere =
the batch↔device sync guarantee.**

**4.2 Device chart trips = pure DB read.** `foraging.device_trips(device, since,
until, min_sec, max_sec)`: summary rows fetched with a ±1-day pad (UTC date keys vs
display-tz window), trips filtered by exact exit instant + bounds. The old
`_device_forage_trip_times` (broken cache key, oldest-800 truncation) is **deleted**.
**No request path reads S3 for the chart, ever** — including today, which shows the
last-computed summary (staleness ≤ ~one reconciler tick).

**4.3 Freshness = dirty flag + existing reconciler.** `_apply_result_to_job` (the
single completion path, reached by both browser pollers and the reconciler) does one
cheap upsert: `stale=True, stale_marked_at=now` for that (user, site, device, day) —
never S3, never raises, skips `device IS NULL` groups (Postgres NULLs break the
unique-row upsert guarantee; the chart never reads them). The reconciler tick calls
`foraging.sweep_stale()` — recomputes up to 4 rows/tick where `stale OR trips IS
NULL`, **newest date first**, with a guarded clear
(`filter(pk, stale_marked_at=seen).update(stale=False)`) so a re-mark landing
mid-recompute survives to the next tick. Lock-free, idempotent, multi-instance safe;
a 639-video batch debounces to one recompute per device-day per tick.

**4.4 Chart split from the 5s poll.** `DeviceStatusView` builds the activity series
only when `?chart=1`. The page polls tiles every 5s; the chart refreshes on load
(server-rendered), on any param change, and every 12th tick (~60s), with a request
sequence number so a stale in-flight response can't overwrite a newer view.

**4.5 Trip bounds UI on the device page** (parity with batch): min/max-seconds inputs
→ `activityQuery()` → both device views → the read-time filter. Both pages share
`aggregate.clamp_trip_bounds` (the batch page's `_trip_bounds` now delegates to it).

**4.6 Capacity.** gunicorn `--threads 4` (gthread) + `--timeout 115` (under the
proxy's 120s so dead requests don't pin threads); `CONN_MAX_AGE=60` in production
(env-overridable via `DB_CONN_MAX_AGE`); `SlowRequestLogMiddleware` logs any request
>5s. **LocMemCache kept** — with trips in the DB nothing needs cross-worker caching;
the v1 DB-cache-table phase was dropped as unnecessary complexity.

**4.7 Batch page hardening.** In-page aggregation capped at 300 videos with a visible
notice (CSV downloads stay uncapped); events CSVs (small, immutable) are memoized 6h
per blob in `read_processed_csv(use_cache=True)` via `collect_events`.

## 5. Feature: WiFi video upload opt-in (SHIPPED; default = manual for EVERYONE — Edward's call)

**Problem:** the uploader pushed every pending video the moment any WiFi was up — and
"WiFi" includes a phone hotspot, silently burning the user's cellular plan.

**Shipped design:**
- `Device.video_upload_mode` (`manual` default | `auto`), pushed in BOTH the heartbeat
  and command-poll responses.
- Device page control (manager+): auto/manual toggle with hotspot warning + **"Upload
  now"** button (shown in manual mode, with the pending count) →
  `pending_command="upload_videos"`.
- Device side: telemetry `_apply_upload_mode` writes
  `RECORD_DIR.parent/video_upload_mode.json`; the `upload_videos` command touches an
  `upload_now` trigger file. The uploader re-reads the mode every ~30s cycle;
  in manual mode videos hold (frames/telemetry unaffected). The trigger is deleted
  only after a pass ends with **nothing pending** (crash mid-drain resumes on
  restart; `.uploaded` sidecars make re-drains idempotent), and a spent trigger is
  cleaned when the backlog is empty so it can't act as permanent auto.
- **Missing state file = manual** (safe: uploading needs the same API a heartbeat
  does, so any unit that can upload learns its real mode first). Reflash caveat: a
  re-imaged unit holds videos until its first beat.
- **Rollout consequence (accepted):** when the next device artifact reaches the
  fleet, all uploads pause until devices are flipped to auto (or drained manually)
  from their pages. `pending_command` is a single slot — a later command replaces a
  queued upload-now.

## 6. Rollout

1. Web deploy: everything in §4 + §5's cloud side. The entrypoint still runs
   `compute_daily_trips` at startup — it now backfills rows whose `trips` JSON is
   missing (newest-first sweep covers the tail) → **run one deploy, then the
   backfill lines can be removed from `entrypoint.sh` in a follow-up.**
2. Device artifact: telemetry + uploader changes (upload mode, upload-now, serial
   metric). Needs a hands-on Pi pass (state file written on beat, hold/drain
   behavior, serial reported) before fleet update.
3. After deploy, verify: App Runner p95 latency + 5xx drop; `beemonitor.slow` log
   lines identify anything still >5s; S3 GET request count falls sharply.

## 7. Feature: Pi serial number on device page (SHIPPED)

- `hardware/telemetry.py` `_serial_number()`: `/proc/device-tree/serial-number`
  (NUL-stripped), fallback `/proc/cpuinfo` `Serial:` line; cached per process;
  merged into beat metrics beside the MAC.
- `detail.html` shows `Serial` next to MAC: `metrics.serial` with
  `device.hw_id` (already captured at zero-touch enrollment) as fallback, so
  enrolled units show it even before the new artifact lands.

## 7b. Feature: RTC internet-time sync (SHIPPED device-side, same artifact)

The WittyPi restores the system clock from its RTC on every wake (before any
network), so RTC drift shifts the local-time wake window and early clip
timestamps. Cellular units are extra exposed (long offline stretches; NTP is
allowed through the firewall but only runs while the link is up).

- After every successful beat (proof a WiFi/cellular link is up), telemetry
  checks `timedatectl NTPSynchronized`:
  - **not synced** → `sudo -n timedatectl set-ntp true` (idempotent nudge; new
    sudoers line in `provision/sudoers.d/beemonitor-timedatectl`, installed by
    the self-provisioning pass) and retry next beat;
  - **synced** → write system time into the WittyPi RTC via utilities.sh
    `system_to_rtc`, throttled to `BEEMONITOR_RTC_SYNC_SECONDS` (default 1 h);
    the first beat after every boot/wake always writes.
- The clock is NEVER written to the RTC unless NTP reports synchronized — a
  stale write would bake drift in instead of fixing it. No WittyPi → no-op.
- Beats now report `ntp_synced` in metrics for drift visibility.

## 8. Verification done at implementation time (2026-07-13)

- `manage.py check` + `makemigrations --check` clean; new migrations
  `analysis/0010`, `devices/0023`.
- 7 new tests green (`apps.analysis.test_foraging`): bounds-as-pure-filter parity,
  clamp behavior, device_trips window/bounds/NULL-row handling, stale-flag
  lifecycle incl. sweep. Full suite: no new failures (3 pre-existing discovery
  errors in `devices/monitor/videos.tests` confirmed identical on clean HEAD).
- End-to-end smoke (test client): device page renders bounds inputs + serial +
  upload control + pending count; tiles poll carries NO chart; `?chart=1` returns
  the series with bounds applied at read time (1 trip at 10/7200 vs 2 at 0/7200
  from stored compact rows, no S3); upload-mode and upload-now endpoints flip the
  model + queue the command.
- `hardware/telemetry.py` + `uploader.py` compile; on-Pi behavior still needs the
  hands-on pass (§6.2).
