# 12 — Device Dashboard & Telemetry v2

> **2026-06-05 simplification (field feedback):** trimmed to what's proven —
> removed **live view (5b)** + **WiFi stream (5c)**, and the **cellular-signal +
> GPS/Location widgets** (modem-status path was unverified). The recorder MJPEG
> server and `cellular-up.sh` modem-status probing were reverted. **Kept:**
> online=freshness, JSON 60s beat, activity card + graph, **take-photo on demand**
> (now faster via a `GET /api/v1/devices/command` poll every ~8s instead of
> waiting for the 60s beat). DB columns (stream_url, last_lat/lon, …) left in
> place (harmless, no drop migration). See commit after a95a5aa.

Status: **planned (not yet implemented)**
Scope: Pi (`hardware/`), backend (`beemonitor_web/apps/{devices,api,videos}`), web UI
Builds on: `10_cellular_telemetry_design.md`, `11_cellular_boot_ordering_cycle_fix.md`

## Motivation

Field experience surfaced gaps in the telemetry/dashboard built in doc 10:
- "Online" only reflects `is_active`, not whether the unit is actually reporting.
- Images in every beat are wasteful on cellular and not needed for liveness.
- No way to see live framing during field setup without pulling video.
- Cellular signal widget is blank; no GPS; activity window too short to be useful.
- No history view of activity over time.

## Changes (8)

### 1. Remove the "Sources" nav menu
- Drop the `Sources` link from `templates/components/navbar.html` (desktop **line 13**
  and mobile **line 45**). Leave the `sources` app/routes intact — just hide the nav.

### 2. Online = telemetry freshness (not is_active)
- A device is **online** only if its last beat arrived within a freshness window.
  With a 60s beat, treat **offline after ~3 missed beats (180s)**.
- Make it configurable: `settings.DEVICE_ONLINE_GRACE_SECONDS` (default 180).
- `devices/views.py::_is_online` currently uses `TELEMETRY_INTERVAL_SECONDS(3600)×2`.
  Retune to the new constant. `last_seen_at` is bumped by `DeviceKeyAuthentication`
  on every device call; since telemetry beats every 60s it's the freshness signal.
- Applies to both list badge and detail header. Still show **Revoked** for
  `is_active=False` (distinct from offline).

### 3. Telemetry = JSON-only @ 60s (drop images)
- `hardware/telemetry.py`: keep the 60s beat (fast offline detection) but **stop
  attaching an image** to the regular beat. Drop `_latest_image`/image POST on the
  normal path.
- `hardware/main_motion.py`: disable the periodic still capture
  (`BEEMONITOR_TELEMETRY_IMAGE_INTERVAL` default → **0**). Stills are now captured
  **on demand only** (see #4).
- Backend heartbeat endpoint keeps multipart/image support (used by on-demand).

### 4. On-demand camera: picture **and** live streaming (field setup)
- **Decision:** build BOTH, both command-driven and bounded (used briefly during
  setup / occasionally). Picture-on-demand first; live view second.
- **Command channel (device is outbound-only / behind cellular NAT):** the device
  polls implicitly via its 60s beat. The heartbeat **response** carries a pending
  command:
  - Backend: `Device.pending_command` (`""` | `"capture_image"` | `"stream"`) +
    optional params (e.g. stream duration/fps), set by dashboard buttons
    (`POST /devices/<pk>/request-image`, `POST /devices/<pk>/request-stream`).
    The heartbeat endpoint returns `{"command": ..., "params": {...}}` when set,
    then clears it.
  - Recorder↔telemetry handoff: telemetry drops a sentinel file (e.g.
    `<queue>/capture.request` / `stream.request`); the recorder (which owns the
    camera) services it, writes JPEG(s) to the queue, telemetry uploads + deletes.
    Avoids two processes contending for the camera.

- **(a) Picture-on-demand** — single still on command, uploaded via the heartbeat
  endpoint's image path. Latency ≤ one beat (~60s; drop the interval for setup).

- **(b) Live streaming** — bounded "live view," two tiers:
  - **Cheap/anywhere (default):** on `stream`, the recorder emits a short burst of
    rapid stills (e.g. ~1–2 fps for N seconds, capped) that upload as they're
    taken; the dashboard auto-refreshes the latest image → a near-live view that
    works over cellular and needs no media server. Bandwidth-bounded by fps ×
    duration; surfaced as the "Live view" button.
  - **True low-latency (WiFi, later):** real MJPEG/WebRTC stream when the unit is
    on WiFi (the recorder can serve an MJPEG endpoint, or WebRTC via a TURN
    relay). Heavier; gate on WiFi. Track as a follow-up; the rapid-stills tier
    covers field setup in the meantime.
  - Both are **bounded** (auto-stop after the duration) so a forgotten stream
    can't run up cellular cost.

### 5. Cellular widget fix (root cause)
- The telemetry service runs as **`beemonitor`**, which can't read `/dev/cdc-wdm0`
  → `qmicli` returns nothing → blank widget. Same blocker hits GPS.
- **Fix (also enables #8):** `cellular-up.sh` (runs as **root**, owns the modem)
  writes a small status file each watchdog cycle — signal + GPS — and
  `telemetry.py` (beemonitor) just **reads the file**. No privileged access needed
  in telemetry.
  - File: `/run/beemonitor/modem-status.json` → `{ "rssi_dbm": -71, "lat": ...,
    "lon": ..., "fix": true, "ts": ... }`.
  - Signal: `qmicli -p -d /dev/cdc-wdm0 --nas-get-signal-strength` (parse dBm).

### 6. Activity period configurable (default 3600s), decoupled from beat
- `snippets_last_period` currently uses the telemetry interval as its window — now
  60s, far too short to show activity. **Decouple:** new
  `BEEMONITOR_ACTIVITY_PERIOD` (default **3600**), independent of
  `BEEMONITOR_TELEMETRY_INTERVAL` (60). Telemetry reports snippets in the trailing
  activity window each beat; `telemetry_period_*` fields reflect it.

### 7. Activity line graph on the device page
- Show snippets/hour over time (auto-scale: hourly for days, daily for weeks/
  months) from stored `DeviceHeartbeat` rows.
- Data: each beat stores `snippets_last_period` (rolling 1h count). Aggregate to a
  series by bucketing heartbeats per clock-hour and taking the **max** in each
  bucket (the rolling count sampled once/hour ≈ that hour's activity). For wide
  ranges, bucket per day.
- Backend: a JSON endpoint or view-context series, owner-scoped:
  `DeviceHeartbeat.objects.filter(device=...)` grouped by hour/day.
- Frontend: **Chart.js via CDN** (no build step; matches the Tailwind+template
  stack) — a line chart on `devices/detail.html`. Range selector (24h / 7d / 30d).
- Note: 60s beats = 1440 rows/device/day. Fine near-term; add downsampling/pruning
  of old heartbeats later if it grows (track as future).

### 8. GPS coordinates from the modem (GNSS) in telemetry
- Quectel modem has GNSS. Enable once (`AT+QGPS=1`); read position
  (`AT+QGPSLOC=2` on the AT port, e.g. `/dev/ttyUSB2/3`) — done by **root** in
  `cellular-up.sh`, written into the modem-status file (#5). `telemetry.py` reads
  `lat`/`lon` and includes them in the beat.
- The Pi always includes `lat`/`lon` in the beat (when it has a fix). Backend
  **always** updates `Device.last_lat/last_lon/last_fix_at`, and **additionally**
  stores per-beat GPS on `DeviceHeartbeat` when
  `settings.DEVICE_STORE_GPS_PER_HEARTBEAT` (env, **default True**) is on — so a
  location history/breadcrumb is available. Flip the env to **False** later to keep
  only latest-on-device (no per-row GPS) per the owner's preference.
- Detail page shows coordinates + a map link (OpenStreetMap/Google Maps URL); the
  per-heartbeat history can later drive a location trail.
- **Pi-verification needed:** GNSS antenna attached, GNSS enabled, outdoor fix can
  take minutes. Implement best-effort (omit when no fix).

## New config knobs (`BEEMONITOR_*` / Django settings)
- `BEEMONITOR_TELEMETRY_INTERVAL` = 60 (beat cadence; stays 60)
- `BEEMONITOR_ACTIVITY_PERIOD` = 3600 (activity window; NEW, decoupled)
- `BEEMONITOR_TELEMETRY_IMAGE_INTERVAL` → 0 (periodic still off)
- `BEEMONITOR_MODEM_STATUS_FILE` = /run/beemonitor/modem-status.json (NEW)
- `BEEMONITOR_STREAM_FPS` = 1.5, `BEEMONITOR_STREAM_MAX_SECONDS` = 60 (NEW, live view caps)
- `settings.DEVICE_ONLINE_GRACE_SECONDS` = 180 (NEW, backend)
- `settings.DEVICE_STORE_GPS_PER_HEARTBEAT` = True (NEW, backend env; False = latest-on-device only)

## Backend model changes
- `Device.pending_command` (CharField, blank) + optional `command_params` (JSON) —
  picture/stream on demand.
- `Device.last_lat` / `last_lon` / `last_fix_at` (nullable) — last known GPS (always updated).
- `DeviceHeartbeat`: add `lat`/`lon` (nullable) — per-beat GPS, written only when
  `DEVICE_STORE_GPS_PER_HEARTBEAT` is on. Migration required.

## Build order (batches) — STATUS
1. ✅ **Quick wins:** remove Sources nav (#1); online=freshness + configurable (#2).
   — commit `cca206b`.
2. ✅ **Telemetry cheapening:** JSON-only @60s, drop images (#3); activity period
   decoupled `BEEMONITOR_ACTIVITY_PERIOD` (#6); recorder periodic still→0.
   — commit `6c2a0cd`.
3. ✅ **Modem status:** `cellular-up.sh` writes signal+GPS to `/run/beemonitor/
   modem-status.json`; `telemetry.py` reads it (#5, #8). Device GPS + per-heartbeat
   GPS; dashboard cellular + Location card. Migration 0003. — commit `c7154ee`.
   **(Pi-verify: qmicli parse, AT port, GNSS fix.)**
4. ✅ **Activity graph (#7):** `snippets_last_period` column (migration 0004) +
   ORM TruncHour/TruncDay aggregation + Chart.js (24h/7d/30d/90d). — commit `3ebb0de`.
5. ✅ **On-demand camera (#4):** request-image/request-stream endpoints +
   latest-image.json; heartbeat-response command; Pi capture via sentinel +
   `send_beat(image=)`; dashboard buttons + polling. — commit `358606a`.
   - 5a ✅ picture-on-demand (single still).
   - 5b ✅ live view via bounded rapid-stills (`BEEMONITOR_STREAM_FPS`/`_MAX_SECONDS`).
   - 5c ✅ live LAN MJPEG stream: recorder serves a bounded on-demand MJPEG
     server (BEEMONITOR_STREAM_PORT 8090); telemetry advertises the LAN URL;
     dashboard "WiFi stream" button + link. LAN-only (open on same network / via
     Pi Connect). Device.stream_url/stream_expires_at, migration 0005.
     **EXPERIMENTAL — Pi-verify.** (WebRTC for cloud-reachable real-time = future.)

**All planned batches complete (2026-06-05). Remaining: on-Pi verification of the
modem-status/GPS path + on-demand capture; and 5c if real-time streaming is wanted.**

## Open / verify
- GPS sourcing path (AT+QGPSLOC vs qmicli loc service vs gpsd) — using Quectel AT
  via root in cellular-up.sh; verify on hardware. (okay)
- Cellular signal + GPS both depend on the modem-status-file approach working on
  the Pi (permissions/AT access).
- Heartbeat row volume at 60s — revisit retention/downsampling for the graph.

## Decisions (resolved)
- **#4 scope:** build BOTH picture-on-demand AND live streaming, command-driven and
  bounded (brief field-setup / occasional use). Live view ships as bounded
  rapid-stills (cheap, works on cellular) now; true low-latency MJPEG/WebRTC over
  WiFi is a follow-up (5c). **Yes, we can "live stream"** for setup via the
  rapid-stills view immediately; WebRTC adds true real-time later.
- **GPS storage:** per-heartbeat GPS, **env-configurable**
  (`DEVICE_STORE_GPS_PER_HEARTBEAT`, default True) so it can drop to
  latest-on-device only later.
