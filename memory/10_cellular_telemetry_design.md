# 10 — Cellular Telemetry & Heartbeat-Image Design

Status: **design locked, implementation pending**
Scope: field hardware units (`hardware/`) + Django backend (`beemonitor_web/`)
Builds on: `00_implementation_plan.md`, `09_aws_migration_plan.md`, the
motion-gated recorder (`hardware/main_motion.py`).

## 1. Motivation

Field units are moving to a **Sixfab 4G/LTE** cellular link. Moving video over
cellular is too expensive (motion-gated snippets are still 10s–100s of MB/day).
The fix is to **split transport by cost-vs-value**:

- **Cellular** carries only tiny, always-on data so we know the unit is alive.
- **WiFi** carries the bulk video, opportunistically, when it's available.

This makes the cellular bill small and predictable while still giving good
operational visibility into every deployed unit.

## 2. Data classes & transport

| Data | Size | Network | Cadence |
|------|------|---------|---------|
| Telemetry (device health JSON) | ~1 KB | Cellular | Hourly |
| Heartbeat image (1 JPEG, ~720p) | ~250 KB | Cellular | Hourly (one per beat) |
| Full video snippets | GBs | **WiFi only** | Opportunistic; held on disk |

Cost check: in the WittyPi daylight window (~11 h/day) ≈ 11 beats/day ×
~250 KB ≈ **~2.7 MB/day ≈ ~80 MB/month**. Telemetry JSON is noise by comparison.

### Decisions (locked)
- **Every telemetry beat carries one image.** Hourly stills give temporal
  coverage across the active day (lighting, when bees appear, framing drift) and
  double as free spot-checks on the motion gate.
- **No heartbeat video clips.** The hourly image replaces their only purpose
  (auditing that the gate isn't silently missing activity) at ~2% of the bytes.
  The recorder's heartbeat-clip path is disabled by default (`HEARTBEAT_INTERVAL=0`)
  but left in the code for bench/WiFi debugging.
- **Video is WiFi-gated and held on disk; never auto-prune un-uploaded clips.**
  Telemetry reports storage % so the 256 GB card is monitored before it fills.
- **WittyPi 4 battery/voltage telemetry is dropped for now** (add later as a
  best-effort field that never blocks the heartbeat).

## 3. Telemetry payload

All fields cheap to read on a Pi:

- `videos_recorded` (count) + `bytes_pending_upload` + `pending_uploads`
- **`snippets_last_period`** — snippets recorded in the trailing telemetry
  window. A snippet only exists because motion was detected, so this is the
  **activity proxy** (bee activity per period). Reported with
  `telemetry_period_seconds` / `telemetry_period_human` + `last_activity_at`.
- `uptime_seconds` / `uptime_human`
- `storage_pct`, `storage_free_human` (+ raw bytes)
- `cpu_temp_c`
- service health: `recorder_active`, `uploader_active`, `cellular_active`
- `cellular_signal` (RSSI via `qmicli`, best-effort)
- WittyPi **scheduled** on/off window (`schedule_window`)

**On/off semantics:** a device cannot report "off" — it's off. The **server
infers offline** from a missed hourly beat (e.g. no beat in 2× the interval).
The device reports its scheduled window so "off as planned" is distinguishable
from "died unexpectedly".

## 4. Components

### 4.1 Recorder hook (`hardware/main_motion.py`)
- On an hourly tick, capture one downscaled (~720p) JPEG into a `telemetry/`
  queue dir. The recorder **owns the camera**, so it must be what captures the
  still — a separate process would conflict on the device.
- Flip default `HEARTBEAT_INTERVAL` to `0` (no motion-independent video clips).

### 4.2 Telemetry service (NEW — `hardware/telemetry.py`)
- **Long-lived loop service** (`beemonitor-telemetry.service`, Type=simple) — not
  a timer — so the cadence is a single env knob: `BEEMONITOR_TELEMETRY_INTERVAL`
  (3600 prod; **set 60 to test**). Same pattern as `cellular.service`. `--once`
  flag for manual/cron runs.
- Collect metrics, attach the **latest** queued image, POST to the API, prune
  the sent + older queued images.
- The activity window for `snippets_last_period` equals the interval.
- **Posts even if the recorder is dead** — a missing image plus
  `recorder_active=false` is itself the alert.

### 4.3 Uploader (`hardware/uploader.py`)
- Add a guard: only push `.mp4` when a **WiFi** route is actually up. In the
  field it quietly queues video on disk; it drains the moment WiFi appears.
- Telemetry is a separate service and is unaffected by this gate.

### 4.4 Backend endpoint (`beemonitor_web`)
`POST /api/v1/devices/heartbeat`

- `APIView`, `authentication_classes=[DeviceKeyAuthentication]`,
  `parser_classes=[MultiPartParser]`, `throttle_classes=[]`,
  registered csrf_exempt in `apps/api/urls.py` (mirror `uploads/initiate`).
- Multipart: telemetry JSON fields + one image file.
- Store image to S3:
  `get_s3_client().upload_stream("raw-videos", key, file, content_type="image/jpeg")`.
  Key: `users/<owner_id>/devices/<device_id>/heartbeats/<yyyy>/<mm>/<dd>/<uuid>.jpg`.
- Persist a `DeviceHeartbeat` row (new model, FK→`Device`): `created_at`,
  `metrics` (JSON), `image_storage_key`, `storage_pct`, etc.
- `DeviceKeyAuthentication` already bumps `Device.last_seen_at`.
- Surface the latest heartbeat (last-seen, storage %, recent image, services) on
  the existing `/devices/` page.

Rationale for single multipart POST through the API (vs presigned direct-to-S3):
bytes-through-App-Runner only matters for GB-scale video; a 250 KB still is fine
and keeps the Pi flow to one call.

### 4.5 Web UI (browser, App Runner-rendered)

Stack matches the existing app: Django server-rendered templates + Tailwind
(amber/gray theme), `templates/base.html`, same as the current `/devices/`
pages (`apps/devices/templates/devices/`).

**(a) Device list — enhance `devices/list.html` + `DeviceListView`.**
- Replace the bare Active/Revoked badge with an **operational status** computed
  in the view: **Online** (green) if the latest heartbeat is younger than
  2× the telemetry interval, **Stale/Offline** (gray/red) otherwise, plus
  **Revoked** for `is_active=False`.
- Add columns: **Storage %** (small bar), **last heartbeat image thumbnail**.
- Each row links to the new detail page.

**(b) Device detail / dashboard — NEW `devices/detail.html`, `DeviceDetailView`,
url `devices/<pk>/`.** The operator's main view:
- Header: name, location, status badge, last-seen ("3 min ago" / "2 days ago").
- **Latest heartbeat image** (large) — the at-a-glance "is the camera alive and
  framed right" check.
- **Health cards**: storage used/free + % bar, uptime, CPU temp, videos recorded
  + bytes pending upload, cellular signal, per-service health
  (recorder / uploader / cellular = active?).
- **Scheduled window** (WittyPi on/off) with an "off as planned vs unexpected"
  indicator derived from the schedule + missed-beat inference.
- **Image timeline/gallery**: recent hourly stills as thumbnails → lightbox.
  This is the daily time-lapse and the free motion-gate spot-check.
- **Videos from this device**: recent uploaded snippets (count + bytes pending),
  each linking to the existing `/videos/<pk>/` detail + playback page.
- (later) sparklines for storage % / temp over time from `DeviceHeartbeat` rows.

**Existing global video browsing (already built — reuse, don't rebuild):**
- `/videos/` — `VideoListView`, lists all of the user's videos with filters
  (site, year/month/day/hour, search) + bulk actions + CSV export.
- `/videos/<pk>/` — `VideoDetailView`, in-browser playback via presigned URL,
  analysis results, delete.
The device dashboard's "Videos from this device" panel is just a **device-scoped
slice** of this — it links into the same detail pages.

**Device ↔ video linkage (model change needed):** today the Pi upload only
stamps `Video.metadata = {"device_id", "device_name"}` (JSON) — no real relation,
so per-device queries are awkward. Add a nullable FK:
`Video.device = ForeignKey(devices.Device, on_delete=SET_NULL, null=True,
related_name="videos")`. Set it in `apps/api/uploads.py::UploadCompleteView`
(it already holds the `Device`); backfill existing rows from `metadata.device_id`.
Then `device.videos` powers the dashboard panel, per-device counts on the list,
and cross-checks against the telemetry `videos_recorded` field.

**(c) View logic & scoping.**
- All queries owner-scoped:
  `DeviceHeartbeat.objects.filter(device__owner=request.user)`.
- Online/offline computed in the view from `last_seen_at` / latest heartbeat vs
  the reported interval (the DB stores facts; "offline" is a derived view-time
  judgment, not a stored flag).
- Images are shown via short-lived **presigned GET URLs**:
  `get_s3_client().generate_presigned_url("raw-videos", key, expiry_hours=1)` —
  the browser fetches image bytes **directly from S3**, never through App Runner
  (same pattern as video playback in `apps/videos/views.py::VideoDetailView`).

### 4.6 End-to-end data flow

```
── UPLOAD (Pi → AWS) ──────────────────────────────────────────────────────

 Raspberry Pi (field)                              AWS
 ┌────────────────────────┐
 │ recorder → snippets ───┼─ WiFi ─────► S3 raw-videos      (presigned PUT;
 │          → hourly .jpg  │  (uploader)                      bytes go direct,
 │                         │                                  never via Django)
 │ uploader  (WiFi-gated)  │
 │                         │
 │ telemetry → metrics +   │             App Runner (Django)
 │             latest .jpg ┼─ Cellular ─► POST /api/v1/devices/heartbeat
 │ cellular.service        │  Bearer        │  ├─► S3 raw-videos  (heartbeat image)
 └────────────────────────┘  bmk_device_…   │  └─► App DB         (DeviceHeartbeat row,
                                             │                      Device.last_seen_at)
                                             ▼

── VIEW (browser → AWS) ────────────────────────────────────────────────────

 Browser ─ session login ─► App Runner (Django)
   GET /devices/            │  reads DeviceHeartbeat (owner-scoped) from App DB
   GET /devices/<id>/       │  computes online/offline
                            │  signs short-lived presigned GET URLs
                            ▼
 Browser ◄─ image / video bytes fetched DIRECTLY from S3 (presigned GET)
```

**Auto-analysis is unchanged and rides on the video upload.** `uploads/complete`
already auto-spawns a SageMaker analysis job (`apps/api/uploads.py::_enqueue_pi_analysis`,
gated on `settings.SAGEMAKER_ENDPOINT_NAME`). Since video is now WiFi-gated,
**analysis is deferred to WiFi time** — "recorded" and "analyzed" decouple; the
hourly cellular telemetry covers health in the gap. Telemetry beats do **not**
trigger analysis.

**The split, stated plainly:**
- **Video** → WiFi → **presigned PUT direct to S3** (bytes never touch App Runner).
- **Telemetry + hourly image** → cellular → **multipart POST to App Runner**, which
  writes the image to S3 and a row to the app DB (tiny, so through-Django is fine).
- **Browser** talks only to App Runner for **HTML + JSON + URL signing**; all
  **media bytes are pulled straight from S3** via presigned GET. App Runner never
  proxies large media in either direction.

## 5. Backend integration points (verified in repo)

- Device auth: `apps/api/authentication.py::DeviceKeyAuthentication`
  → `request.auth = Device`, `request.user = owner`.
- S3 helper: `config/storage.py::get_s3_client()` →
  `S3StorageClient.upload_stream(container, key, file, content_type=...)`.
- Bucket aliases: `"raw-videos"`, `"processed"`, `"models"`, `"user-configs"`.
- URL + view patterns: `apps/api/uploads.py` (Pi upload) is the closest template.
- `Device` model: `apps/devices/models.py` (has `owner`, `last_seen_at`,
  `is_active`, `key_hash`).

## 6. Pi configuration knobs (env, `BEEMONITOR_*`)
- `BEEMONITOR_TELEMETRY_INTERVAL` — beat cadence + activity window, seconds
  (default **3600**; set **60** for testing). Drives telemetry.py's loop.
- `BEEMONITOR_TELEMETRY_IMAGE_INTERVAL` — recorder still cadence (default 3600;
  match the telemetry interval, e.g. 60 in testing).
- `BEEMONITOR_TELEMETRY_QUEUE` — image queue dir (default `<RECORD_DIR>/../telemetry`)
- `BEEMONITOR_TELEMETRY_IMAGE_HEIGHT` — still downscale height (default 720)
- `BEEMONITOR_SCHEDULE_WINDOW` — WittyPi on/off window string (optional)
- `BEEMONITOR_HEARTBEAT_INTERVAL` → default **0** (video heartbeat clips off)
- `BEEMONITOR_WIFI_ONLY_VIDEO` (default true) — uploader WiFi gate
- existing: `BEEMONITOR_API_BASE`, `BEEMONITOR_DEVICE_KEY`, `BEEMONITOR_RECORD_DIR`

**Testing tip:** set both `BEEMONITOR_TELEMETRY_INTERVAL=60` and
`BEEMONITOR_TELEMETRY_IMAGE_INTERVAL=60` to get a beat + fresh image every minute.

## 7. Build order (status)
1. ✅ **Backend API:** `DeviceHeartbeat` model + migration → `/api/v1/devices/heartbeat`
   (multipart → S3 + row). Verified end-to-end against real S3.
2. ✅ **Device ↔ video link:** `Video.device` FK + migration + backfill; set in
   `UploadCompleteView`.
3. ✅ **Backend UI:** `DeviceDetailView` + `devices/detail.html` (health cards incl.
   Activity, latest image, image timeline, videos-from-this-device), enhanced
   `devices/list.html` (online/offline + storage % + thumbnail + count),
   presigned GET images, `devices/<pk>/` url, admin. Pages render 200.
4. ✅ **Pi telemetry:** `hardware/telemetry.py` (loop service) +
   `beemonitor-telemetry.service`. Verified Pi-script → endpoint → S3.
5. ✅ **Recorder:** hourly telemetry-still hook; default `HEARTBEAT_INTERVAL=0`.
6. ✅ **Uploader:** WiFi gate (`_wifi_connected`, `BEEMONITOR_WIFI_ONLY_VIDEO`).
7. ✅ **README:** split-transport model, telemetry service, device-monitoring
   section, knobs table, cellular section reconciled (telemetry/cellular,
   video/WiFi), testing stages updated.

**All build steps complete.** Verified: backend e2e against real S3, Pi
telemetry.py → endpoint → S3, dashboard renders. Remaining = deploy (run
migrations on prod, install the telemetry service on the Pi) + on-hardware
verification of the recorder still-capture (`capture_array("main")` format).

## 8. Open items / future
- WittyPi 4 battery/voltage telemetry (best-effort, later).
- Server-side offline detection + alerting on missed beats.
- Retention policy for the on-disk video backlog (warn-only for now; never prune
  un-uploaded).
