# BeeMonitor Cloud Platform — Staged Implementation Plan

## Overview

10 stages, ordered by dependency. Each stage is self-contained, testable, and builds on the previous. The existing `src/beemonitor/` package stays **untouched** — we build around it.

**New directory structure:**
```
BeeMonitor_eai6/
├── src/beemonitor/            # EXISTING - no changes
├── cloud/                     # Stages 1-3: Modal backend + Azure storage
│   ├── storage/
│   ├── wrapper/
│   ├── modal_app/
│   ├── ingestion/
│   └── scripts/
├── beemonitor_web/            # Stages 4-8: Django project
│   ├── config/
│   ├── apps/
│   ├── templates/
│   └── static/
└── beemonitor-mobile/         # Stages 9-10: React Native app
    ├── app/
    └── src/
```

---

## STAGE 1: Azure Blob Storage + Cloud Wrapper

**Objective:** Azure Blob Storage as canonical data layer + thin wrapper around `beemonitor` for cloud invocation.

**Dependencies:** None (foundation)

**Deliverables:**
- `cloud/storage/azure_client.py` — Upload, download, SAS URL generation, lifecycle
- `cloud/storage/container_setup.py` — Create containers: raw-videos, processed, models, user-configs
- `cloud/storage/config.py` — Connection settings from env vars
- `cloud/wrapper/pipeline.py` — CloudPipeline: download video → run BeeMonitor → upload results
- `cloud/wrapper/model_manager.py` — Download models from Azure, cache locally
- `cloud/scripts/seed_models.py` — Upload .pt and .pkl models to Azure
- `cloud/scripts/setup_containers.py` — One-time container creation
- `cloud/requirements-cloud.txt`

**Integration:** `wrapper/pipeline.py` imports `BeeMonitor` and `Config` from existing package. `azure_client.py` manages container layout from design doc 03.

**Validation:**
- Unit tests with Azurite local emulator
- CloudPipeline processes a sample video end-to-end
- Model files round-trip through Azure (upload → download → checksum)

---

## STAGE 2: Modal.com Serverless Functions

**Objective:** Deploy BeeMonitor pipeline as Modal serverless functions with GPU support.

**Dependencies:** Stage 1

**Deliverables:**
- `cloud/modal_app/app.py` — `modal.App("beemonitor-cloud")`, shared image
- `cloud/modal_app/image.py` — Container image with all deps + beemonitor package
- `cloud/modal_app/functions/process_video.py` — `@app.function(gpu="L4")` main processing
- `cloud/modal_app/functions/ingest_video.py` — `@app.function()` CPU, download from sources
- `cloud/modal_app/functions/generate_results.py` — CPU, CSV + annotated video
- `cloud/modal_app/functions/batch_process.py` — `Function.map()` orchestrator
- `cloud/modal_app/volumes.py` — Modal Volume for models
- `cloud/modal_app/secrets.py` — Modal Secret references

**Integration:** `process_video.py` uses `cloud.wrapper.pipeline.CloudPipeline` (Stage 1). `ingest_video.py` uses `cloud.storage.azure_client` (Stage 1). Modal image copies `src/beemonitor/`.

**Validation:**
- `modal run` deploys successfully
- Process sample video → results appear in Azure `processed/` container
- CPU ingest downloads and stores in `raw-videos/`
- Batch of 3 videos processes in parallel

---

## STAGE 3: Multi-Cloud Data Ingestion

**Objective:** Connectors for AWS S3, Azure Blob, GCS, Google Drive, and direct upload.

**Dependencies:** Stages 1-2

**Deliverables:**
- `cloud/ingestion/base_connector.py` — Abstract: `list_files()`, `download_to_azure()`, `test_connection()`
- `cloud/ingestion/s3_connector.py` — boto3-based
- `cloud/ingestion/azure_connector.py` — Same-cloud optimized copy
- `cloud/ingestion/gcs_connector.py` — google-cloud-storage
- `cloud/ingestion/gdrive_connector.py` — Google Drive API v3 + OAuth2
- `cloud/ingestion/direct_upload.py` — Chunked multipart handler
- `cloud/ingestion/credential_manager.py` — Fernet encryption for credentials

**Integration:** Connectors used by `modal_app/functions/ingest_video.py` (Stage 2). `credential_manager.py` used later by Django `sources` app (Stage 6).

**Validation:**
- Unit tests with mocked SDKs (moto for S3)
- Integration: S3 → Azure transfer
- Connection validation returns success/failure per source
- Streaming transfer (bounded memory for large files)

---

## STAGE 4: Django Project Skeleton + Database Models

**Objective:** Django project with all apps, models, PostgreSQL, Redis + Celery, and admin.

**Dependencies:** Stage 1 (Azure config), conceptual awareness of 2-3

**Deliverables:**
- Django project: `beemonitor_web/` with `config/settings/{base,dev,prod}.py`
- 7 apps: `accounts`, `videos`, `analysis`, `sources`, `dashboard`, `developer`, `api`
- All models: `UserProfile`, `APIKey`, `DataSource`, `Video`, `Job`, `JobResult`, `UsageLog`, `WebhookEndpoint`
- Database migrations
- Celery + Redis configuration
- Admin site with all models
- `docker-compose.yml` for local PostgreSQL + Redis

**Key Models:**
```
UserProfile: user, organization, tier, monthly_job_count, storage_used_bytes
APIKey: user, key_hash, prefix, name, key_type, permissions, rate_limit, is_active
DataSource: user, name, source_type, config_encrypted, is_connected
Video: user, source, title, azure_blob_path, file_size, duration, resolution, fps, status
Job: user, video, modal_job_id, status, config, progress_pct, error_message, compute_cost
JobResult: job, events_csv_path, tracking_csv_path, annotated_video_path, stats
```

**Validation:**
- `manage.py migrate` succeeds
- All models in admin site
- FK relationships verified in shell
- Celery worker connects to Redis

---

## STAGE 5: REST API (Django REST Framework)

**Objective:** Complete REST API with key auth, rate limiting, all CRUD endpoints, Celery tasks for Modal, and Swagger docs.

**Dependencies:** Stage 4 (models), Stage 2 (Modal), Stage 3 (ingestion)

**Deliverables:**
- `apps/api/` — Serializers, ViewSets, URLs, permissions, auth, throttling, filters
- `apps/analysis/tasks.py` — Celery: `submit_analysis_job`, `check_job_status`
- `apps/analysis/modal_client.py` — Wrapper for `modal.Function.lookup()`
- `apps/videos/tasks.py` — Celery: `ingest_from_source`, `process_upload`
- `apps/videos/upload_handler.py` — Chunked upload + SAS tokens
- `apps/sources/connectors.py` — Thin wrappers around `cloud/ingestion/`
- `apps/developer/webhook_dispatcher.py` — Job completion webhooks

**API Endpoints (full list):**
- Auth: register, login, api-key CRUD
- Sources: CRUD + test connection
- Videos: upload (chunked), ingest, list, detail, delete
- Jobs: submit, list, detail, results, batch, cancel
- Results: events CSV, tracks, video (SAS URL), stats
- Webhooks: register, list, delete
- Health + Usage

**API Key Tiers:**
| Tier | Rate | Jobs/Day | GPU |
|------|------|----------|-----|
| Free | 10/min | 5 | CPU only |
| Standard | 60/min | 100 | Yes |
| Pro | 300/min | Unlimited | Priority |

**Validation:**
- All endpoints return correct HTTP codes
- API key auth works (valid=200, invalid=401, revoked=401)
- Rate limiting returns 429
- Job submission triggers Celery → Modal
- Swagger UI at `/api/v1/docs/`
- E2E: upload → submit → poll → download results

---

## STAGE 6: Django Web UI (Templates + Views)

**Objective:** Server-rendered web UI with Django templates, HTMX, Alpine.js, Tailwind CSS.

**Dependencies:** Stages 4-5

**Deliverables:**
- `templates/base.html` — Layout with Tailwind + HTMX + Alpine.js
- All page templates: landing, dashboard, videos, sources, analysis, results, developer, settings
- Django views for each app
- HTMX real-time job status polling
- Chunked video upload JS + progress bar
- django-allauth (Google, GitHub social auth)
- Chart.js for results visualization

**8 Page Sections:**
1. Landing page
2. Dashboard (stats, recent jobs, activity feed)
3. Videos (upload, list, detail, drag-and-drop)
4. Data Sources (connect S3/Azure/GCS/GDrive, test, manage)
5. Analysis (new job, queue, status, batch)
6. Results Viewer (events table, charts, video player, nest heatmap, export)
7. Developer Portal (API keys, docs, usage, webhooks)
8. Settings (profile, notifications, defaults, billing)

**Validation:**
- All pages render without errors
- Full user flow: register → login → upload → analyze → view results
- HTMX polling updates job status live
- Responsive on mobile viewport
- Social auth works

---

## STAGE 7: Progressive Web App (PWA)

**Objective:** Add offline-first PWA capabilities to the Django web app.

**Dependencies:** Stage 6

**Deliverables:**
- `manifest.json` — Web app manifest
- `sw.js` — Workbox service worker (cache strategies, background sync)
- `offline-db.js` — Dexie.js IndexedDB schema
- `sync-manager.js` — Background sync queue
- `push-manager.js` — Push notification subscription
- `offline.html` — Offline fallback page
- `apps/pwa/push_service.py` — web-push server integration

**Offline Capabilities:**
- View dashboard with cached data
- Browse previously loaded videos/results
- Queue video uploads (auto-sync when online)
- Queue analysis jobs (auto-submit when online)
- Push notifications on job completion

**Validation:**
- Lighthouse PWA score > 95
- Install prompt on Android Chrome / iOS Safari
- Works offline after first visit
- Background sync: queue offline → process online
- Push notifications received

---

## STAGE 8: Deployment + CI/CD

**Objective:** Production deployment, CI/CD, monitoring, security hardening.

**Dependencies:** Stages 4-7

**Deliverables:**
- `Dockerfile` (Django + Gunicorn), `Dockerfile.celery`
- `docker-compose.prod.yml`
- `.github/workflows/{test,deploy,lint}.yml`
- Production settings (HTTPS, CORS, CSP, HSTS)
- Sentry error tracking
- Azure CDN for static files
- Health check endpoint

**Deployment Stack:**
- Azure App Service (Django + Gunicorn)
- Azure Database for PostgreSQL
- Azure Cache for Redis
- Azure Container Instances (Celery workers)
- Azure Blob Storage + CDN
- Modal.com (GPU processing)

**Validation:**
- Docker builds succeed
- CI pipeline runs on PR
- Production deploy via GitHub Actions
- HTTPS enforced, security headers present
- Sentry captures errors
- Health endpoint returns 200

---

## STAGE 9: React Native App — Core + Offline

**Objective:** React Native mobile app with auth, dashboard, video/job lists, offline DB, and API client.

**Dependencies:** Stage 5 (REST API deployed)

**Deliverables:**
- Expo project with React Navigation (tab + stack)
- Auth flow (login, register, MMKV token storage)
- API client (Axios + interceptors)
- WatermelonDB offline database + sync
- Core screens: Dashboard, Video List, Job List, Settings
- Offline detection + banner
- TanStack Query for server state

**Validation:**
- Builds on iOS simulator + Android emulator
- Auth flow works against deployed API
- Dashboard shows real data
- Network kill → offline banner → cached data visible
- Network restore → data syncs
- WatermelonDB persists across app restarts

---

## STAGE 10: React Native — Upload, Capture, Field Features

**Objective:** Video capture, chunked upload, push notifications, GPS tagging, battery awareness, app store prep.

**Dependencies:** Stage 9

**Deliverables:**
- Camera capture (react-native-vision-camera)
- Chunked upload with background processing + resume
- Upload queue with offline queuing
- Push notifications (Firebase Cloud Messaging)
- GPS tagging, battery-aware scheduling
- New job submission screen
- EAS Build for production

**Field Features:**
- Low bandwidth mode (compress, skip annotated video)
- GPS auto-tagging per video
- Multi-site management
- Quick field notes (text/voice)
- Solar power awareness (schedule uploads during peak)

**Validation:**
- Camera captures and saves locally
- Chunked upload resumes after app kill
- Offline → record → online → auto-upload
- Push notification on job completion
- GPS coordinates in video metadata
- Battery < 20% pauses uploads
- EAS Build produces installable APK/IPA

---

## Dependency Graph

```
Stage 1: Azure Storage + Cloud Wrapper
    ↓
Stage 2: Modal Serverless Functions ────┐
    ↓                                    │
Stage 3: Multi-Cloud Ingestion          │
    ↓                                    ↓
Stage 4: Django Models + DB ────────→ Stage 5: REST API
                                        ↓          ↓
                                   Stage 6: Web UI   Stage 9: RN Core
                                        ↓                ↓
                                   Stage 7: PWA     Stage 10: RN Full
                                        ↓
                                   Stage 8: Deploy + CI/CD
```

**Parallel tracks after Stage 5:**
- Track A (Web): Stages 6 → 7 → 8
- Track B (Mobile): Stages 9 → 10

---

## Key Architectural Decisions

1. **`src/beemonitor/` stays untouched.** `cloud/wrapper/pipeline.py` imports and calls it.
2. **Django is single source of truth** for user data. Modal writes to Azure; Django manages state.
3. **Azure Blob is canonical file store.** All clouds ingest INTO Azure. Django serves SAS URLs.
4. **PWA is NOT separate.** Same Django templates + service worker layer. No separate build.
5. **React Native is fully separate** in `beemonitor-mobile/`. Same DRF API as web.
6. **Modal invoked via Celery tasks**, not Django views directly. Keeps web requests fast.
