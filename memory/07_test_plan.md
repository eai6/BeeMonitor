# BeeMonitor Cloud Platform — Test Plan

## Overview

Comprehensive test plan covering all 10 stages. Tests are organized by layer: unit, integration, end-to-end. Each section lists what to test, how to test it, and the expected outcome.

---

## 1. Cloud Storage Layer (Stage 1)

### Unit Tests (47 existing, all passing)

| Test | File | Status |
|------|------|--------|
| StorageConfig defaults and validation | `cloud/tests/test_storage_config.py` | PASS |
| AzureBlobClient init (conn string, account key, missing creds) | `cloud/tests/test_azure_client.py` | PASS |
| Upload, download, list, delete operations (mocked) | `cloud/tests/test_azure_client.py` | PASS |
| Container creation (including already-exists) | `cloud/tests/test_azure_client.py` | PASS |
| ModelManager download, cache, skip-cached, upload, verify | `cloud/tests/test_model_manager.py` | PASS |
| CloudPipeline process flow, cleanup | `cloud/tests/test_pipeline.py` | PASS |

### Integration Tests (TO ADD)

| Test | Description | How to Run |
|------|-------------|------------|
| Azurite local emulator | Start Azurite Docker container, test real upload/download/SAS URL generation | `docker run -p 10000:10000 mcr.microsoft.com/azure-storage/azurite` |
| Model round-trip | Upload 3 model files to Azurite, download, compare checksums | `python -m cloud.scripts.seed_models --verify` |
| Container lifecycle | Create all containers, list, verify names match config | `python -m cloud.scripts.setup_containers` |
| SAS URL access | Generate SAS URL, fetch blob via HTTP, verify content matches | Manual with curl |
| Large file streaming | Upload/download a 500MB test file, verify memory stays bounded | Profile with `tracemalloc` |

---

## 2. Modal Serverless Functions (Stage 2)

### Unit Tests (7 existing, all passing)

| Test | File | Status |
|------|------|--------|
| Ingest dispatch: S3, Azure, GCS, GDrive, direct, unsupported | `cloud/tests/test_modal_functions.py` | PASS |
| Process video module imports | `cloud/tests/test_modal_functions.py` | PASS |

### Integration Tests (TO ADD — requires Modal account)

| Test | Description | Validation |
|------|-------------|------------|
| `modal run` deploys | Deploy app, verify health endpoint returns 200 | `curl https://beemonitor-cloud--health.modal.run` |
| GPU process_video | Upload sample video to Azure, call process_video, check results in processed/ | Events CSV exists, >0 events |
| CPU ingest_video | Call ingest with `direct` source, verify blob in raw-videos/ | Blob exists, size matches |
| Batch processing | Submit 3 videos via batch_process, verify all complete | 3 result dicts returned |
| Model caching | Run process_video twice, verify second run skips download | Modal Volume has model files |
| Timeout handling | Submit very long video, verify timeout error after 2hr | FunctionTimeoutError raised |

---

## 3. Multi-Cloud Ingestion (Stage 3)

### Unit Tests (12 existing, all passing)

| Test | File | Status |
|------|------|--------|
| S3Connector: connection test, list, download | `cloud/tests/test_ingestion.py` | PASS |
| DirectUploadHandler: init, chunks, finalize, missing chunks | `cloud/tests/test_ingestion.py` | PASS |
| CredentialManager: encrypt/decrypt round-trip, wrong key fails | `cloud/tests/test_ingestion.py` | PASS |

### Integration Tests (TO ADD — requires cloud credentials)

| Test | Source | Description | Validation |
|------|--------|-------------|------------|
| S3 real download | AWS S3 | Upload test.mp4 to S3, use S3Connector to download | File size matches |
| Azure same-cloud copy | Azure Blob | Copy blob between containers, verify server-side copy | No egress, blob exists |
| GCS download | GCP Storage | Upload to GCS, download via GCSConnector | Content matches |
| Google Drive download | Google Drive | Share test video, download via OAuth token | File size > 0 |
| Credential encryption at rest | All | Encrypt creds, store as bytes, decrypt, use to connect | Connection succeeds |
| Streaming memory bound | S3 | Download 1GB file, assert peak memory < 100MB | `tracemalloc` check |

---

## 4. Django Models & Database (Stage 4)

### Unit Tests (TO ADD)

| Test | App | Description |
|------|-----|-------------|
| UserProfile auto-creation | accounts | Signal creates profile on User.save() (if signal added) |
| APIKey.create_key() | accounts | Returns (instance, raw_key), key starts with `bmk_`, hash stored |
| APIKey hash verification | accounts | Hash raw_key with SHA-256, matches stored key_hash |
| Video status transitions | videos | uploading → ready → processing → archived |
| Job status transitions | analysis | queued → processing → completed / failed / cancelled |
| JobResult OneToOne | analysis | job.result accessible after JobResult creation |
| DataSource encrypted config | sources | Store encrypted, retrieve and decrypt, matches original |
| Cascade deletes | all | Delete User → cascades to Profile, Videos, Jobs, Sources |
| Model __str__ methods | all | All models return meaningful string representations |

### How to run:
```bash
cd beemonitor_web
python manage.py test apps.accounts apps.videos apps.analysis apps.sources apps.developer
```

---

## 5. REST API (Stage 5)

### Unit Tests (TO ADD)

| Test | Endpoint | Method | Auth | Expected |
|------|----------|--------|------|----------|
| Health check | `/api/v1/health/` | GET | None | 200, `{"status": "ok"}` |
| List videos (unauth) | `/api/v1/videos/` | GET | None | 401 |
| List videos (auth) | `/api/v1/videos/` | GET | API key | 200, user's videos only |
| Create video | `/api/v1/videos/` | POST | API key | 201, video record created |
| Submit job | `/api/v1/jobs/` | POST | API key | 201, status=queued |
| Submit job (Celery) | `/api/v1/jobs/{id}/submit/` | POST | API key | 202, Celery task dispatched |
| Job status | `/api/v1/jobs/{id}/` | GET | API key | 200, current status |
| List sources | `/api/v1/sources/` | GET | API key | 200, user's sources |
| Create source | `/api/v1/sources/` | POST | API key | 201, credentials encrypted |
| Test connection | `/api/v1/sources/{id}/test_connection/` | POST | API key | 200, success/failure |
| Create API key | `/api/v1/api-keys/` | POST | Session | 201, raw_key in response |
| Revoke API key | `/api/v1/api-keys/{id}/` | DELETE | Session | 204, key deactivated |
| Rate limiting (free) | `/api/v1/videos/` | GET×11 | Free key | 429 on 11th request |
| Rate limiting (pro) | `/api/v1/videos/` | GET×301 | Pro key | 429 on 301st request |
| Invalid API key | `/api/v1/videos/` | GET | Bad key | 401 |
| Revoked API key | `/api/v1/videos/` | GET | Revoked key | 401 |
| Cross-user isolation | `/api/v1/videos/` | GET | User B key | Cannot see User A videos |
| Webhook create | `/api/v1/webhooks/` | POST | API key | 201 |
| Pagination | `/api/v1/videos/?page=2` | GET | API key | 200, paginated results |
| Filter jobs by status | `/api/v1/jobs/?status=completed` | GET | API key | 200, filtered |

### How to run:
```bash
cd beemonitor_web
python manage.py test apps.api.tests
# Or with pytest:
pytest apps/api/tests/ -v
```

---

## 6. Django Web UI (Stage 6)

### Functional Tests (TO ADD)

| Test | Page | Description | Validation |
|------|------|-------------|------------|
| Dashboard loads | `/` | Authenticated user sees stats | 200, stat cards rendered |
| Dashboard redirect | `/` | Unauthenticated → login | 302 to /accounts/login/ |
| Login flow | `/accounts/login/` | POST valid credentials | 302 to / |
| Login invalid | `/accounts/login/` | POST bad password | 200, error message |
| Register flow | `/accounts/register/` | POST valid form | 302 to /, user created |
| Video list | `/videos/` | Shows user's videos | 200, table rendered |
| Video upload | `/videos/upload/` | POST multipart file | 302 to video detail |
| Video detail | `/videos/{id}/` | Shows metadata | 200, title displayed |
| Job list | `/analysis/` | Shows user's jobs | 200, table rendered |
| Job create | `/analysis/new/` | POST with video_id | 302 to job detail |
| Job detail polling | `/analysis/{id}/` | HTMX fragment endpoint | 200, status HTML |
| Results page | `/analysis/{id}/results/` | Completed job results | 200, stats displayed |
| Source list | `/sources/` | Shows connected sources | 200 |
| Developer portal | `/developer/` | API key management | 200, keys listed |

### How to run:
```bash
cd beemonitor_web
python manage.py test  # Django TestClient tests
# Or manual browser testing:
python manage.py runserver
# Visit http://localhost:8000
```

---

## 7. PWA (Stage 7)

### Manual Tests

| Test | Description | Validation |
|------|-------------|------------|
| Manifest served | Visit `/manifest.json` | Valid JSON, theme_color=#F59E0B |
| Service worker registered | Check browser DevTools → Application | SW status: activated |
| Offline page | Kill network, navigate to new page | Offline page with "Try Again" |
| Cache-first static | Kill network, reload cached page | Page loads from cache |
| API cache fallback | Load jobs list, kill network, reload | Cached data displayed |
| Install prompt | Visit on mobile Chrome | "Add to Home Screen" appears |
| Lighthouse PWA audit | Run Lighthouse in DevTools | Score > 90 |
| Background sync stub | Queue action offline, restore network | Console log "Processing queue" |

---

## 8. Deployment & CI/CD (Stage 8)

### Tests

| Test | Description | Validation |
|------|-------------|------------|
| Docker build (web) | `cd beemonitor_web && docker build .` | Image builds successfully |
| Docker build (celery) | `docker build -f Dockerfile.celery .` | Image builds successfully |
| Docker compose up | `docker-compose -f docker-compose.prod.yml up` | All 4 services healthy |
| Health endpoint | `curl localhost:8000/api/v1/health/` | 200 OK |
| Static files served | `curl localhost:8000/static/js/sw.js` | 200, JS content |
| CI pipeline | Push PR to GitHub | `test.yml` runs, all checks pass |
| Migration on startup | Start fresh container | Entrypoint runs migrate |

---

## 9-10. React Native App (Stages 9-10)

### Unit Tests (TO ADD)

| Test | File | Description |
|------|------|-------------|
| formatBytes | `src/utils/formatters.ts` | 0, 1024, 1048576, large values |
| formatDate | `src/utils/formatters.ts` | ISO string → readable format |
| formatDuration | `src/utils/formatters.ts` | Seconds → "Xm Ys" format |
| authStore login | `src/stores/authStore.ts` | Sets token, isLoggedIn=true |
| authStore logout | `src/stores/authStore.ts` | Clears token, isLoggedIn=false |
| API client auth header | `src/api/client.ts` | Adds Bearer token when logged in |
| API client 401 handler | `src/api/client.ts` | Calls logout on 401 |
| JobStatusBadge colors | `src/components/` | Each status maps to correct color |
| EmptyState renders | `src/components/` | Shows title and message |

### Integration Tests

| Test | Screen | Description | Validation |
|------|--------|-------------|------------|
| Login flow | (auth)/login | Enter creds, submit, navigate to tabs | Dashboard visible |
| Register flow | (auth)/register | Fill form, submit | Auto-login, dashboard |
| Video list loads | (tabs)/videos | API returns videos | FlatList populated |
| Job list loads | (tabs)/jobs | API returns jobs | FlatList populated |
| Job polling | job/[id] | Active job auto-refreshes | Status updates |
| Upload flow | video/upload | Pick file, upload | Progress bar, completion |
| Offline banner | Any screen | Kill network | Yellow banner appears |
| Offline → Online | Any screen | Restore network | Banner disappears, data refreshes |
| Deep link | beemonitor://job/123 | Open from push notification | Job detail screen |

### How to run:
```bash
cd beemonitor-mobile
npm install
npx expo start          # Dev server
npx jest                # Unit tests
npx expo run:ios        # iOS simulator
npx expo run:android    # Android emulator
```

---

## End-to-End Test Scenarios

### E2E 1: Full Video Processing Pipeline

```
1. Register new user via web UI
2. Upload video via /videos/upload/
3. Verify video appears in /videos/ list (status=ready)
4. Submit analysis job via /analysis/new/
5. Monitor job status (HTMX polling on /analysis/{id}/)
6. Wait for status=completed
7. View results at /analysis/{id}/results/
8. Download events CSV
9. Verify events count > 0
```

**Expected duration:** 2-10 minutes depending on video length
**Infrastructure needed:** Django + Celery + Redis + Modal (GPU) + Azure Blob

### E2E 2: Multi-Cloud Ingestion

```
1. Connect S3 source via /sources/add/
2. Test connection (should succeed)
3. Ingest video from S3 via API: POST /api/v1/videos/ingest
4. Verify video appears in Azure raw-videos container
5. Submit analysis job
6. Verify results generated
```

### E2E 3: API Key Workflow

```
1. Create API key via /developer/ (type=live)
2. Copy raw key from response
3. Use key in curl: curl -H "Authorization: Bearer bmk_live_xxx" /api/v1/videos/
4. Verify 200 response with user's videos
5. Revoke key via /developer/
6. Repeat curl → verify 401
```

### E2E 4: Mobile App Full Flow

```
1. Login on mobile app
2. Dashboard loads with stats
3. Upload video from device gallery
4. Wait for upload completion
5. Create analysis job
6. Monitor job status (auto-polling)
7. View results when completed
8. Kill network → verify offline banner + cached data visible
9. Restore network → verify data syncs
```

### E2E 5: Batch Processing

```
1. Upload 5 videos via API
2. Submit batch job: POST /api/v1/jobs/batch
3. Monitor all 5 jobs (should process in parallel on Modal)
4. Verify all 5 complete with results
5. Check Modal dashboard for GPU utilization
```

---

## Test Environment Setup

### Local Development
```bash
# Terminal 1: Django
cd beemonitor_web && python manage.py runserver

# Terminal 2: Celery
cd beemonitor_web && celery -A config.celery worker -l info

# Terminal 3: Redis
docker run -p 6379:6379 redis:7-alpine

# Terminal 4: Azurite (local Azure emulator)
docker run -p 10000:10000 mcr.microsoft.com/azure-storage/azurite

# Terminal 5: React Native
cd beemonitor-mobile && npx expo start
```

### Environment Variables (.env)
```
DJANGO_SECRET_KEY=test-secret-key-change-in-prod
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=...;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;
CELERY_BROKER_URL=redis://localhost:6379/0
BEEMONITOR_CREDENTIAL_KEY=<generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
MODAL_TOKEN_ID=<from modal dashboard>
MODAL_TOKEN_SECRET=<from modal dashboard>
```

---

## Test Coverage Targets

| Layer | Current Tests | Target Tests | Coverage Goal |
|-------|--------------|-------------|---------------|
| Cloud Storage | 28 | 40+ | 90% |
| Modal Functions | 7 | 15+ | 80% |
| Ingestion | 12 | 25+ | 85% |
| Django Models | 0 | 20+ | 90% |
| REST API | 0 | 30+ | 85% |
| Web UI | 0 | 15+ | 70% |
| PWA | 0 | 8 (manual) | Manual |
| React Native | 0 | 20+ | 75% |
| **Total** | **47** | **175+** | **80% avg** |

---

## Priority Order for Test Implementation

1. **REST API tests** — Highest risk, most critical path (Stage 5)
2. **Django model tests** — Data integrity (Stage 4)
3. **Cloud integration tests with Azurite** — Storage reliability (Stage 1)
4. **Web UI functional tests** — User-facing flows (Stage 6)
5. **React Native unit tests** — Formatters, stores (Stage 9)
6. **Modal integration tests** — Requires account setup (Stage 2)
7. **Multi-cloud connector tests** — Requires credentials (Stage 3)
8. **PWA manual tests** — Browser-based (Stage 7)
