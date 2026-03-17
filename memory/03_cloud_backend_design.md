# BeeMonitor Cloud Backend Design (Modal.com)

## Architecture Overview

A serverless GPU/CPU cloud backend built on **Modal.com** for large-scale video processing, multi-cloud data ingestion, REST API access, and Azure Blob Storage for primary user data.

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT APPLICATIONS                         │
│  Desktop App │ Django Web │ PWA │ React Native │ 3rd-Party API  │
└──────────┬──────────┬──────────┬──────────┬──────────┬──────────┘
           │          │          │          │          │
           ▼          ▼          ▼          ▼          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    API GATEWAY (Modal FastAPI)                   │
│  Authentication │ Rate Limiting │ API Key Validation             │
│  POST /api/v1/analyze │ POST /api/v1/upload │ GET /api/v1/jobs  │
└──────────┬──────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODAL SERVERLESS PLATFORM                     │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  INGEST SVC  │  │  PROCESS SVC │  │  RESULTS SVC         │  │
│  │  (CPU)       │  │  (GPU/CPU)   │  │  (CPU)               │  │
│  │              │  │              │  │                       │  │
│  │ • S3 fetch   │  │ • YOLO det.  │  │ • CSV generation     │  │
│  │ • Azure pull │  │ • Tracking   │  │ • Stats aggregation  │  │
│  │ • GCS pull   │  │ • Events ML  │  │ • Video annotation   │  │
│  │ • GDrive     │  │ • Batch map  │  │ • Webhook callback   │  │
│  │ • Direct up  │  │              │  │                       │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬───────────────┘  │
│         │                 │                   │                   │
│         ▼                 ▼                   ▼                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              MODAL VOLUMES (Hot Storage)                 │    │
│  │  • Model weights (nest_detection.pt, bee_tracking.pt)   │    │
│  │  • Temp video chunks during processing                  │    │
│  │  • Processing state / checkpoints                       │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 AZURE BLOB STORAGE (Primary Data)                │
│                                                                  │
│  containers/                                                     │
│  ├── raw-videos/{user_id}/{upload_id}/video.mp4                 │
│  ├── processed/{user_id}/{job_id}/                              │
│  │   ├── events.csv                                             │
│  │   ├── tracking_results.csv                                   │
│  │   └── annotated_video.mp4                                    │
│  ├── models/                                                     │
│  │   ├── nest_detection.pt                                      │
│  │   ├── bee_tracking.pt                                        │
│  │   └── event_classifier_model.pkl                             │
│  └── user-data/{user_id}/config.yaml                            │
└─────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────┐
│              MULTI-CLOUD DATA SOURCES (User-Provided)            │
│                                                                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────────┐  │
│  │  AWS S3  │  │  Azure  │  │  GCS    │  │  Google Drive    │  │
│  │  Bucket  │  │  Blob   │  │  Bucket │  │  (OAuth2)        │  │
│  └─────────┘  └─────────┘  └─────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Modal Function Architecture

### Compute Strategy: GPU vs CPU

| Task | Compute | GPU Type | Est. Cost/min |
|------|---------|----------|---------------|
| Video download/upload | CPU (2 cores, 4 GiB) | — | $0.002 |
| YOLO detection + tracking | GPU | L4 | $0.013 |
| Event ML classification | CPU (4 cores, 8 GiB) | — | $0.004 |
| Video annotation rendering | CPU (4 cores, 8 GiB) | — | $0.004 |
| Batch (10 videos parallel) | GPU × 10 | L4 | $0.13/min total |

**Decision:** Use **L4 GPU** for detection+tracking (best price/performance). CPU for everything else. Total cost per 1-hour video ≈ **$0.50-$1.50**.

### Modal Functions

```python
import modal

app = modal.App("beemonitor-cloud")

# Shared image with all dependencies
beemonitor_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("ultralytics", "opencv-python-headless", "torch", "torchvision",
                 "scikit-learn", "pandas", "numpy", "scipy", "filterpy",
                 "azure-storage-blob", "boto3", "google-cloud-storage",
                 "google-auth", "fastapi", "pydantic")
    .copy_local_dir("src/beemonitor", "/app/beemonitor")
)

model_volume = modal.Volume.from_name("beemonitor-models", create_if_missing=True)

# GPU function for video processing
@app.function(
    image=beemonitor_image,
    gpu="L4",
    timeout=7200,  # 2 hours max
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("azure-storage"), modal.Secret.from_name("aws-creds")],
)
def process_video(job_id: str, video_source: dict, user_id: str):
    """Main GPU processing function."""
    # 1. Download video from source (S3/Azure/GCS/GDrive)
    # 2. Run BeeMonitor pipeline
    # 3. Upload results to Azure Blob
    # 4. Return job status
    ...

# CPU function for data ingestion
@app.function(image=beemonitor_image, timeout=3600)
def ingest_video(source_type: str, source_config: dict, user_id: str) -> str:
    """Download video from any cloud source to Azure Blob."""
    ...

# Batch processing
@app.function(image=beemonitor_image, gpu="L4", timeout=7200)
def process_video_batch(job_configs: list):
    """Process multiple videos via Function.map()."""
    ...
```

---

## 2. REST API Design

### Authentication

Three tiers of access:
1. **User Auth** (Django sessions) — Web app users
2. **API Key Auth** (Bearer token) — Programmatic access for developers
3. **Device Auth** (API key + device ID) — Hardware devices (Raspberry Pi)

### API Endpoints

```
BASE URL: https://beemonitor--api.modal.run/api/v1

# Authentication
POST   /auth/register          — Create account
POST   /auth/login             — Get session/token
POST   /auth/api-key           — Generate API key
DELETE /auth/api-key/{key_id}  — Revoke API key

# Data Sources (connect external storage)
POST   /sources                — Register a data source
GET    /sources                — List connected sources
DELETE /sources/{source_id}    — Remove source
POST   /sources/{id}/test      — Test connection

  Supported source_types:
  - "aws_s3"      → {bucket, prefix, access_key_id, secret_access_key}
  - "azure_blob"  → {container, connection_string}
  - "gcs"         → {bucket, prefix, service_account_json}
  - "google_drive" → {folder_id, oauth_token}
  - "direct_upload" → (multipart form data)

# Video Upload & Ingestion
POST   /videos/upload          — Direct video upload (multipart, max 10 GB)
POST   /videos/ingest          — Ingest from connected source
GET    /videos                 — List user's videos
GET    /videos/{video_id}      — Video metadata
DELETE /videos/{video_id}      — Delete video

# Processing Jobs
POST   /jobs                   — Submit analysis job
GET    /jobs                   — List jobs (with status filter)
GET    /jobs/{job_id}          — Job status + progress
GET    /jobs/{job_id}/results  — Download results (CSV, video)
POST   /jobs/batch             — Submit batch of videos
DELETE /jobs/{job_id}          — Cancel job

# Results
GET    /results/{job_id}/events     — Events CSV
GET    /results/{job_id}/tracks     — Tracking data
GET    /results/{job_id}/video      — Annotated video (signed URL)
GET    /results/{job_id}/stats      — Summary statistics

# Webhooks (for async notifications)
POST   /webhooks               — Register callback URL
GET    /webhooks               — List webhooks
DELETE /webhooks/{webhook_id}  — Remove webhook

# Health
GET    /health                 — Service health check
GET    /usage                  — User's compute usage + billing
```

### Request/Response Examples

**Submit Job:**
```json
POST /api/v1/jobs
Authorization: Bearer <api_key>
{
  "video_id": "vid_abc123",
  "config": {
    "detection_mode": "yolo",
    "confidence_threshold": 0.25,
    "ml_threshold": 0.6,
    "visualize": true,
    "gpu": true
  }
}

Response: 202 Accepted
{
  "job_id": "job_xyz789",
  "status": "queued",
  "estimated_duration_seconds": 300,
  "webhook_url": "https://..."
}
```

**Ingest from S3:**
```json
POST /api/v1/videos/ingest
{
  "source_type": "aws_s3",
  "source_config": {
    "bucket": "my-bee-videos",
    "key": "field-site-1/2024-06-15_10_00_00.mp4"
  }
}
```

---

## 3. Multi-Cloud Data Ingestion

### Supported Sources

| Source | Auth Method | Max File Size | Transfer Method |
|--------|------------|---------------|-----------------|
| AWS S3 | Access Key or OIDC | Unlimited | boto3 streaming |
| Azure Blob | Connection String or SAS | Unlimited | azure-storage-blob |
| GCP Storage | Service Account JSON | Unlimited | google-cloud-storage |
| Google Drive | OAuth2 token | 5 GB | Google Drive API v3 |
| Direct Upload | API key | 10 GB | Multipart upload via API |
| Raspberry Pi | Device API key | 10 GB | Chunked upload via API |

### Ingestion Flow

```
User/Device → API (source_type + credentials)
           → Ingest Worker (CPU)
              → Download from source (streaming, chunked)
              → Upload to Azure Blob (raw-videos/{user_id}/{upload_id}/)
              → Create video record in DB
              → Return video_id
```

---

## 4. Azure Blob Storage Architecture

### Container Layout

```
beemonitor-storage/
├── raw-videos/
│   └── {user_id}/
│       └── {upload_id}/
│           ├── video.mp4
│           └── metadata.json       # source, upload time, size, duration
│
├── processed/
│   └── {user_id}/
│       └── {job_id}/
│           ├── events.csv
│           ├── tracking_results.csv
│           ├── annotated_video.mp4
│           ├── nest_detections.json
│           └── job_metadata.json   # config, timing, stats
│
├── models/
│   ├── v1/
│   │   ├── nest_detection.pt
│   │   ├── bee_tracking.pt
│   │   └── event_classifier_model.pkl
│   └── latest → v1/
│
└── user-configs/
    └── {user_id}/
        └── config.yaml            # user-specific defaults
```

### Storage Tiers
- **Hot:** Active videos being processed (auto-delete after 30 days)
- **Cool:** Processed results (user-accessible, lower cost)
- **Archive:** Old raw videos (user can request restore)

### Access Patterns
- **Write:** Ingest workers upload raw video; processing workers upload results
- **Read:** Processing workers read raw video; API serves results via SAS URLs
- **Delete:** Automatic lifecycle policies + user-initiated

---

## 5. API Key System for Developers

### Key Types

| Type | Permissions | Rate Limit | Use Case |
|------|------------|------------|----------|
| Free | 5 jobs/day, CPU only | 10 req/min | Testing/evaluation |
| Standard | 100 jobs/day, GPU | 60 req/min | Research use |
| Pro | Unlimited, priority GPU | 300 req/min | Production integration |
| Device | Upload + trigger only | 30 req/min | Raspberry Pi hardware |

### Key Format
```
bmk_live_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   (production)
bmk_test_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   (sandbox)
bmk_dev_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx    (device)
```

### SDK Usage (Python)
```python
from beemonitor_sdk import BeeMonitorCloud

client = BeeMonitorCloud(api_key="bmk_live_xxx")

# Upload and analyze
video_id = client.upload("field_video.mp4")
job = client.analyze(video_id, detection_mode="yolo", gpu=True)

# Poll or await
result = job.wait()  # blocks until complete
events_df = result.events()
result.download_video("annotated.mp4")
```

---

## 6. Processing Pipeline (Modal)

```
Job Submitted → Queue
    ↓
[1] Ingest (CPU, 2 cores)
    • Download video from Azure Blob (or source)
    • Validate format, extract metadata
    • Split into chunks if > 1 hour
    ↓
[2] Nest Detection (GPU, L4)
    • Load nest_detection.pt from Volume
    • YOLO inference on reference frames
    • Cluster into grid
    • Cache nest positions for user's site
    ↓
[3] Detection + Tracking (GPU, L4)
    • Load bee_tracking.pt from Volume
    • Frame-by-frame YOLO + BeeTracker
    • For long videos: parallel chunk processing via Function.map()
    • Output: tracking DataFrame
    ↓
[4] Event Classification (CPU, 4 cores)
    • Load event_classifier_model.pkl
    • Extract 20 trajectory features
    • Random Forest inference
    • Output: events DataFrame
    ↓
[5] Output Generation (CPU, 4 cores)
    • Generate timestamped CSV
    • Render annotated video (optional)
    • Compute summary statistics
    ↓
[6] Upload Results (CPU, 2 cores)
    • Write to Azure Blob (processed/{user_id}/{job_id}/)
    • Update job status → "completed"
    • Fire webhook notification
```

---

## 7. Scaling & Cost Optimization

### Autoscaling Rules
```python
# GPU workers: scale to zero, burst to 50
@app.function(gpu="L4",
    min_containers=0,       # scale to zero when idle
    max_containers=50,      # max parallel GPU workers
    buffer_containers=1,    # 1 warm container during business hours
    scaledown_window=300,   # keep alive 5 min after last request
)
```

### Cost Estimates

| Scenario | Videos/Month | Est. Cost |
|----------|-------------|-----------|
| Individual researcher | 50 (1hr each) | ~$25-75 |
| Small lab | 500 videos | ~$250-750 |
| Multi-site study | 5,000 videos | ~$2,500-7,500 |

*Based on L4 GPU at $0.013/min, avg 1-min processing per 1-min video*

### Optimization Strategies
- **Two-Mode Adaptive:** Use CPU-only motion detection, GPU only when bees detected
- **Chunk Processing:** Split long videos, process chunks in parallel
- **Model Caching:** Keep models on Modal Volume, avoid re-downloading
- **Spot Instances:** Use GPU fallback lists for cheaper availability
- **Result Caching:** Cache nest positions per site (don't re-detect for same camera)

---

## 8. Security

- All API traffic over HTTPS
- API keys stored hashed (bcrypt) in database
- Azure Blob accessed via SAS tokens (time-limited, read-only for downloads)
- User credentials for external sources encrypted at rest (Azure Key Vault)
- CORS configured for web app domains only
- Rate limiting at API gateway level
- Video data isolated per user (no cross-tenant access)
