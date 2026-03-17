# BeeMonitor Django Web Application Design

## Overview

A Django-based web application that serves as the primary web interface for BeeMonitor Cloud. Users can upload videos, manage data sources, run analyses, view results, and manage API keys — mirroring the desktop app functionality through a browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DJANGO WEB APP                            │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Frontend   │  │   Django     │  │   Background     │  │
│  │   Templates  │  │   Views      │  │   Tasks          │  │
│  │   + HTMX     │  │   + DRF API  │  │   (Celery)       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────────┘  │
│         │                 │                   │              │
│  ┌──────┴─────────────────┴───────────────────┴──────────┐  │
│  │                 Django ORM + Models                     │  │
│  └────────────────────────┬───────────────────────────────┘  │
│                           │                                  │
│  ┌────────────────────────┴───────────────────────────────┐  │
│  │              PostgreSQL Database                        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────┐
│  Modal Cloud Backend │  │  Azure Blob Storage      │
│  (GPU Processing)    │  │  (Videos + Results)      │
└──────────────────────┘  └──────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend | Django 5.x | Batteries-included, ORM, admin, auth |
| API | Django REST Framework | Serialization, auth, throttling |
| Frontend | Django Templates + HTMX + Alpine.js | Server-rendered, minimal JS, fast |
| CSS | Tailwind CSS | Utility-first, responsive |
| Database | PostgreSQL | Robust, JSON support, full-text search |
| Cache | Redis | Sessions, task queue, caching |
| Task Queue | Celery + Redis | Async job management |
| File Storage | Azure Blob (django-storages) | Direct integration |
| Auth | django-allauth | Social auth (Google, GitHub) |
| Deployment | Azure App Service or Railway | Managed hosting |

---

## Django Project Structure

```
beemonitor_web/
├── manage.py
├── config/
│   ├── settings/
│   │   ├── base.py
│   │   ├── development.py
│   │   └── production.py
│   ├── urls.py
│   ├── wsgi.py
│   └── celery.py
│
├── apps/
│   ├── accounts/              # User management
│   │   ├── models.py          # UserProfile, APIKey
│   │   ├── views.py           # Registration, login, profile
│   │   ├── forms.py
│   │   ├── urls.py
│   │   └── templates/accounts/
│   │
│   ├── videos/                # Video management
│   │   ├── models.py          # Video, VideoSource
│   │   ├── views.py           # Upload, list, detail
│   │   ├── forms.py           # Upload form, source config
│   │   ├── tasks.py           # Celery: ingest from sources
│   │   ├── urls.py
│   │   └── templates/videos/
│   │
│   ├── analysis/              # Processing jobs
│   │   ├── models.py          # Job, JobResult
│   │   ├── views.py           # Submit, status, results
│   │   ├── tasks.py           # Celery: trigger Modal processing
│   │   ├── urls.py
│   │   └── templates/analysis/
│   │
│   ├── sources/               # Data source connections
│   │   ├── models.py          # DataSource (S3, Azure, GCS, GDrive)
│   │   ├── views.py           # Connect, test, manage
│   │   ├── connectors.py      # Cloud-specific connection logic
│   │   ├── urls.py
│   │   └── templates/sources/
│   │
│   ├── dashboard/             # Main dashboard
│   │   ├── views.py           # Overview, stats, activity feed
│   │   ├── urls.py
│   │   └── templates/dashboard/
│   │
│   ├── developer/             # API key management
│   │   ├── models.py          # APIKey, UsageLog
│   │   ├── views.py           # Key generation, docs
│   │   ├── urls.py
│   │   └── templates/developer/
│   │
│   └── api/                   # REST API (DRF)
│       ├── serializers.py
│       ├── views.py
│       ├── permissions.py
│       ├── throttling.py
│       └── urls.py
│
├── templates/
│   ├── base.html              # Base layout (Tailwind + HTMX)
│   ├── components/            # Reusable template components
│   │   ├── navbar.html
│   │   ├── sidebar.html
│   │   ├── video_card.html
│   │   ├── job_status.html
│   │   └── stats_widget.html
│   └── pages/
│       ├── landing.html
│       └── pricing.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── img/
│
└── requirements/
    ├── base.txt
    ├── development.txt
    └── production.txt
```

---

## Data Models

```python
# accounts/models.py
class UserProfile(models.Model):
    user = models.OneToOneField(User)
    organization = models.CharField(max_length=200, blank=True)
    tier = models.CharField(choices=["free", "standard", "pro"], default="free")
    monthly_job_count = models.IntegerField(default=0)
    storage_used_bytes = models.BigIntegerField(default=0)

class APIKey(models.Model):
    user = models.ForeignKey(User)
    key_hash = models.CharField(max_length=64, unique=True)  # bcrypt hash
    prefix = models.CharField(max_length=8)  # "bmk_live" visible part
    name = models.CharField(max_length=100)
    key_type = models.CharField(choices=["live", "test", "device"])
    permissions = models.JSONField(default=dict)
    rate_limit = models.IntegerField(default=60)  # req/min
    is_active = models.BooleanField(default=True)
    last_used_at = models.DateTimeField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

# sources/models.py
class DataSource(models.Model):
    user = models.ForeignKey(User)
    name = models.CharField(max_length=200)
    source_type = models.CharField(choices=[
        "aws_s3", "azure_blob", "gcs", "google_drive"
    ])
    config_encrypted = models.BinaryField()  # Fernet-encrypted credentials
    is_connected = models.BooleanField(default=False)
    last_synced_at = models.DateTimeField(null=True)

# videos/models.py
class Video(models.Model):
    user = models.ForeignKey(User)
    source = models.ForeignKey(DataSource, null=True)
    title = models.CharField(max_length=300)
    azure_blob_path = models.CharField(max_length=500)
    file_size_bytes = models.BigIntegerField()
    duration_seconds = models.FloatField(null=True)
    resolution = models.CharField(max_length=20)  # "1280x720"
    fps = models.FloatField(null=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(choices=["uploading", "ready", "processing", "archived"])
    metadata = models.JSONField(default=dict)

# analysis/models.py
class Job(models.Model):
    user = models.ForeignKey(User)
    video = models.ForeignKey(Video)
    modal_job_id = models.CharField(max_length=100, unique=True)
    status = models.CharField(choices=[
        "queued", "ingesting", "processing", "post_processing",
        "completed", "failed", "cancelled"
    ])
    config = models.JSONField()  # detection_mode, thresholds, etc.
    progress_pct = models.IntegerField(default=0)
    started_at = models.DateTimeField(null=True)
    completed_at = models.DateTimeField(null=True)
    error_message = models.TextField(blank=True)
    compute_cost_usd = models.DecimalField(max_digits=8, decimal_places=4, null=True)

class JobResult(models.Model):
    job = models.OneToOneField(Job)
    events_csv_path = models.CharField(max_length=500)
    tracking_csv_path = models.CharField(max_length=500)
    annotated_video_path = models.CharField(max_length=500, blank=True)
    total_events = models.IntegerField(default=0)
    entry_count = models.IntegerField(default=0)
    exit_count = models.IntegerField(default=0)
    unique_tracks = models.IntegerField(default=0)
    nest_count = models.IntegerField(default=0)
    summary_stats = models.JSONField(default=dict)
```

---

## Page Layout & User Flow

### 1. Landing Page
- Hero: "AI-Powered Bee Monitoring at Scale"
- Features: Multi-cloud, GPU processing, API access
- CTA: Sign up / Try demo

### 2. Dashboard (`/dashboard/`)
- Recent jobs with status indicators
- Storage usage bar
- Quick stats: total videos, events detected, active sources
- Activity feed

### 3. Videos (`/videos/`)
- Grid/list view of uploaded videos with thumbnails
- Upload button (drag-and-drop, multipart)
- Filter by status, date, source
- Video detail page with metadata + linked jobs

### 4. Data Sources (`/sources/`)
- Connected sources list with status indicators
- "Add Source" wizard (S3, Azure, GCS, Google Drive)
- Test connection button
- Browse remote files → select for ingestion

### 5. Analysis (`/analysis/`)
- New job form: select video(s) + config options
- Job queue with real-time status (HTMX polling)
- Results viewer: events table, charts, video player
- Download buttons for CSV / annotated video
- Batch job submission

### 6. Results Viewer (`/analysis/{job_id}/results/`)
- **Events Table** — Sortable, filterable, downloadable
- **Activity Chart** — Hourly entry/exit counts (Chart.js)
- **Video Player** — Annotated video with event timeline
- **Nest Heatmap** — Activity per nest tube
- **Export** — CSV, JSON, PDF report

### 7. Developer Portal (`/developer/`)
- API key management (create, revoke, view usage)
- Interactive API docs (Swagger/ReDoc)
- Code examples (Python, cURL, JavaScript)
- Usage dashboard (requests, compute time, costs)
- Webhook configuration

### 8. Settings (`/settings/`)
- Profile, organization
- Default analysis config
- Notification preferences
- Billing / tier management

---

## Key Features

### Real-Time Job Updates (HTMX)
```html
<!-- Job status with auto-polling -->
<div hx-get="/analysis/{{ job.id }}/status/"
     hx-trigger="every 3s"
     hx-swap="innerHTML">
  <span class="badge badge-{{ job.status }}">{{ job.status }}</span>
  <div class="progress-bar" style="width: {{ job.progress_pct }}%"></div>
</div>
```

### Video Upload (Chunked)
- Client-side chunking (5 MB chunks) via JavaScript
- Resume on failure
- Progress bar with HTMX
- Direct-to-Azure upload with SAS tokens (bypass Django for large files)

### Multi-Source Ingestion
```python
# sources/connectors.py
class S3Connector:
    def list_files(self, prefix="") -> list
    def download_to_azure(self, key, azure_path) -> str

class AzureBlobConnector:
    def list_files(self, prefix="") -> list
    def copy_internal(self, source_path, dest_path) -> str

class GCSConnector:
    def list_files(self, prefix="") -> list
    def download_to_azure(self, key, azure_path) -> str

class GoogleDriveConnector:
    def list_files(self, folder_id) -> list
    def download_to_azure(self, file_id, azure_path) -> str
```

### Modal Integration (Celery Tasks)
```python
# analysis/tasks.py
@shared_task
def submit_analysis_job(job_id):
    job = Job.objects.get(id=job_id)

    # Call Modal API to trigger processing
    modal_response = modal_client.functions.call(
        "beemonitor-cloud/process_video",
        job_id=str(job.modal_job_id),
        video_source={"azure_path": job.video.azure_blob_path},
        user_id=str(job.user.id),
        config=job.config,
    )

    job.status = "processing"
    job.save()

@shared_task
def check_job_status(job_id):
    """Periodic task to sync Modal job status."""
    ...
```

---

## Deployment

```
Production Stack:
├── Azure App Service (Django + Gunicorn)
├── PostgreSQL (Azure Database for PostgreSQL)
├── Redis (Azure Cache for Redis)
├── Celery Workers (Azure Container Instances)
├── Azure Blob Storage (media files)
├── Azure CDN (static files)
└── Modal.com (GPU processing)
```

### Environment Variables
```
DATABASE_URL=postgres://...
REDIS_URL=redis://...
AZURE_STORAGE_CONNECTION_STRING=...
AZURE_STORAGE_CONTAINER=beemonitor-storage
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
SECRET_KEY=...
ALLOWED_HOSTS=app.beemonitor.io
```
