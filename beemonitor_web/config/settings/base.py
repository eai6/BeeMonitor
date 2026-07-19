"""Base Django settings shared across all environments."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BASE_DIR.parent  # one level above beemonitor_web/, holds cloud/

# Load a local .env (repo root) for development convenience. Real environment
# variables — e.g. the App Runner task env in production — are NOT overridden,
# and in prod there is no .env file, so this is a no-op there.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:
    pass

# Make the repo root importable so Django code can `from cloud.storage...`
# (cloud/ is a sibling of beemonitor_web/, not a Django app).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-dev-only-change-me-in-production",
)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Third-party
    "rest_framework",
    "corsheaders",
    "anymail",
    # Project apps
    "apps.accounts",
    "apps.videos",
    "apps.analysis",
    "apps.sources",
    "apps.dashboard",
    "apps.developer",
    "apps.api",
    "apps.pwa",
    "apps.annotations",
    "apps.training",
    "apps.docs",
    "apps.devices",
    "apps.setup",
    "apps.monitor",
    "apps.pipelines",
]

MIDDLEWARE = [
    # Outermost so its timing covers the whole stack: logs any request > 5s
    # (App Runner hard-504s at 120s — slowness must show in logs first).
    "config.middleware.SlowRequestLogMiddleware",
    "django.middleware.security.SecurityMiddleware",
    # whitenoise serves /static/ in production without S3/CloudFront. Must
    # come immediately after SecurityMiddleware per the whitenoise docs.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# WhiteNoise: serve hashed + gzipped/brotli static files from STATIC_ROOT.
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": "whitenoise.storage.CompressedManifestStaticFilesStorage",
    },
}

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# REST Framework
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "apps.api.authentication.APIKeyAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticated",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_THROTTLE_CLASSES": [
        "apps.api.throttling.TierBasedThrottle",
    ],
}

# CORS
CORS_ALLOW_ALL_ORIGINS = False
CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(",")

# CSRF
CSRF_TRUSTED_ORIGINS = os.environ.get(
    "CSRF_TRUSTED_ORIGINS", "http://localhost:8000"
).split(",")

# AWS S3 — primary storage (bucket per logical area; see cloud/storage/config.py).
# Locally set these in .env; in AWS they come from ECS task env / Secrets Manager.
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_S3_BUCKET_RAW_VIDEOS = os.environ.get("AWS_S3_BUCKET_RAW_VIDEOS", "")
AWS_S3_BUCKET_PROCESSED = os.environ.get("AWS_S3_BUCKET_PROCESSED", "")
AWS_S3_BUCKET_MODELS = os.environ.get("AWS_S3_BUCKET_MODELS", "")
AWS_S3_BUCKET_USER_CONFIGS = os.environ.get("AWS_S3_BUCKET_USER_CONFIGS", "")

# Email — transactional mail (password reset) via AWS SES through django-anymail's
# SES v2 backend, which reuses the same boto3 IAM credential chain as S3 (no SMTP
# creds). Dev defaults to the console backend so no AWS/SES setup is needed locally;
# production.py overrides EMAIL_BACKEND to SES. SES can only send FROM a verified
# identity — DEFAULT_FROM_EMAIL must be an address on a domain verified in SES
# (NOT the awsapprunner.com host). See memory/21_password_reset_ses_design.md.
EMAIL_BACKEND = os.environ.get(
    "EMAIL_BACKEND", "django.core.mail.backends.console.EmailBackend")
DEFAULT_FROM_EMAIL = os.environ.get(
    "DEFAULT_FROM_EMAIL", "no-reply@beemonitor.edwardamoah.com")
SERVER_EMAIL = DEFAULT_FROM_EMAIL
ANYMAIL = {"AMAZON_SES_CLIENT_PARAMS": {"region_name": AWS_REGION}}
# How long a password-reset link stays valid (seconds); default 1 day.
PASSWORD_RESET_TIMEOUT = int(os.environ.get("PASSWORD_RESET_TIMEOUT", 60 * 60 * 24))

# SageMaker Async Inference (Phase 4 of memory/09_aws_migration_plan.md).
# The endpoint + buckets are provisioned by infra/aws-sagemaker/.
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
# SAM 3 auto-labeler endpoint (GPU async) — used by pre-annotation when labeler=sam3
# to seed YOLO training data on new-domain footage the current detector misses.
SAGEMAKER_SAM3_ENDPOINT_NAME = os.environ.get("SAGEMAKER_SAM3_ENDPOINT_NAME", "")
SAGEMAKER_INPUT_BUCKET = os.environ.get("SAGEMAKER_INPUT_BUCKET", "")
SAGEMAKER_OUTPUT_BUCKET = os.environ.get("SAGEMAKER_OUTPUT_BUCKET", "")
# Max GPU jobs to have invoking the SageMaker async endpoint at once. Jobs are
# created QUEUED and the background reconciler promotes them to PROCESSING only
# while fewer than this are in flight, so a large pipeline launch drains in waves
# instead of flooding the endpoint (which caps at ~6 concurrent invocations).
# Track this to the endpoint's real capacity; raise it when infra is scaled up.
SAGEMAKER_MAX_CONCURRENT = int(os.environ.get("SAGEMAKER_MAX_CONCURRENT", "6"))
# Fine-tuning (SageMaker training jobs) — role SageMaker assumes for the job and
# the training image. Set by the aws-sagemaker stack via App Runner env.
SAGEMAKER_TRAINING_ROLE_ARN = os.environ.get("SAGEMAKER_TRAINING_ROLE_ARN", "")
SAGEMAKER_TRAINING_IMAGE = os.environ.get("SAGEMAKER_TRAINING_IMAGE", "")

# GitHub token (gist scope) used to push a generated pipeline notebook to a public
# gist so "Open in Colab" can open it in the browser. Empty = fall back to a plain
# .ipynb download. The notebook holds no secrets (API key entered at runtime).
GITHUB_GIST_TOKEN = os.environ.get("GITHUB_GIST_TOKEN", "")

# AI pre-annotation defaults (overridable per request from the project page).
PREANNOTATE_SAMPLE_INTERVAL = int(os.environ.get("BEEMONITOR_PREANNOTATE_SAMPLE_INTERVAL", "10"))
PREANNOTATE_MAX_FRAMES = int(os.environ.get("BEEMONITOR_PREANNOTATE_MAX_FRAMES", "300"))
PREANNOTATE_CONFIDENCE = float(os.environ.get("BEEMONITOR_PREANNOTATE_CONFIDENCE", "0.15"))

# Modal
MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET", "")

# Devices — a unit is "online" if its last check-in is within its EXPECTED beat
# cadence, not a flat window: online if last_seen <= max(GRACE, MISSED_BEATS ×
# telemetry_interval). So a slow-beat cellular unit (which only contacts the
# server at its beat) isn't shown offline just for being between beats — it flips
# offline only after missing ~MISSED_BEATS of its own beats. GRACE is the floor
# for fast beats. (See Device.is_online / Device.online_window_seconds.)
DEVICE_ONLINE_GRACE_SECONDS = int(os.environ.get("DEVICE_ONLINE_GRACE_SECONDS", "180"))
DEVICE_ONLINE_MISSED_BEATS = int(os.environ.get("DEVICE_ONLINE_MISSED_BEATS", "3"))
# Base URL a field device's telemetry/uploader points at (baked into the setup
# guide's uploader.env block). The App Runner host in prod.
BEEMONITOR_DEVICE_API_BASE = os.environ.get(
    "BEEMONITOR_DEVICE_API_BASE", "https://mqnafc3ejc.us-east-1.awsapprunner.com")
# Public download URL for the pre-built "golden" SD image (.img.xz) users flash
# with Raspberry Pi Imager before browser-enrolling. Empty = no download link
# shown (the enrollment page falls back to "build/flash your own" guidance).
BEEMONITOR_GOLDEN_IMAGE_URL = os.environ.get("BEEMONITOR_GOLDEN_IMAGE_URL", "")
# Preferred: keep the (private) golden image in the models bucket and serve it via
# a short-lived presigned redirect (devices:golden_image), so the buckets stay
# locked down (no public-read). This is the S3 key within AWS_S3_BUCKET_MODELS;
# empty = disabled. A direct BEEMONITOR_GOLDEN_IMAGE_URL above still wins if set.
BEEMONITOR_GOLDEN_IMAGE_S3_KEY = os.environ.get(
    "BEEMONITOR_GOLDEN_IMAGE_S3_KEY", "golden/beemonitor-golden.img.xz")

# --- Taxonomic monitoring (activity frames + BioCLIP) ----------------------
# Max sampled frames accepted per activity (server-side guard; the device also
# self-limits with its own daily cap). Beyond this, extra frames are ignored.
# Counts crops AND the 'wide' source frames now archived alongside them, so the
# default allows a few of each per activity.
MONITOR_MAX_FRAMES_PER_ACTIVITY = int(os.environ.get("MONITOR_MAX_FRAMES_PER_ACTIVITY", "8"))
# BioCLIP insect-ID endpoint (SageMaker Serverless, CPU). Empty = perception
# disabled: frames still ingest + display, just no taxonomic classification, so
# Phase 0 keeps working until the endpoint is deployed.
SAGEMAKER_BIOCLIP_ENDPOINT_NAME = os.environ.get("SAGEMAKER_BIOCLIP_ENDPOINT_NAME", "")
# Kill-switch for automatic per-frame BioCLIP classification on ingest. Set
# MONITOR_BIOCLIP_AUTOPROCESS=false to stop spending SageMaker compute on incoming
# frames (e.g. while field sites have no insects yet) WITHOUT unconfiguring the
# endpoint — frames still ingest + display, and manual/backfill classification
# (manage.py classify_activities) still works. Off => the ingest hook no-ops.
MONITOR_BIOCLIP_AUTOPROCESS = os.environ.get(
    "MONITOR_BIOCLIP_AUTOPROCESS", "true").strip().lower() in {"1", "true", "yes", "on"}
# Keep this many ranked predictions per frame (top-1 drives the Detection; the
# rest live in Detection.raw for the reasoning agent later).
MONITOR_BIOCLIP_TOPK = int(os.environ.get("MONITOR_BIOCLIP_TOPK", "5"))
# Below this top-1 score, an activity is marked no_detection rather than asserting
# a likely-wrong ID (field crops are blurry/partial; better to flag than guess).
MONITOR_BIOCLIP_MIN_CONFIDENCE = float(os.environ.get("MONITOR_BIOCLIP_MIN_CONFIDENCE", "0.2"))
# Bounded worker pool for off-request-path classification (mirrors the analysis
# spawn pool so a frame burst can't exhaust DB connections).
MONITOR_CLASSIFY_MAX_WORKERS = int(os.environ.get("MONITOR_CLASSIFY_MAX_WORKERS", "3"))
# Phase 2 location priors: constrain BioCLIP to taxa that occur near the device
# (iNaturalist, GBIF fallback) instead of the whole Tree of Life. False =
# unconstrained ToL (Phase 1 behaviour).
MONITOR_USE_LOCATION_PRIORS = os.environ.get(
    "MONITOR_USE_LOCATION_PRIORS", "true").strip().lower() in {"1", "true", "yes", "on"}
MONITOR_PRIOR_RADIUS_KM = int(os.environ.get("MONITOR_PRIOR_RADIUS_KM", "50"))
MONITOR_PRIOR_MAX_TAXA = int(os.environ.get("MONITOR_PRIOR_MAX_TAXA", "300"))
MONITOR_PRIOR_TTL_SECONDS = int(os.environ.get("MONITOR_PRIOR_TTL_SECONDS", str(60 * 60 * 24 * 30)))
# Below this constrained top score, retry unconstrained ToL and keep whichever is
# more confident (so a locally-unrecorded true species isn't forced into a wrong
# neighbour). The precision/recall dial for the location constraint.
MONITOR_BIOCLIP_GENUS_FALLBACK = float(os.environ.get("MONITOR_BIOCLIP_GENUS_FALLBACK", "0.15"))

# --- AI setup assistant (Claude) -------------------------------------------
# Set ANTHROPIC_API_KEY in the environment to enable the in-dashboard tutor.
# When empty, the assistant degrades gracefully to a "not configured" notice.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Worker model for the tutor/debugger; Haiku for cheap routing/risk-classify.
ASSISTANT_MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-4-6")
ASSISTANT_FAST_MODEL = os.environ.get("ASSISTANT_FAST_MODEL", "claude-haiku-4-5-20251001")
# Most capable vision model — used for the interactive per-frame annotation
# helpers (AI Review + AI Detect), where quality matters and volume is low.
ASSISTANT_VISION_MODEL = os.environ.get("ASSISTANT_VISION_MODEL", "claude-opus-4-8")
ASSISTANT_MAX_TOKENS = int(os.environ.get("ASSISTANT_MAX_TOKENS", "1500"))
# Per-user safety valve so a runaway chat can't burn the account.
ASSISTANT_DAILY_MESSAGE_LIMIT = int(os.environ.get("ASSISTANT_DAILY_MESSAGE_LIMIT", "200"))
# Persist per-heartbeat GPS on DeviceHeartbeat (location history). False = only
# keep the latest fix on Device.
DEVICE_STORE_GPS_PER_HEARTBEAT = os.environ.get(
    "DEVICE_STORE_GPS_PER_HEARTBEAT", "true").strip().lower() in {"1", "true", "yes", "on"}

# Celery
CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"

# Credential encryption key
BEEMONITOR_CREDENTIAL_KEY = os.environ.get("BEEMONITOR_CREDENTIAL_KEY", "")

# File uploads (10 GB max)
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024 * 1024  # 10 GB
DATA_UPLOAD_MAX_NUMBER_FIELDS = 20000  # Support batch analysis of thousands of videos
FILE_UPLOAD_MAX_MEMORY_SIZE = 50 * 1024 * 1024  # 50 MB in memory, rest to disk

# Auth
LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/"
LOGOUT_REDIRECT_URL = "/accounts/login/"

# Logging
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{asctime} [{levelname}] {name}: {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "loggers": {
        "apps.training": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
        "apps.analysis": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}
