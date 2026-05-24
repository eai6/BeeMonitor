"""Base Django settings shared across all environments."""

import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPO_ROOT = BASE_DIR.parent  # one level above beemonitor_web/, holds cloud/

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
]

MIDDLEWARE = [
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

# SageMaker Async Inference (Phase 4 of memory/09_aws_migration_plan.md).
# The endpoint + buckets are provisioned by infra/aws-sagemaker/.
SAGEMAKER_ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "")
SAGEMAKER_INPUT_BUCKET = os.environ.get("SAGEMAKER_INPUT_BUCKET", "")
SAGEMAKER_OUTPUT_BUCKET = os.environ.get("SAGEMAKER_OUTPUT_BUCKET", "")

# Modal
MODAL_TOKEN_ID = os.environ.get("MODAL_TOKEN_ID", "")
MODAL_TOKEN_SECRET = os.environ.get("MODAL_TOKEN_SECRET", "")

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
