"""Production settings — PostgreSQL, security hardening."""

import os

from config.settings.base import *  # noqa

DEBUG = False
ALLOWED_HOSTS = os.environ.get("ALLOWED_HOSTS", "").split(",")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.environ.get("DB_NAME", "beemonitor"),
        "USER": os.environ.get("DB_USER", "beemonitor"),
        "PASSWORD": os.environ.get("DB_PASSWORD", ""),
        "HOST": os.environ.get("DB_HOST", "localhost"),
        "PORT": os.environ.get("DB_PORT", "5432"),
        "OPTIONS": {
            "sslmode": "require",
        },
        # gthread workers open a Postgres connection per request thread; keep
        # them alive briefly so 2 workers × 4 threads don't reconnect per hit.
        "CONN_MAX_AGE": int(os.environ.get("DB_CONN_MAX_AGE", "60")),
    }
}

# Email — send real mail via AWS SES in production (django-anymail SES v2 backend).
# Still env-overridable so it can be swapped without a code change.
EMAIL_BACKEND = os.environ.get("EMAIL_BACKEND", "anymail.backends.amazon_ses.EmailBackend")

# Security
SECURE_SSL_REDIRECT = False  # ALB / fronting load balancer handles HTTPS termination
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_HSTS_SECONDS = 31536000
SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_CONTENT_TYPE_NOSNIFF = True
