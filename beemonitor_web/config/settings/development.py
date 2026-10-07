"""Development settings — SQLite, DEBUG=True."""

import os

from config.settings.base import *  # noqa

DEBUG = True
ALLOWED_HOSTS = ["*"]

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": os.environ.get("DB_PATH", str(BASE_DIR / "db.sqlite3")),  # noqa: F405
    }
}

CORS_ALLOW_ALL_ORIGINS = True

# Tests render templates with {% static %} but never run collectstatic, so the
# hashed manifest the production storage reads does not exist. Unhashed names
# there; the image still builds and serves the manifest (Dockerfile).
import sys as _sys  # noqa: E402

if "test" in _sys.argv:
    STORAGES = {**STORAGES, "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"}}
