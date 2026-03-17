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
