"""Django settings — auto-select based on DJANGO_ENV."""

import os

env = os.environ.get("DJANGO_ENV", "development")

if env == "production":
    from config.settings.production import *  # noqa
else:
    from config.settings.development import *  # noqa
