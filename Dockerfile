# BeeMonitor Django web app image — for App Runner.
#
# Build context is the repo root so we can include both beemonitor_web/ (the
# Django project) and cloud/ (the cloud.storage package the Django code now
# imports from). The legacy Dockerfile under beemonitor_web/ assumed the
# context started inside that directory; this one supersedes it.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_ENV=production \
    PYTHONPATH=/app/beemonitor_web:/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so they cache across code changes.
COPY beemonitor_web/requirements/ /app/beemonitor_web/requirements/
RUN pip install --no-cache-dir -r /app/beemonitor_web/requirements/base.txt django-filter

# Application code — Django project plus the cloud package it imports.
COPY beemonitor_web/ /app/beemonitor_web/
COPY cloud/ /app/cloud/

# collectstatic at build time so we don't pay it on every container start;
# whitenoise serves out of STATIC_ROOT (beemonitor_web/staticfiles).
RUN cd /app/beemonitor_web && \
    mkdir -p staticfiles && \
    python manage.py collectstatic --noinput 2>/dev/null || true

RUN chmod +x /app/beemonitor_web/scripts/entrypoint.sh

EXPOSE 8000

WORKDIR /app/beemonitor_web
CMD ["scripts/entrypoint.sh"]
