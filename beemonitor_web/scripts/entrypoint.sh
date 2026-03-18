#!/bin/bash
set -e

echo "Running migrations..."
python manage.py migrate --noinput

echo "Backfilling video timestamps..."
python manage.py backfill_video_timestamps 2>/dev/null || true

echo "Collecting static files..."
python manage.py collectstatic --noinput 2>/dev/null || true

echo "Starting Gunicorn..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
