#!/bin/bash
set -e

# Use /home for persistent storage on Azure App Service
# /home survives container restarts and redeploys
export DB_PATH="/home/db.sqlite3"

echo "Using database at: $DB_PATH"
echo "Running migrations..."
python manage.py migrate --noinput

echo "Collecting static files..."
python manage.py collectstatic --noinput 2>/dev/null || true

echo "Starting Gunicorn..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
