#!/bin/bash
set -e

echo "Running migrations..."
python manage.py migrate --noinput

echo "Backfilling video timestamps..."
python manage.py backfill_video_timestamps || true

echo "Cleaning up stale jobs..."
python manage.py cleanup_stale_jobs || true

echo "Collecting static files..."
python manage.py collectstatic --noinput 2>/dev/null || true

echo "Starting Gunicorn..."
# threads>1 switches to the gthread worker: one slow request can no longer
# starve the site. Timeout sits just UNDER App Runner's hard 120s cap — past
# that the proxy has already 504'd, so a longer-running request only pins a
# thread serving a dead connection. A worker is replaced after ~300 requests,
# so memory one big request made it take is given back (memory/76).
gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --threads 4 \
    --timeout 115 \
    --max-requests 300 \
    --max-requests-jitter 50 \
    --access-logfile - \
    --error-logfile - &

GUNICORN_PID=$!

# Each run's slim copy of its results, for pages that list many clips
# (memory/76). Pages build any still missing as they meet them.
(python manage.py fill_run_summaries || true) &

# Fold stored heartbeats into the per-minute health history (idempotent).
(python manage.py backfill_health_samples || true) &

# Run foraging trip backfills in background after Gunicorn is up
echo "Backfilling foraging trips in background..."
(python manage.py backfill_foraging_trips && echo "Computing daily cross-video trips..." && python manage.py compute_daily_trips) &

# Wait for Gunicorn (foreground process)
wait $GUNICORN_PID
