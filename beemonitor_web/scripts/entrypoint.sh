#!/bin/bash
set -e

echo "Running migrations..."
python manage.py migrate --noinput

echo "Backfilling video timestamps..."
python manage.py backfill_video_timestamps 2>/dev/null || true

echo "Ensuring admin tier..."
python -c "
import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings.production')
django.setup()
from django.contrib.auth.models import User
from apps.accounts.models import UserProfile, TIER_LIMITS
for u in User.objects.filter(is_superuser=True):
    p, _ = UserProfile.objects.get_or_create(user=u)
    if p.tier != 'enterprise':
        limits = TIER_LIMITS['enterprise']
        p.tier = 'enterprise'
        p.monthly_credits = limits['monthly_credits']
        p.max_concurrent_jobs = limits['max_concurrent_jobs']
        p.used_credits = 0
        p.save(update_fields=['tier', 'monthly_credits', 'max_concurrent_jobs', 'used_credits'])
        print(f'Upgraded {u.username} to enterprise')
" 2>/dev/null || true

echo "Backfilling foraging trips..."
python manage.py backfill_foraging_trips 2>/dev/null || true

echo "Cleaning up stale jobs..."
python manage.py cleanup_stale_jobs 2>/dev/null || true

echo "Collecting static files..."
python manage.py collectstatic --noinput 2>/dev/null || true

echo "Starting Gunicorn..."
exec gunicorn config.wsgi:application \
    --bind 0.0.0.0:8000 \
    --workers 2 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile -
