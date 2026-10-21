"""Slowing down guessing: sign-in, password-reset email and promo codes.

The counts live in the database, not the cache: there is no shared cache, and a
per-process one would give each gunicorn worker its own allowance.

Sign-in counts come from ``LoginEvent`` (every failure is already recorded
there). Too many recent failures for one username, or from one address, and
sign-in is refused until the window passes, even with the right password. The
address is the first X-Forwarded-For entry, which a client can forge, so the
per-username limit is the one that holds against a determined guesser.
"""

from datetime import timedelta

from django.contrib.auth.backends import ModelBackend
from django.core.exceptions import PermissionDenied
from django.utils import timezone

LOGIN_WINDOW = timedelta(minutes=15)
LOGIN_FAILURES_PER_USERNAME = 10
LOGIN_FAILURES_PER_IP = 30


def login_locked(username, ip) -> bool:
    from .models import LoginEvent

    since = timezone.now() - LOGIN_WINDOW
    failed = LoginEvent.objects.filter(success=False, created_at__gte=since)
    if username and failed.filter(username__iexact=username).count() >= LOGIN_FAILURES_PER_USERNAME:
        return True
    return bool(ip) and failed.filter(ip=ip).count() >= LOGIN_FAILURES_PER_IP


class ThrottledModelBackend(ModelBackend):
    """Django's backend, refusing while a username or address is locked out.

    PermissionDenied stops authentication outright (no other backend is tried)
    and is recorded as a failed sign-in like any other."""

    def authenticate(self, request, username=None, password=None, **kwargs):
        from .signals import client_ip

        if login_locked(username, client_ip(request)):
            raise PermissionDenied("Too many failed sign-ins. Try again in 15 minutes.")
        return super().authenticate(request, username=username, password=password, **kwargs)


def allow(key, limit, window) -> bool:
    """Record one attempt at ``key`` and say whether it is within ``limit`` per
    ``window``. Attempts past the limit are still recorded, so hammering keeps
    the window full."""
    from .models import Attempt

    now = timezone.now()
    Attempt.objects.filter(created_at__lt=now - timedelta(days=1)).delete()
    recent = Attempt.objects.filter(key=key, created_at__gte=now - window).count()
    Attempt.objects.create(key=key)
    return recent < limit
