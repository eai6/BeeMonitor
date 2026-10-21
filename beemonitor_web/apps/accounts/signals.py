from django.contrib.auth import get_user_model
from django.contrib.auth.signals import user_logged_in, user_login_failed
from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import UserProfile

User = get_user_model()


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """Auto-create a UserProfile when a new User is created."""
    if created:
        UserProfile.objects.get_or_create(user=instance)


def client_ip(request):
    """The caller's address, for showing on the dashboard only: the first
    X-Forwarded-For entry is whatever the client sent, so never trust it."""
    if request is None:
        return None
    forwarded = request.META.get("HTTP_X_FORWARDED_FOR", "")
    ip = forwarded.split(",")[0].strip() if forwarded else request.META.get("REMOTE_ADDR", "")
    return ip or None


def _method(request):
    from .models import LoginEvent

    path = getattr(request, "path", "") or ""
    if path.startswith("/admin/"):
        return LoginEvent.Method.ADMIN
    if path.startswith("/api/"):
        return LoginEvent.Method.API
    return LoginEvent.Method.WEB


def record_login(request, user=None, username="", success=True):
    from .models import LoginEvent

    LoginEvent.objects.create(
        user=user, username=(username or getattr(user, "username", ""))[:254], success=success,
        method=_method(request), ip=client_ip(request),
        user_agent=(request.META.get("HTTP_USER_AGENT", "") if request is not None else "")[:300],
    )
    LoginEvent.prune()


@receiver(user_logged_in)
def on_login(sender, request, user, **kwargs):
    record_login(request, user=user)


@receiver(user_login_failed)
def on_login_failed(sender, credentials, request=None, **kwargs):
    username = (credentials or {}).get("username", "") or ""
    record_login(request, user=User.objects.filter(username=username).first() if username else None,
                 username=username, success=False)
