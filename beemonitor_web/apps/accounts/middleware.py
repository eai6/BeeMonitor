"""Last seen: when each account last made a web request or API call."""

from django.utils import timezone

# Written at most this often per account, so a busy page or a unit uploading
# all day costs one UPDATE every few minutes, not one per request.
LAST_SEEN_EVERY_SECONDS = 300


class LastSeenMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        # Checked after the view: API-key and device authentication happen in
        # DRF, which sets request.user only once the view runs.
        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated:
            self._touch(request, user)
        return response

    @staticmethod
    def _touch(request, user):
        from .models import UserProfile

        now = timezone.now()
        session = getattr(request, "session", None)
        key = "_last_seen_at"
        if session is not None:
            last = session.get(key)
            if last and now.timestamp() - last < LAST_SEEN_EVERY_SECONDS:
                return
        cutoff = now - timezone.timedelta(seconds=LAST_SEEN_EVERY_SECONDS)
        # The filter makes this a no-op when another request wrote it recently.
        from django.db.models import Q
        UserProfile.objects.filter(user_id=user.pk).filter(
            Q(last_seen_at__isnull=True) | Q(last_seen_at__lt=cutoff)).update(last_seen_at=now)
        if session is not None and not session.is_empty():
            session[key] = now.timestamp()
