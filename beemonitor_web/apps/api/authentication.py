import hashlib

from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from apps.accounts.models import APIKey


class APIKeyAuthentication(BaseAuthentication):
    """
    Custom authentication using Bearer token with BeeMonitor API keys.

    Expects the Authorization header in the format:
        Authorization: Bearer bmk_<key_type>_<random>
    """

    keyword = "Bearer"

    def authenticate(self, request):
        auth_header = request.META.get("HTTP_AUTHORIZATION", "")

        if not auth_header.startswith(f"{self.keyword} "):
            return None

        raw_key = auth_header[len(self.keyword) + 1 :].strip()

        if not raw_key.startswith("bmk_"):
            return None

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        try:
            api_key = APIKey.objects.select_related("user").get(
                key_hash=key_hash,
                is_active=True,
            )
        except APIKey.DoesNotExist:
            raise AuthenticationFailed("Invalid or inactive API key.")

        # Update last_used_at timestamp
        api_key.last_used_at = timezone.now()
        api_key.save(update_fields=["last_used_at"])

        return (api_key.user, api_key)

    def authenticate_header(self, request):
        return self.keyword
