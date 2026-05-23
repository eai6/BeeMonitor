import hashlib

from django.utils import timezone
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed

from apps.accounts.models import APIKey
from apps.devices.models import Device


class APIKeyAuthentication(BaseAuthentication):
    """
    Custom authentication using Bearer token with BeeMonitor API keys.

    Expects the Authorization header in the format:
        Authorization: Bearer bmk_<key_type>_<random>

    Device-issued keys (``bmk_device_*``) skip this backend and are handled
    by ``DeviceKeyAuthentication`` so the request gets a ``Device`` in
    ``request.auth`` instead of a generic ``APIKey``.
    """

    keyword = "Bearer"

    def authenticate(self, request):
        auth_header = request.META.get("HTTP_AUTHORIZATION", "")

        if not auth_header.startswith(f"{self.keyword} "):
            return None

        raw_key = auth_header[len(self.keyword) + 1 :].strip()

        if not raw_key.startswith("bmk_"):
            return None
        # Device keys are handled by DeviceKeyAuthentication.
        if raw_key.startswith("bmk_device_"):
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


class DeviceKeyAuthentication(BaseAuthentication):
    """Bearer ``bmk_device_*`` → resolves to (owner User, Device).

    Used by the Pi upload endpoints (``/api/v1/uploads/*``). DRF stores the
    ``Device`` in ``request.auth``, so endpoint code can derive the S3 key
    prefix from ``request.auth.id`` and ``request.user.id`` without a
    second DB hit.
    """

    keyword = "Bearer"

    def authenticate(self, request):
        auth_header = request.META.get("HTTP_AUTHORIZATION", "")
        if not auth_header.startswith(f"{self.keyword} "):
            return None
        raw_key = auth_header[len(self.keyword) + 1:].strip()
        if not raw_key.startswith("bmk_device_"):
            return None

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        try:
            device = Device.objects.select_related("owner").get(
                key_hash=key_hash,
                is_active=True,
            )
        except Device.DoesNotExist:
            raise AuthenticationFailed("Invalid or inactive device key.")

        device.last_seen_at = timezone.now()
        device.save(update_fields=["last_seen_at"])

        return (device.owner, device)

    def authenticate_header(self, request):
        return self.keyword
