import hashlib
import secrets

from django.conf import settings
from django.db import models


class UserProfile(models.Model):
    class Tier(models.TextChoices):
        FREE = "free", "Free"
        STANDARD = "standard", "Standard"
        PRO = "pro", "Pro"

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="profile",
    )
    organization = models.CharField(max_length=200, blank=True)
    tier = models.CharField(
        max_length=20,
        choices=Tier.choices,
        default=Tier.FREE,
    )
    monthly_job_count = models.IntegerField(default=0)
    storage_used_bytes = models.BigIntegerField(default=0)

    def __str__(self):
        return f"{self.user.username} ({self.get_tier_display()})"


class APIKey(models.Model):
    class KeyType(models.TextChoices):
        LIVE = "live", "Live"
        TEST = "test", "Test"
        DEVICE = "device", "Device"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="api_keys",
    )
    key_hash = models.CharField(max_length=128, unique=True)
    prefix = models.CharField(max_length=12)
    name = models.CharField(max_length=100)
    key_type = models.CharField(
        max_length=10,
        choices=KeyType.choices,
    )
    permissions = models.JSONField(default=dict)
    rate_limit = models.IntegerField(default=60)
    is_active = models.BooleanField(default=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"

    def __str__(self):
        return f"{self.name} ({self.prefix}...)"

    @classmethod
    def create_key(cls, user, name, key_type):
        """Generate a new API key, hash it, save the model, and return (instance, raw_key)."""
        random_part = secrets.token_urlsafe(32)
        raw_key = f"bmk_{key_type}_{random_part}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        prefix = raw_key[:12]

        instance = cls.objects.create(
            user=user,
            key_hash=key_hash,
            prefix=prefix,
            name=name,
            key_type=key_type,
            is_active=True,
        )
        return instance, raw_key
