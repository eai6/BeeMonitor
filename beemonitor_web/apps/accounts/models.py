import hashlib
import secrets

from django.conf import settings
from django.db import models
from django.utils import timezone


# Tier limits
TIER_LIMITS = {
    "free": {
        "label": "Free",
        "monthly_credit_cents": 3000,  # $30
        "max_concurrent_jobs": 10,
        "max_video_hours_per_month": 50,
    },
    "standard": {
        "label": "Standard",
        "monthly_credit_cents": 25000,  # $250
        "max_concurrent_jobs": 50,
        "max_video_hours_per_month": 500,
    },
    "pro": {
        "label": "Pro",
        "monthly_credit_cents": 100000,  # $1,000
        "max_concurrent_jobs": 200,
        "max_video_hours_per_month": 5000,
    },
    "enterprise": {
        "label": "Enterprise",
        "monthly_credit_cents": 0,  # Unlimited
        "max_concurrent_jobs": 500,
        "max_video_hours_per_month": 0,  # Unlimited
    },
}


class UserProfile(models.Model):
    class Tier(models.TextChoices):
        FREE = "free", "Free"
        STANDARD = "standard", "Standard"
        PRO = "pro", "Pro"
        ENTERPRISE = "enterprise", "Enterprise"

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

    # Credits (in cents to avoid float issues)
    monthly_credit_cents = models.IntegerField(default=3000, help_text="Monthly credit limit in cents")
    used_credit_cents = models.IntegerField(default=0, help_text="Credits used this month in cents")
    credit_reset_date = models.DateField(null=True, blank=True, help_text="Next monthly reset date")

    # Quotas
    max_concurrent_jobs = models.IntegerField(default=10)
    total_jobs_submitted = models.IntegerField(default=0)
    total_gpu_seconds = models.FloatField(default=0, help_text="Lifetime GPU seconds consumed")

    # Legacy
    monthly_job_count = models.IntegerField(default=0)
    storage_used_bytes = models.BigIntegerField(default=0)

    def __str__(self):
        return f"{self.user.username} ({self.get_tier_display()})"

    @property
    def remaining_credit_cents(self) -> int:
        return max(0, self.monthly_credit_cents - self.used_credit_cents)

    @property
    def remaining_credit_usd(self) -> float:
        return self.remaining_credit_cents / 100

    @property
    def used_credit_usd(self) -> float:
        return self.used_credit_cents / 100

    @property
    def monthly_credit_usd(self) -> float:
        return self.monthly_credit_cents / 100

    @property
    def credit_usage_pct(self) -> int:
        if self.monthly_credit_cents == 0:
            return 0
        return min(100, int(self.used_credit_cents / self.monthly_credit_cents * 100))

    def has_budget(self, estimated_cost_cents: int) -> bool:
        """Check if user has enough credits for estimated cost."""
        if self.tier == self.Tier.ENTERPRISE:
            return True  # Unlimited
        return self.remaining_credit_cents >= estimated_cost_cents

    def charge(self, cost_cents: int, gpu_seconds: float = 0):
        """Deduct credits after a job completes."""
        self.used_credit_cents += cost_cents
        self.total_gpu_seconds += gpu_seconds
        self.total_jobs_submitted += 1
        self.save(update_fields=["used_credit_cents", "total_gpu_seconds", "total_jobs_submitted"])

    def reset_monthly_credits(self):
        """Reset monthly usage (called by management command)."""
        limits = TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])
        self.monthly_credit_cents = limits["monthly_credit_cents"]
        self.max_concurrent_jobs = limits["max_concurrent_jobs"]
        self.used_credit_cents = 0
        self.credit_reset_date = timezone.now().date() + timezone.timedelta(days=30)
        self.save(update_fields=[
            "monthly_credit_cents", "max_concurrent_jobs",
            "used_credit_cents", "credit_reset_date",
        ])

    @property
    def tier_limits(self) -> dict:
        return TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])


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
