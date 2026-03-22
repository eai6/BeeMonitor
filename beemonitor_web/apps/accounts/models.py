import hashlib
import secrets

from django.conf import settings
from django.db import models
from django.utils import timezone


# Tier limits
# Credits are abstract units (1 credit ≈ 1 second of GPU time)
TIER_LIMITS = {
    "free": {
        "label": "Free",
        "monthly_credits": 3000,
        "max_concurrent_jobs": 10,
        "max_video_hours_per_month": 50,
    },
    "standard": {
        "label": "Standard",
        "monthly_credits": 25000,
        "max_concurrent_jobs": 50,
        "max_video_hours_per_month": 500,
    },
    "pro": {
        "label": "Pro",
        "monthly_credits": 100000,
        "max_concurrent_jobs": 50,
        "max_video_hours_per_month": 5000,
    },
    "enterprise": {
        "label": "Enterprise",
        "monthly_credits": 0,  # Unlimited
        "max_concurrent_jobs": 50,
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

    # Credits (abstract units, ~1 credit ≈ 1 second GPU time)
    monthly_credits = models.IntegerField(default=3000, help_text="Monthly credit limit")
    used_credits = models.IntegerField(default=0, help_text="Credits used this month")
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
    def remaining_credits(self) -> int:
        return max(0, self.monthly_credits - self.used_credits)

    @property
    def credit_usage_pct(self) -> int:
        if self.monthly_credits == 0:
            return 0
        return min(100, int(self.used_credits / self.monthly_credits * 100))

    def has_budget(self, estimated_credits: int) -> bool:
        """Check if user has enough credits."""
        if self.tier == self.Tier.ENTERPRISE:
            return True
        return self.remaining_credits >= estimated_credits

    def charge(self, credits: int, gpu_seconds: float = 0):
        """Deduct credits after a job completes."""
        self.used_credits += credits
        self.total_gpu_seconds += gpu_seconds
        self.total_jobs_submitted += 1
        self.save(update_fields=["used_credits", "total_gpu_seconds", "total_jobs_submitted"])

    def reset_monthly_credits(self):
        """Reset monthly usage."""
        limits = TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])
        self.monthly_credits = limits["monthly_credits"]
        self.max_concurrent_jobs = limits["max_concurrent_jobs"]
        self.used_credits = 0
        self.credit_reset_date = timezone.now().date() + timezone.timedelta(days=30)
        self.save(update_fields=[
            "monthly_credits", "max_concurrent_jobs",
            "used_credits", "credit_reset_date",
        ])

    @property
    def tier_limits(self) -> dict:
        return TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])


class Coupon(models.Model):
    class CouponType(models.TextChoices):
        CREDITS = "credits", "Credits"
        TIER_UPGRADE = "tier_upgrade", "Tier Upgrade"

    code = models.CharField(max_length=50, unique=True)
    coupon_type = models.CharField(max_length=20, choices=CouponType.choices)
    credits_amount = models.IntegerField(default=0, help_text="Bonus credits to add")
    upgrade_tier = models.CharField(
        max_length=20,
        choices=UserProfile.Tier.choices,
        blank=True,
        default="",
        help_text="Tier to upgrade user to",
    )
    upgrade_duration_days = models.IntegerField(default=30, help_text="How long tier upgrade lasts")
    max_redemptions = models.IntegerField(default=0, help_text="0 = unlimited")
    times_redeemed = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.code} ({self.get_coupon_type_display()})"

    @property
    def is_valid(self):
        if not self.is_active:
            return False
        if self.max_redemptions > 0 and self.times_redeemed >= self.max_redemptions:
            return False
        if self.expires_at and timezone.now() > self.expires_at:
            return False
        return True


class CouponRedemption(models.Model):
    coupon = models.ForeignKey(Coupon, on_delete=models.CASCADE, related_name="redemptions")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="coupon_redemptions",
    )
    redeemed_at = models.DateTimeField(auto_now_add=True)
    credits_added = models.IntegerField(default=0)
    previous_tier = models.CharField(max_length=20, blank=True)
    new_tier = models.CharField(max_length=20, blank=True)

    class Meta:
        unique_together = ("coupon", "user")

    def __str__(self):
        return f"{self.user.username} redeemed {self.coupon.code}"


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
