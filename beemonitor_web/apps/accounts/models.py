import hashlib
import secrets

from django.conf import settings
from django.db import models
from django.utils import timezone


# Tier limits
# ``monthly_credits`` / ``max_video_hours_per_month`` are the legacy GPU-second
# meter. The meter that gates GPU work is ``monthly_gpu_credits``: credits in the
# unit of apps/accounts/quota.py (1 credit = 1 GPU-hour on the video machine),
# None = unlimited. See memory/50_tiers_quotas_and_platform_admin.md.
TIER_LIMITS = {
    "free": {
        "label": "Free",
        "monthly_gpu_credits": 10,
        "monthly_credits": 3000,
        "max_video_hours_per_month": 50,
    },
    "standard": {
        "label": "Standard",
        "monthly_gpu_credits": 10,
        "monthly_credits": 25000,
        "max_video_hours_per_month": 500,
    },
    "pro": {
        "label": "Pro",
        "monthly_gpu_credits": None,
        "monthly_credits": 100000,
        "max_video_hours_per_month": 5000,
    },
    "enterprise": {
        "label": "Enterprise",
        "monthly_gpu_credits": None,
        "monthly_credits": 0,  # Unlimited
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
    # Unused: there is no per-user GPU limit; the platform scales to the work
    # and queues behind one global SageMaker cap (analysis.views._drain_queue).
    max_concurrent_jobs = models.IntegerField(default=10)
    total_jobs_submitted = models.IntegerField(default=0)
    total_gpu_seconds = models.FloatField(default=0, help_text="Lifetime GPU seconds consumed")

    # Support flag — when True, the user can READ any owner's resources
    # via the read-bypass views (DeviceListView, JobListView, etc.).
    # WRITES still go through the regular ownership scoping; support staff
    # cannot mutate someone else's data without explicit Django admin.
    is_support = models.BooleanField(
        default=False,
        help_text="Allow read-only cross-tenant access (analytics, debugging). Writes still owner-scoped.",
    )

    # GPU credits (apps/accounts/quota.py). The month's free allowance comes
    # from the tier; ``monthly_gpu_credits_used`` counts what was taken from it
    # since ``credit_period_start``. ``granted_gpu_credits`` is what was granted
    # on top (requests, promo codes) and never resets; it is spent after the
    # month's allowance and may go slightly negative when a run overruns.
    credit_period_start = models.DateField(null=True, blank=True)
    monthly_gpu_credits_used = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    granted_gpu_credits = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    # Set on the platform dashboard for one account: replaces the tier's free
    # monthly credits (None = the tier's). A big customer is a Pro or Standard
    # account with its own numbers, not a tier of its own.
    monthly_gpu_credits_override = models.DecimalField(max_digits=12, decimal_places=4,
                                                       null=True, blank=True)
    # When a granted tier (Pro, a promo-code trial) ends; the account drops to
    # Standard on the next credit check after it (quota._expire_tier).
    tier_expires_at = models.DateTimeField(null=True, blank=True)
    # Last web request or API call, written at most every 5 minutes
    # (accounts.middleware.LastSeenMiddleware).
    last_seen_at = models.DateTimeField(null=True, blank=True)

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

    def reset_monthly_credits(self):
        """Reset monthly usage."""
        limits = TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])
        self.monthly_credits = limits["monthly_credits"]
        self.used_credits = 0
        self.credit_reset_date = timezone.now().date() + timezone.timedelta(days=30)
        self.save(update_fields=["monthly_credits", "used_credits", "credit_reset_date"])

    @property
    def tier_limits(self) -> dict:
        limits = TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])
        if self.monthly_gpu_credits_override is not None:
            limits = {**limits, "monthly_gpu_credits": self.monthly_gpu_credits_override}
        return limits


class Coupon(models.Model):
    class CouponType(models.TextChoices):
        CREDITS = "credits", "Credits"
        TIER_UPGRADE = "tier_upgrade", "Tier Upgrade"

    code = models.CharField(max_length=50, unique=True)
    coupon_type = models.CharField(max_length=20, choices=CouponType.choices)
    credits_amount = models.IntegerField(default=0, help_text="Credits to grant (1 credit = 1 GPU-hour on the video machine)")
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
    note = models.CharField(max_length=200, blank=True, default="", help_text="Only admins see it")

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


class UsageEvent(models.Model):
    """One charge for GPU work: what ran, on which machine, for how long.

    Every credit a user spends is one of these, so the totals on the profile can
    be checked against them, and estimates come from them (quota.estimate). The
    (source_type, source_id) pair is unique, which is what makes charging a
    finished job twice — two pollers, a retried collect — a no-op.
    """

    class Kind(models.TextChoices):
        ANALYSIS = "analysis", "Video analysis"
        PHOTO = "photo", "Photo detection"
        ANNOTATED_VIDEO = "annotated_video", "Annotated video"
        AUTO_LABEL = "auto_label", "Auto-label"
        SAMPLING = "sampling", "Sampling"
        TRAINING = "training", "Training"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="usage_events")
    kind = models.CharField(max_length=20, choices=Kind.choices)
    gpu_seconds = models.FloatField(help_text="Billed instance seconds")
    instance_type = models.CharField(max_length=40)
    credits = models.DecimalField(max_digits=12, decimal_places=4)
    cost_usd = models.DecimalField(max_digits=12, decimal_places=4, default=0)
    # Work done, in the kind's unit (quota.UNITS): seconds of footage, photos,
    # frames, clips or image-epochs. Estimates are gpu_seconds per unit.
    units = models.FloatField(default=0)
    source_type = models.CharField(max_length=40)
    source_id = models.CharField(max_length=80)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=["source_type", "source_id"], name="usage_event_source_once"),
        ]
        indexes = [models.Index(fields=["kind", "instance_type", "-created_at"]),
                   models.Index(fields=["user", "-created_at"])]

    def __str__(self):
        return f"{self.user_id} {self.kind} {self.credits} cr"


class Attempt(models.Model):
    """One rate-limited action (a password-reset email, a promo-code try), by
    key: ``throttle.allow``. Rows older than a day are deleted as new ones come."""

    key = models.CharField(max_length=200, db_index=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    def __str__(self):
        return f"{self.key} {self.created_at:%Y-%m-%d %H:%M}"


class CreditHold(models.Model):
    """Credits ``quota.check`` just approved for a launch that is still creating
    its jobs. Counted as spent until the jobs exist (and count themselves), so
    two launches sent at the same moment can't both spend the same balance.
    Expires on its own; nothing deletes it on success."""

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="credit_holds")
    credits = models.DecimalField(max_digits=12, decimal_places=4)
    expires_at = models.DateTimeField(db_index=True)

    def __str__(self):
        return f"{self.user_id} holds {self.credits} until {self.expires_at:%H:%M:%S}"


class EmailLog(models.Model):
    """Every email the platform tried to send, and whether SES took it.

    Mail fails quietly: the password-reset page says "check your email" either
    way, on purpose. This table is how anyone finds out it didn't go, and it is
    what the footage-deletion notices must check before deleting anything.
    """

    class Status(models.TextChoices):
        SENT = "sent", "Sent"
        FAILED = "failed", "Failed"

    to = models.EmailField()
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                             on_delete=models.SET_NULL, related_name="email_logs")
    kind = models.CharField(max_length=40)
    subject = models.CharField(max_length=255)
    status = models.CharField(max_length=10, choices=Status.choices)
    message_id = models.CharField(max_length=200, blank=True, default="")
    error = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.kind} to {self.to}: {self.status}"


class LoginEvent(models.Model):
    """One sign-in attempt, successful or not. IP and browser are kept 90 days
    (``prune``); the platform dashboard shows them per user and as failed-login
    bursts."""

    class Method(models.TextChoices):
        WEB = "web", "Web"
        ADMIN = "admin", "Django admin"
        API = "api", "API login"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                             on_delete=models.CASCADE, related_name="login_events")
    # What was typed, for failures that match no account.
    username = models.CharField(max_length=254, blank=True, default="")
    success = models.BooleanField()
    method = models.CharField(max_length=10, choices=Method.choices, default=Method.WEB)
    ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=300, blank=True, default="")
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    KEEP_DAYS = 90

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["user", "-created_at"])]

    def __str__(self):
        return f"{self.username or self.user_id} {'ok' if self.success else 'failed'} {self.created_at:%Y-%m-%d %H:%M}"

    @classmethod
    def prune(cls):
        cls.objects.filter(created_at__lt=timezone.now() - timezone.timedelta(days=cls.KEEP_DAYS)).delete()


class AdminAction(models.Model):
    """The audit log: every change a platform admin makes to someone's account
    (tier, credits, limits, suspension, password reset, promo codes), and the
    changes the platform makes on its own (a tier running out)."""

    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                              on_delete=models.SET_NULL, related_name="admin_actions")
    target = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True,
                               on_delete=models.SET_NULL, related_name="admin_actions_received")
    action = models.CharField(max_length=40)
    # One sentence a person reads in the log, written when it happened.
    summary = models.CharField(max_length=300)
    reason = models.CharField(max_length=300, blank=True, default="")
    detail = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(default=timezone.now, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.summary

    @classmethod
    def record(cls, actor, action, summary, target=None, reason="", **detail):
        return cls.objects.create(actor=actor if getattr(actor, "pk", None) else None,
                                  target=target, action=action, summary=summary,
                                  reason=reason, detail=detail)
