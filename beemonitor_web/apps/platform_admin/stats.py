"""The numbers behind /platform/: per-user totals and platform-wide ones.

Every credit and GPU-hour comes from UsageEvent; storage, clips and units from
their own tables. Per-user columns are subqueries, not joins, so summing one
table never multiplies by the rows of another.
"""

from datetime import timedelta
from decimal import Decimal

from django.contrib.auth import get_user_model
from django.db.models import (Count, DecimalField, F, FloatField, IntegerField, OuterRef, Q,
                              Subquery, Sum, Value)
from django.db.models.functions import Coalesce
from django.utils import timezone

from apps.accounts import quota
from apps.accounts.models import LoginEvent, UsageEvent, UserProfile
from apps.analysis.models import Job
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()

ACTIVE_DAYS = 30


def month_start():
    return timezone.localdate().replace(day=1)


def _sub(model_qs, user_field, expr, output):
    """One number per user from another table, as a correlated subquery."""
    return Coalesce(Subquery(
        model_qs.filter(**{user_field: OuterRef("pk")}).order_by()
        .values(user_field).annotate(v=expr).values("v")[:1],
        output_field=output), Value(0), output_field=output)


def users_with_totals():
    since = month_start()
    dec = DecimalField(max_digits=14, decimal_places=4)
    return (User.objects.select_related("profile").annotate(
        month_credits=_sub(UsageEvent.objects.filter(created_at__date__gte=since), "user",
                           Sum("credits"), dec),
        month_cost=_sub(UsageEvent.objects.filter(created_at__date__gte=since), "user",
                        Sum("cost_usd"), dec),
        life_seconds=_sub(UsageEvent.objects.all(), "user", Sum("gpu_seconds"), FloatField()),
        # Bytes in our buckets; clips kept in a user's own bucket don't count.
        storage_bytes=_sub(Video.everything.exclude(storage_key__startswith="s3://"), "user",
                           Sum("file_size_bytes"), IntegerField()),
        clips=_sub(Video.objects.all(), "user", Count("id"), IntegerField()),
        units=_sub(Device.objects.all(), "owner", Count("id"), IntegerField()),
        last_seen=F("profile__last_seen_at"),
    ))


def tier_label(user) -> str:
    if user.is_superuser:
        return "Super admin"
    profile = getattr(user, "profile", None)
    if profile is None:
        return "Standard"
    label = profile.get_tier_display()
    if profile.tier_expires_at:
        label += f" · to {profile.tier_expires_at:%b %Y}"
    return label


def status(user):
    """(label, tone) for the users table: what needs a look first."""
    if not user.is_active:
        return "Suspended", "off"
    if not quota.is_unlimited(user) and quota.available(user) <= 0:
        return "Out of credits", "warn"
    seen = user.last_seen or user.last_login
    if seen is None or seen < timezone.now() - timedelta(days=ACTIVE_DAYS):
        return "Inactive", "off"
    return "Active", "ok"


def overview():
    now = timezone.now()
    since = month_start()
    week_ago, month_ago = now - timedelta(days=7), now - timedelta(days=ACTIVE_DAYS)
    seen = Q(profile__last_seen_at__gte=month_ago) | Q(last_login__gte=month_ago)
    seen7 = Q(profile__last_seen_at__gte=week_ago) | Q(last_login__gte=week_ago)
    month_events = UsageEvent.objects.filter(created_at__date__gte=since)
    agg = month_events.aggregate(cr=Sum("credits"), cost=Sum("cost_usd"), secs=Sum("gpu_seconds"))
    by_kind = [
        {"kind": UsageEvent.Kind(r["kind"]).label if r["kind"] in UsageEvent.Kind.values else r["kind"],
         "credits": r["cr"] or Decimal(0), "hours": (r["secs"] or 0) / 3600}
        for r in month_events.values("kind").annotate(cr=Sum("credits"), secs=Sum("gpu_seconds"))
        .order_by("-cr")
    ]
    jobs = Job.objects.filter(created_at__date__gte=since)
    finished = jobs.filter(status__in=[Job.Status.COMPLETED, Job.Status.FAILED]).count()
    failed = jobs.filter(status=Job.Status.FAILED).count()
    tiers = dict(UserProfile.objects.filter(user__is_superuser=False)
                 .values_list("tier").annotate(n=Count("id")))
    storage = (Video.everything.exclude(storage_key__startswith="s3://")
               .aggregate(b=Sum("file_size_bytes"))["b"] or 0)
    return {
        "users": User.objects.count(),
        "new_this_week": User.objects.filter(date_joined__gte=week_ago).count(),
        "active_30d": User.objects.filter(seen).count(),
        "active_7d": User.objects.filter(seen7).count(),
        "credits_month": agg["cr"] or Decimal(0),
        "cost_month": agg["cost"] or Decimal(0),
        "hours_month": (agg["secs"] or 0) / 3600,
        "by_kind": by_kind,
        "failure_pct": (failed / finished * 100) if finished else 0,
        "failed_jobs": failed,
        "storage_bytes": storage,
        "tiers": [{"label": UserProfile.Tier(t).label, "n": n} for t, n in sorted(tiers.items())],
        "superusers": User.objects.filter(is_superuser=True).count(),
        "out_of_credits": UserProfile.objects.filter(
            tier=UserProfile.Tier.STANDARD, granted_gpu_credits__lte=0, user__is_superuser=False).count(),
        "expiring_soon": UserProfile.objects.filter(
            tier_expires_at__gte=now, tier_expires_at__lte=now + timedelta(days=30)).count(),
        "failed_logins_7d": LoginEvent.objects.filter(success=False, created_at__gte=week_ago).count(),
        "month_start": since,
    }
