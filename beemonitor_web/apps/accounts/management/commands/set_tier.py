"""Set a user's account tier.

Usage:
    python manage.py set_tier <username> <tier>

Example:
    python manage.py set_tier edward enterprise
"""

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError

from apps.accounts.models import TIER_LIMITS, UserProfile


class Command(BaseCommand):
    help = "Set a user's account tier (free, standard, pro, enterprise)"

    def add_arguments(self, parser):
        parser.add_argument("username", type=str)
        parser.add_argument("tier", type=str, choices=list(TIER_LIMITS.keys()))

    def handle(self, *args, **options):
        username = options["username"]
        tier = options["tier"]

        try:
            user = User.objects.get(username=username)
        except User.DoesNotExist:
            raise CommandError(f"User '{username}' not found")

        profile, _ = UserProfile.objects.get_or_create(user=user)
        old_tier = profile.tier
        limits = TIER_LIMITS[tier]
        profile.tier = tier
        profile.monthly_credits = limits["monthly_credits"]
        profile.max_concurrent_jobs = limits["max_concurrent_jobs"]
        profile.used_credits = 0
        profile.save(update_fields=["tier", "monthly_credits", "max_concurrent_jobs", "used_credits"])

        self.stdout.write(
            self.style.SUCCESS(
                f"Updated {username}: {old_tier} -> {tier} "
                f"(credits: {profile.monthly_credits}, "
                f"concurrent: {profile.max_concurrent_jobs})"
            )
        )
