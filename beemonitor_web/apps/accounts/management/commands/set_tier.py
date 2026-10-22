"""Set a user's account tier.

Usage:
    python manage.py set_tier <username> <tier>

Example:
    python manage.py set_tier edward pro
"""

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand, CommandError

from apps.accounts.models import TIER_LIMITS, UserProfile


class Command(BaseCommand):
    help = "Set a user's account tier (standard, pro)"

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
        profile.used_credits = 0
        profile.save(update_fields=["tier", "monthly_credits", "used_credits"])

        self.stdout.write(self.style.SUCCESS(
            f"Updated {username}: {old_tier} -> {tier} "
            f"(GPU credits: {'unlimited' if limits['unlimited_gpu'] else 'balance unchanged'})"))
