"""Reset monthly credits for all users whose reset date has passed.

Run daily via cron or Azure scheduled task:
    python manage.py reset_monthly_credits
"""

from django.core.management.base import BaseCommand
from django.utils import timezone

from apps.accounts.models import UserProfile


class Command(BaseCommand):
    help = "Reset monthly credits for users whose reset date has passed"

    def handle(self, *args, **options):
        today = timezone.now().date()
        profiles = UserProfile.objects.filter(
            credit_reset_date__lte=today,
        ) | UserProfile.objects.filter(credit_reset_date__isnull=True)

        count = 0
        for profile in profiles:
            old_used = profile.used_credit_cents
            profile.reset_monthly_credits()
            count += 1
            self.stdout.write(
                f"  Reset {profile.user.username}: "
                f"{old_used} credits used → 0 "
                f"(limit: {profile.monthly_credits})"
            )

        self.stdout.write(self.style.SUCCESS(f"Reset {count} profile(s)."))
