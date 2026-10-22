"""Once-a-day tasks, run from the reconcile loop (apps/analysis/reconcile.py).

That loop ticks every couple of minutes in every web worker. ``run_daily``
claims each task for today with one conditional UPDATE on ``DailyRun``, so
exactly one worker runs it, once a day, and a deploy mid-day doesn't repeat it.
"""

import logging
from datetime import timedelta

from django.utils import timezone

logger = logging.getLogger(__name__)

PRO_REMINDER_DAYS = 30


def _claim(name, today) -> bool:
    from django.db.models import Q

    from .models import DailyRun

    DailyRun.objects.get_or_create(name=name)
    return bool(DailyRun.objects.filter(name=name).filter(Q(ran_on__isnull=True) | Q(ran_on__lt=today))
                .update(ran_on=today))


def _record(name, result):
    from .models import DailyRun

    DailyRun.objects.filter(name=name).update(result=result)


def pro_reminders(today=None) -> int:
    """Email Pro accounts whose plan ends within 30 days, once per end date."""
    from .emails import send
    from .models import Attempt, UserProfile

    today = today or timezone.localdate()
    soon = UserProfile.objects.filter(
        tier=UserProfile.Tier.PRO, tier_expires_at__date__gt=today,
        tier_expires_at__date__lte=today + timedelta(days=PRO_REMINDER_DAYS),
    ).select_related("user")
    sent = 0
    for profile in soon:
        ends = timezone.localdate(profile.tier_expires_at)
        key = f"pro-reminder:{profile.user_id}:{ends.isoformat()}"
        if Attempt.objects.filter(key=key).exists() or not profile.user.email:
            continue
        send("pro_reminder", profile.user.email, f"Your BeeMonitor Pro plan ends on {ends:%b} {ends.day}",
             f"Hello {profile.user.username},\n\nYour Pro plan ends on {ends:%B} {ends.day}, {ends.year}. "
             "After that the account is Standard: credits you were granted stay, and raw footage "
             "older than 3 months gets 30 days' notice before it is deleted.\n\n"
             "To keep Pro, request it again from Usage & credits.\n", user=profile.user)
        Attempt.objects.create(key=key)
        sent += 1
    return sent


def _raw_footage(today):
    from apps.videos import retention
    return retention.run(today)


TASKS = {
    "raw_footage": _raw_footage,
    "pro_reminders": lambda today: {"sent": pro_reminders(today)},
}


def run_daily(today=None) -> dict:
    today = today or timezone.localdate()
    done = {}
    for name, task in TASKS.items():
        if not _claim(name, today):
            continue
        try:
            result = task(today)
        except Exception:
            logger.exception("daily task %s failed", name)
            result = {"error": True}
        _record(name, {**result, "on": today.isoformat()})
        done[name] = result
    return done
