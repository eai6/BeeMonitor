"""Find and resolve duplicate-email user accounts.

Django's ``User.email`` isn't unique, so accidental duplicates can exist (they're
what made password reset 500). This consolidates them: pick the most-active
account as the keeper and either delete empty duplicates or merge data-bearing
ones into the keeper.

Used by the ``dedupe_accounts`` management command and the User admin action.
"""

from __future__ import annotations

from datetime import datetime, timezone as dt_timezone

from django.contrib.auth import get_user_model
from django.db import transaction
from django.db.models import Count
from django.db.models.functions import Lower

_EPOCH = datetime.min.replace(tzinfo=dt_timezone.utc)


def _counts(user) -> dict:
    """Per-user counts of the data we care about preserving."""
    from apps.devices.models import Device, DeviceShare, EnrollmentToken
    from apps.videos.models import Video
    return {
        "devices": Device.objects.filter(owner=user).count(),
        "videos": Video.objects.filter(user=user).count(),
        "shares": DeviceShare.objects.filter(user=user).count(),
        "enroll_tokens": EnrollmentToken.objects.filter(user=user).count(),
    }


def _data_total(user) -> int:
    return sum(_counts(user).values())


def find_duplicate_groups(email: str | None = None):
    """Return [(lower_email, [users...]), ...] for emails with >1 account."""
    User = get_user_model()
    base = User.objects.exclude(email="").annotate(le=Lower("email"))
    if email:
        les = [email.strip().lower()]
    else:
        les = [r["le"] for r in base.values("le").annotate(n=Count("id")).filter(n__gt=1)]
    groups = []
    for le in les:
        users = list(base.filter(le=le).order_by("id"))
        if len(users) > 1:
            groups.append((le, users))
    return groups


def _pick_keeper(users):
    """Keep the account with the most data, then most recent login, then oldest."""
    return max(users, key=lambda u: (_data_total(u), (u.last_login or _EPOCH), -u.id))


def _reassign(dup, keeper):
    """Move a duplicate's BeeMonitor data to the keeper before deleting it."""
    from apps.devices.models import Device, DeviceShare, EnrollmentToken
    from apps.videos.models import Video
    Device.objects.filter(owner=dup).update(owner=keeper)
    Video.objects.filter(user=dup).update(user=keeper)
    EnrollmentToken.objects.filter(user=dup).update(user=keeper)
    DeviceShare.objects.filter(created_by=dup).update(created_by=keeper)
    # Shares have a unique (device, user) — fold rather than collide.
    for s in DeviceShare.objects.filter(user=dup):
        if DeviceShare.objects.filter(device=s.device, user=keeper).exists():
            s.delete()
        else:
            s.user = keeper
            s.save(update_fields=["user"])


def dedupe_group(users, apply=False, merge=False):
    """Resolve one duplicate group. Returns (keeper, [(action, user, note), ...])."""
    keeper = _pick_keeper(users)
    actions = []
    for u in users:
        if u.id == keeper.id:
            continue
        n = _data_total(u)
        if n > 0 and not merge:
            actions.append(("skip", u, f"has data {_counts(u)} — use merge to consolidate"))
            continue
        if apply:
            with transaction.atomic():
                if n > 0:
                    _reassign(u, keeper)
                u.delete()
            actions.append(("deleted", u, "merged into keeper" if n else "empty duplicate"))
        else:
            actions.append(("would-delete", u, f"data={n}" + ("" if n == 0 else " (needs merge)")))
    return keeper, actions


def dedupe_all(apply=False, merge=False, email=None):
    """Resolve all (or one email's) duplicate groups. Returns a report list."""
    report = []
    for le, users in find_duplicate_groups(email=email):
        keeper, actions = dedupe_group(users, apply=apply, merge=merge)
        report.append({"email": le, "keeper": keeper, "actions": actions})
    return report
