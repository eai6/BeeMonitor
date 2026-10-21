"""Make the usernames in BEEMONITOR_SUPERUSERS superusers (platform managers).

Usage:
    BEEMONITOR_SUPERUSERS=eai6,other python manage.py ensure_superusers

Runs on every container start (scripts/entrypoint.sh); the list comes from the
Pulumi config ``superusers`` (infra/aws). It only ever promotes: removing a name
from the list does not demote anyone — do that in Django admin. Missing
usernames are reported and skipped.
"""

import os

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Promote the usernames in BEEMONITOR_SUPERUSERS to superuser"

    def handle(self, *args, **options):
        names = [n.strip() for n in os.environ.get("BEEMONITOR_SUPERUSERS", "").split(",") if n.strip()]
        for name in names:
            user = User.objects.filter(username=name).first()
            if user is None:
                self.stdout.write(f"ensure_superusers: no user '{name}', skipped")
            elif not (user.is_superuser and user.is_staff):
                user.is_superuser = user.is_staff = True
                user.save(update_fields=["is_superuser", "is_staff"])
                self.stdout.write(f"ensure_superusers: {name} is now a superuser")
