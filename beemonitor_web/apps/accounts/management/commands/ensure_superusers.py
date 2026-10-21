"""One-time bootstrap: promote a platform manager to superuser.

Usage:
    BEEMONITOR_SUPERUSERS="name:email[,name2:email2]" python manage.py ensure_superusers

Runs on container start (scripts/entrypoint.sh) and does nothing unless the
variable is set. It comes from the Pulumi *secret* ``superusers``, which is
meant to be set for one deploy and then removed (infra/aws/__main__.py). Both
the username and the account's email must match, so a username alone can't
claim it. It only promotes; demote in Django admin.
"""

import os

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand


def parse(raw):
    pairs = []
    for item in (raw or "").split(","):
        name, _, email = item.strip().partition(":")
        if name.strip() and email.strip():
            pairs.append((name.strip(), email.strip()))
    return pairs


class Command(BaseCommand):
    help = "Promote the username:email pairs in BEEMONITOR_SUPERUSERS to superuser"

    def handle(self, *args, **options):
        for name, email in parse(os.environ.get("BEEMONITOR_SUPERUSERS")):
            user = User.objects.filter(username=name, email__iexact=email).first()
            if user is None:
                self.stdout.write(f"ensure_superusers: no user '{name}' with that email, skipped")
            elif not (user.is_superuser and user.is_staff):
                user.is_superuser = user.is_staff = True
                user.save(update_fields=["is_superuser", "is_staff"])
                self.stdout.write(f"ensure_superusers: {name} is now a superuser")
