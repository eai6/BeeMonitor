"""Find and resolve duplicate-email user accounts.

    python manage.py dedupe_accounts                 # dry-run report
    python manage.py dedupe_accounts --apply         # delete EMPTY duplicates
    python manage.py dedupe_accounts --apply --merge # consolidate (reassign + delete)
    python manage.py dedupe_accounts --email a@b.com  # limit to one email

Dry-run by default. ``--merge`` moves the duplicate's devices/videos/shares/
enrollment tokens to the keeper before deleting it; without it, only empty
duplicates are removed and data-bearing ones are reported for review.
"""

from django.core.management.base import BaseCommand

from apps.accounts.dedupe import dedupe_all


class Command(BaseCommand):
    help = "Resolve duplicate-email user accounts (keep the most active)."

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", help="Actually delete/merge (default: dry-run).")
        parser.add_argument("--merge", action="store_true", help="Reassign a duplicate's data to the keeper, then delete it.")
        parser.add_argument("--email", default=None, help="Limit to a single email address.")

    def handle(self, *args, **opts):
        report = dedupe_all(apply=opts["apply"], merge=opts["merge"], email=opts["email"])
        if not report:
            self.stdout.write(self.style.SUCCESS("No duplicate-email accounts found."))
            return
        mode = "APPLIED" if opts["apply"] else "DRY-RUN (nothing changed)"
        for grp in report:
            k = grp["keeper"]
            self.stdout.write(self.style.MIGRATE_HEADING(f"\n{grp['email']}"))
            self.stdout.write(f"  keep: {k.username} (id={k.id}, last_login={k.last_login})")
            for action, u, note in grp["actions"]:
                style = self.style.WARNING if action == "skip" else self.style.NOTICE
                self.stdout.write(style(f"  {action}: {u.username} (id={u.id}) — {note}"))
        self.stdout.write(self.style.SUCCESS(f"\n{mode}. {len(report)} duplicate group(s)."))
