"""Generate per-device daily digests (run on a schedule, e.g. nightly cron).

Defaults to *yesterday* (UTC). Only devices that were relevant that day — had a
heartbeat or any activity in the window — get a digest, so long-dormant units
don't accrue daily noise.

  python manage.py generate_digests                 # yesterday, all devices
  python manage.py generate_digests --date 2026-06-10 --device 3
"""

from datetime import date as date_cls, datetime, time, timedelta, timezone as dt_timezone

from django.core.management.base import BaseCommand, CommandError

from apps.devices.models import Device
from apps.monitor.agent.digest import generate_digest
from apps.monitor.models import Activity


class Command(BaseCommand):
    help = "Generate per-device daily digests for a UTC day (default: yesterday)."

    def add_arguments(self, parser):
        parser.add_argument("--date", help="UTC day YYYY-MM-DD (default: yesterday).")
        parser.add_argument("--device", type=int, help="Limit to one device id.")

    def handle(self, *args, **opts):
        if opts.get("date"):
            try:
                day = datetime.strptime(opts["date"], "%Y-%m-%d").date()
            except ValueError:
                raise CommandError("--date must be YYYY-MM-DD.")
        else:
            day = (datetime.now(dt_timezone.utc) - timedelta(days=1)).date()

        devices = Device.objects.all()
        if opts.get("device"):
            devices = devices.filter(pk=opts["device"])

        start = datetime.combine(day, time.min, tzinfo=dt_timezone.utc)
        end = start + timedelta(days=1)
        made = skipped = 0
        for device in devices:
            relevant = (
                Activity.objects.filter(device=device, started_at__gte=start,
                                        started_at__lt=end).exists()
                or device.heartbeats.filter(created_at__gte=start,
                                            created_at__lt=end).exists())
            if not relevant:
                skipped += 1
                continue
            digest = generate_digest(device, day)
            made += 1
            self.stdout.write(f"  {device.name} {day}: {digest.summary[:90]}")
        self.stdout.write(self.style.SUCCESS(
            f"Done: {made} digest(s) for {day}, {skipped} device(s) skipped (no activity)."))
