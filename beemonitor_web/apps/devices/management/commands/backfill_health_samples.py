"""Fold every stored heartbeat into the per-minute health history.

Idempotent (one row per device per minute, upserted), so it is safe to re-run;
the entrypoint runs it in the background after each deploy. The reconciler's
prune folds beats before deleting them anyway — this covers the ones younger
than the prune cutoff that predate the history table.
"""

from django.core.management.base import BaseCommand

from apps.devices.health import fold_heartbeats
from apps.devices.models import DeviceHeartbeat


class Command(BaseCommand):
    help = "Fold stored heartbeats into DeviceHealthSample (idempotent)."

    def add_arguments(self, parser):
        parser.add_argument("--batch", type=int, default=2000)

    def handle(self, *args, batch, **opts):
        last_pk, total = 0, 0
        while True:
            ids = list(DeviceHeartbeat.objects.filter(pk__gt=last_pk)
                       .order_by("pk").values_list("pk", flat=True)[:batch])
            if not ids:
                break
            total += fold_heartbeats(DeviceHeartbeat.objects.filter(pk__in=ids))
            last_pk = ids[-1]
        self.stdout.write(f"folded {total} heartbeats into health samples")
