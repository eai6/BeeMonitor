"""Device health history: per-minute maintenance metrics, kept indefinitely.

Every heartbeat folds into its minute's DeviceHealthSample (record_beat). Raw
DeviceHeartbeat rows carry the full JSON and would grow by tens of MB per device
per day at the 5s WiFi beat, so prune_heartbeats (run from the background
reconciler) deletes those older than RAW_HEARTBEAT_DAYS — folding each batch
into the history first, so nothing is lost even for beats from before this
existed. Beats that carry a camera image are kept (the camera card uses them).
"""

from __future__ import annotations

import logging
from datetime import timedelta

from django.utils import timezone

from .models import DeviceHealthSample, DeviceHeartbeat

logger = logging.getLogger(__name__)

RAW_HEARTBEAT_DAYS = 7
PRUNE_BATCH = 2000

_FIELDS = ["storage_pct", "cpu_temp_c", "uptime_seconds", "pending_uploads",
           "clips_this_hour", "services_healthy", "services_total", "transport",
           "battery_voltage"]

# range key -> (span, bucket) for the chart: ~300 points whatever the range.
HEALTH_RANGES = {
    "24h": (timedelta(hours=24), timedelta(minutes=5)),
    "7d": (timedelta(days=7), timedelta(minutes=30)),
    "30d": (timedelta(days=30), timedelta(hours=2)),
    "90d": (timedelta(days=90), timedelta(hours=6)),
}


def _num(v, cast=float):
    if isinstance(v, bool) or v is None:
        return None
    try:
        return cast(v)
    except (TypeError, ValueError):
        return None


def sample_from_metrics(device_id: int, at, metrics: dict) -> DeviceHealthSample:
    """The minute row one beat's metrics produce (unsaved)."""
    from .views import _service_rows, _services_summary

    summary = _services_summary(_service_rows(metrics or {}))
    return DeviceHealthSample(
        device_id=device_id,
        minute=at.replace(second=0, microsecond=0),
        storage_pct=_num(metrics.get("storage_pct")),
        cpu_temp_c=_num(metrics.get("cpu_temp_c")),
        uptime_seconds=_num(metrics.get("uptime_seconds"), int),
        pending_uploads=_num(metrics.get("pending_uploads"), int),
        clips_this_hour=_num(metrics.get("clips_this_hour"), int),
        services_healthy=summary["healthy"] if summary["total"] else None,
        services_total=summary["total"] or None,
        transport=str(metrics.get("active_transport") or "")[:16],
        battery_voltage=_num(metrics.get("battery_voltage")),
    )


def _upsert(samples: list) -> None:
    # One row per (device, minute); within a batch keep the latest beat.
    latest = {}
    for s in samples:
        latest[(s.device_id, s.minute)] = s
    if latest:
        DeviceHealthSample.objects.bulk_create(
            list(latest.values()), update_conflicts=True,
            unique_fields=["device", "minute"], update_fields=_FIELDS)


def record_beat(hb: DeviceHeartbeat) -> None:
    """Fold a just-received beat into its minute. Never raises."""
    try:
        _upsert([sample_from_metrics(hb.device_id, hb.created_at, hb.metrics or {})])
    except Exception:
        logger.exception("health sample for heartbeat %s failed", hb.pk)


def fold_heartbeats(qs) -> int:
    """Write the history rows for a batch of heartbeats. Returns rows folded."""
    rows = list(qs.values_list("device_id", "created_at", "metrics"))
    # Oldest first so, within a minute, the newest beat wins the upsert.
    rows.sort(key=lambda r: r[1])
    _upsert([sample_from_metrics(d, at, m or {}) for d, at, m in rows])
    return len(rows)


def prune_heartbeats(now=None, max_batches: int = 5) -> int:
    """Delete raw beats older than RAW_HEARTBEAT_DAYS (image beats kept), folding
    each batch into the history first. Bounded per call; idempotent."""
    from .models import Device

    cutoff = (now or timezone.now()) - timedelta(days=RAW_HEARTBEAT_DAYS)
    deleted = 0
    for device_id in Device.objects.values_list("pk", flat=True):
        for _ in range(max_batches):
            ids = list(DeviceHeartbeat.objects
                       .filter(device_id=device_id, created_at__lt=cutoff, image_storage_key="")
                       .order_by("created_at").values_list("pk", flat=True)[:PRUNE_BATCH])
            if not ids:
                break
            fold_heartbeats(DeviceHeartbeat.objects.filter(pk__in=ids))
            deleted += DeviceHeartbeat.objects.filter(pk__in=ids).delete()[0]
    return deleted


def health_series(device, range_key: str, zone, now=None) -> dict:
    """Bucketed history for the chart, labelled in ``zone`` (the device's display
    tz). Empty buckets are None (the device was off), and reboots are the
    buckets where uptime went backwards."""
    if range_key not in HEALTH_RANGES:
        range_key = "24h"
    span, step = HEALTH_RANGES[range_key]
    end = (now or timezone.now()).replace(second=0, microsecond=0)
    start = end - span
    rows = (DeviceHealthSample.objects
            .filter(device=device, minute__gte=start, minute__lte=end)
            .order_by("minute")
            .values_list("minute", "storage_pct", "cpu_temp_c", "pending_uploads",
                         "services_healthy", "services_total", "uptime_seconds"))
    n = int(span / step)
    acc = [dict(storage=[], temp=[], pending=[], healthy=[], total=[]) for _ in range(n)]
    reboots, last_uptime = set(), None
    for minute, storage, temp, pending, healthy, total, uptime in rows:
        i = min(n - 1, int((minute - start) / step))
        b = acc[i]
        for key, v in (("storage", storage), ("temp", temp), ("pending", pending),
                       ("healthy", healthy), ("total", total)):
            if v is not None:
                b[key].append(v)
        if uptime is not None:
            if last_uptime is not None and uptime < last_uptime:
                reboots.add(i)
            last_uptime = uptime

    def avg(xs):
        return round(sum(xs) / len(xs), 1) if xs else None

    fmt = "%H:%M" if range_key == "24h" else "%b %d %H:%M"
    points = []
    for i, b in enumerate(acc):
        t = start + step * i
        points.append({
            "iso": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "t": t.astimezone(zone).strftime(fmt),
            "storage_pct": avg(b["storage"]),
            "cpu_temp_c": avg(b["temp"]),
            # Worst case in the bucket: most pending, fewest healthy services.
            "pending_uploads": max(b["pending"]) if b["pending"] else None,
            "services_healthy": min(b["healthy"]) if b["healthy"] else None,
            "services_total": max(b["total"]) if b["total"] else None,
            "reboot": i in reboots,
        })
    return {"range": range_key, "step_minutes": int(step.total_seconds() // 60),
            "points": points}
