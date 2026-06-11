"""Device heartbeat endpoint.

Field units POST a small health beat hourly over cellular (see
``hardware/telemetry.py`` and design doc ``memory/10_cellular_telemetry_design.md``).
This is the *cheap* channel — telemetry JSON + one ~250 KB image — kept separate
from the WiFi-gated bulk-video upload (``uploads.py``).

  POST /api/v1/devices/heartbeat   (multipart/form-data, device-authenticated)
      metrics : JSON string  — free-form health payload (storage, uptime, temp,
                               service health, cellular signal, schedule, …)
      image   : file (opt.)  — one JPEG still; stored in S3 raw-videos

The endpoint stores the image in S3, records a ``DeviceHeartbeat`` row, and
returns its id. ``DeviceKeyAuthentication`` already stamps
``Device.last_seen_at`` on every authenticated call, which is what drives the
online/offline indicator in the UI.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timezone as dt_timezone

from django.conf import settings
from django.utils import timezone
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.devices.models import Device, DeviceHeartbeat
from config.storage import get_s3_client

from .authentication import DeviceKeyAuthentication

logger = logging.getLogger(__name__)

# Reject absurdly large "images" — a heartbeat still is a few hundred KB.
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MiB


def _as_float(value):
    """Coerce a metric to float, or None if missing/unparseable."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _heartbeat_image_key(owner_id: int, device_id: int) -> str:
    """``users/<owner>/devices/<device>/heartbeats/<yyyy>/<mm>/<dd>/<uuid>.jpg``."""
    now = timezone.now().astimezone(dt_timezone.utc)
    return (
        f"users/{owner_id}/devices/{device_id}/heartbeats/"
        f"{now.year:04d}/{now.month:02d}/{now.day:02d}/{uuid.uuid4().hex}.jpg"
    )


class DeviceHeartbeatView(APIView):
    """Receive a telemetry beat (+ optional image) from a field device."""

    authentication_classes = [DeviceKeyAuthentication]
    parser_classes = [MultiPartParser, FormParser]
    # Bytes are tiny and uploads must never be throttled away — a dropped beat
    # reads as "device offline".
    throttle_classes: list = []

    def post(self, request):
        device: Device = request.auth  # set by DeviceKeyAuthentication
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)

        # `metrics` arrives as a JSON string in a multipart field.
        raw_metrics = request.data.get("metrics", "")
        if isinstance(raw_metrics, str) and raw_metrics:
            try:
                metrics = json.loads(raw_metrics)
            except ValueError:
                return Response({"detail": "metrics must be valid JSON."}, status=400)
        elif isinstance(raw_metrics, dict):
            metrics = raw_metrics
        else:
            metrics = {}
        if not isinstance(metrics, dict):
            return Response({"detail": "metrics must be a JSON object."}, status=400)

        # Optional image.
        image = request.FILES.get("image")
        image_key = ""
        if image is not None:
            if image.size > MAX_IMAGE_BYTES:
                return Response(
                    {"detail": f"image exceeds {MAX_IMAGE_BYTES} bytes."}, status=400,
                )
            image_key = _heartbeat_image_key(device.owner_id, device.id)
            try:
                get_s3_client().upload_stream(
                    "raw-videos", image_key, image, content_type="image/jpeg",
                )
            except Exception as e:
                logger.exception("heartbeat image upload failed for device %s", device.id)
                # Don't fail the whole beat over the image — telemetry still lands.
                image_key = ""
                metrics.setdefault("_warnings", []).append(f"image_upload_failed: {e}")

        storage_pct = _as_float(metrics.get("storage_pct"))
        lat = _as_float(metrics.get("gps_lat"))
        lon = _as_float(metrics.get("gps_lon"))

        snips = metrics.get("snippets_last_period")
        try:
            snips = int(snips) if snips is not None else None
        except (TypeError, ValueError):
            snips = None

        hb = DeviceHeartbeat.objects.create(
            device=device,
            metrics=metrics,
            image_storage_key=image_key,
            storage_pct=storage_pct,
            snippets_last_period=snips,
            lat=lat if settings.DEVICE_STORE_GPS_PER_HEARTBEAT else None,
            lon=lon if settings.DEVICE_STORE_GPS_PER_HEARTBEAT else None,
        )

        # Device's local timezone — used to bucket activity by the hive's local
        # clock-hour. Persisted when it changes.
        tz_updates = []
        tz_name = (metrics.get("tz") or "")[:64]
        if tz_name and tz_name != device.tz_name:
            device.tz_name = tz_name
            tz_updates.append("tz_name")
        tz_off = metrics.get("tz_offset_min")
        if isinstance(tz_off, (int, float)) and int(tz_off) != device.tz_offset_min:
            device.tz_offset_min = int(tz_off)
            tz_updates.append("tz_offset_min")
        if tz_updates:
            device.save(update_fields=tz_updates)

        # 5c: device advertises a live LAN MJPEG stream while one is active.
        stream_url = (metrics.get("stream_url") or "").strip()
        stream_until = _as_float(metrics.get("stream_until"))
        if stream_url and stream_until:
            from datetime import datetime, timezone as _tz
            device.stream_url = stream_url[:200]
            device.stream_expires_at = datetime.fromtimestamp(stream_until, tz=_tz.utc)
            device.save(update_fields=["stream_url", "stream_expires_at"])

        # Auto-capture one picture the first time a device is online with no
        # image yet, so the camera card isn't blank. Fires once per device.
        if (not device.first_image_requested and not device.pending_command
                and not device.heartbeats.exclude(image_storage_key="").exists()):
            device.pending_command = "capture_image"
            device.command_params = {}
            device.first_image_requested = True
            device.save(update_fields=["pending_command", "command_params",
                                       "first_image_requested"])

        # Hand back any pending command (picture/stream on demand), then clear it.
        command = device.pending_command or ""
        params = device.command_params or {}
        if command:
            device.pending_command = ""
            device.command_params = {}
            device.save(update_fields=["pending_command", "command_params"])

        logger.info(
            "heartbeat: device=%s user=%s hb=%s image=%s gps=%s cmd=%s",
            device.id, device.owner_id, hb.id, bool(image_key),
            bool(lat and lon), command or "-",
        )
        return Response(
            {
                "heartbeat_id": hb.id,
                "image_stored": bool(image_key),
                "command": command,
                "params": params,
                # Device caches this to name the USB-transfer folder per hive
                # (hardware/usb-transfer.sh); see telemetry's _cache_location.
                "location": device.location or "",
                # The device adopts this as its beat cadence (dashboard-set).
                "telemetry_interval": device.telemetry_interval_seconds,
                # Manual motion-tuning overrides (empty = use auto-calibration).
                "motion_tuning": device.motion_tuning_dict(),
                # ROI editor outputs (normalized): hotel ROI + nest layout.
                "roi_override": device.roi_override or None,
                "nest_layout": device.nest_layout or [],
            },
            status=201,
        )


class DeviceCommandView(APIView):
    """Lightweight command poll — lets the device pick up an on-demand request
    (e.g. capture_image) within seconds, without waiting for the 60s beat.

    GET returns the pending command and clears it. Cheap (no metrics/image), so
    the device can poll this every ~10s between beats.
    """

    authentication_classes = [DeviceKeyAuthentication]
    throttle_classes: list = []

    def get(self, request):
        device: Device = request.auth
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)
        command = device.pending_command or ""
        params = device.command_params or {}
        if command:
            device.pending_command = ""
            device.command_params = {}
            device.save(update_fields=["pending_command", "command_params"])
        return Response({
            "command": command,
            "params": params,
            # Cached device-side to name the USB-transfer folder per hive.
            "location": device.location or "",
            # Carry the cadence here too so a rate change applies within seconds
            # even when the beat interval is long.
            "telemetry_interval": device.telemetry_interval_seconds,
        })
