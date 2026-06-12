"""Activity-frame ingest endpoint.

Field units sample a few crops of the mover per activity (see
``hardware/main_motion.py``) and ship them over cellular under a daily cap (see
``hardware/telemetry.py``). Each frame is one multipart POST here: we store the
JPEG in S3, upsert the Activity it belongs to (keyed by the device-generated
``activity_uid``), and record an ActivityFrame. The taxonomic analysis (BioCLIP)
runs later — this endpoint just gets the pixels in. Design:
``memory/15_monitoring_agent_design.md``.

  POST /api/v1/devices/frames   (multipart/form-data, device-authenticated)
      frame : file              — one JPEG crop of the mover
      meta  : JSON string       — { activity_uid, started_at, captured_at,
                                     bbox, motion_score, peak_motion, kind,
                                     lat, lon, width, height }
  -> { activity_id, frame_id, capped }
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone as dt_timezone

from django.conf import settings
from django.utils import timezone
from rest_framework.parsers import FormParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.devices.models import Device
from apps.monitor.models import Activity, ActivityFrame
from config.storage import get_s3_client

from .authentication import DeviceKeyAuthentication

logger = logging.getLogger(__name__)

# A mover crop is tiny; reject anything that clearly isn't one.
MAX_FRAME_BYTES = 5 * 1024 * 1024  # 5 MiB


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _epoch_to_dt(value):
    """Coerce an epoch-seconds value (device-reported, UTC) to an aware datetime."""
    v = _as_float(value)
    if v is None:
        return None
    try:
        return datetime.fromtimestamp(v, tz=dt_timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _frame_key(owner_id: int, device_id: int) -> str:
    """``users/<owner>/devices/<device>/activity_frames/<yyyy>/<mm>/<dd>/<uuid>.jpg``."""
    now = timezone.now().astimezone(dt_timezone.utc)
    return (
        f"users/{owner_id}/devices/{device_id}/activity_frames/"
        f"{now.year:04d}/{now.month:02d}/{now.day:02d}/{uuid.uuid4().hex}.jpg"
    )


class DeviceFrameView(APIView):
    """Receive one sampled activity frame from a field device."""

    authentication_classes = [DeviceKeyAuthentication]
    parser_classes = [MultiPartParser, FormParser]
    throttle_classes: list = []

    def post(self, request):
        device: Device = request.auth  # set by DeviceKeyAuthentication
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)

        raw_meta = request.data.get("meta", "")
        if isinstance(raw_meta, str) and raw_meta:
            try:
                meta = json.loads(raw_meta)
            except ValueError:
                return Response({"detail": "meta must be valid JSON."}, status=400)
        elif isinstance(raw_meta, dict):
            meta = raw_meta
        else:
            meta = {}
        if not isinstance(meta, dict):
            return Response({"detail": "meta must be a JSON object."}, status=400)

        activity_uid = str(meta.get("activity_uid") or "").strip()[:64]
        if not activity_uid:
            return Response({"detail": "meta.activity_uid is required."}, status=400)

        image = request.FILES.get("frame")
        if image is None:
            return Response({"detail": "a 'frame' file is required."}, status=400)
        if image.size > MAX_FRAME_BYTES:
            return Response({"detail": f"frame exceeds {MAX_FRAME_BYTES} bytes."}, status=400)

        # Upsert the activity this frame belongs to. started_at falls back to now
        # so a frame is never dropped just because the device omitted a timestamp.
        started_at = _epoch_to_dt(meta.get("started_at")) or timezone.now()
        lat = _as_float(meta.get("lat"))
        lon = _as_float(meta.get("lon"))
        if lat is None and lon is None:  # fall back to the device's set coordinates
            lat, lon = device.lat, device.lon
        peak_motion = _as_float(meta.get("peak_motion"))

        activity, _created = Activity.objects.get_or_create(
            device=device, activity_uid=activity_uid,
            defaults={"started_at": started_at, "lat": lat, "lon": lon,
                      "peak_motion": peak_motion},
        )

        # Server-side guard so a misbehaving device can't pile unbounded frames
        # onto one activity. The device also self-limits with its daily cap.
        if activity.frames.count() >= settings.MONITOR_MAX_FRAMES_PER_ACTIVITY:
            logger.info("frames: activity %s at cap, ignoring extra frame from device %s",
                        activity.id, device.id)
            return Response({"activity_id": activity.id, "frame_id": None, "capped": True},
                            status=200)

        # Keep the strongest motion score seen for the activity.
        if peak_motion is not None and (activity.peak_motion is None
                                        or peak_motion > activity.peak_motion):
            activity.peak_motion = peak_motion
            activity.save(update_fields=["peak_motion"])

        key = _frame_key(device.owner_id, device.id)
        try:
            get_s3_client().upload_stream("raw-videos", key, image, content_type="image/jpeg")
        except Exception as e:
            logger.exception("frame upload failed for device %s", device.id)
            return Response({"detail": f"storage error: {e}"}, status=502)

        kind = meta.get("kind") if meta.get("kind") in dict(ActivityFrame.Kind.choices) else "crop"
        frame = ActivityFrame.objects.create(
            activity=activity,
            kind=kind,
            storage_key=key,
            bbox=meta.get("bbox") if isinstance(meta.get("bbox"), list) else None,
            motion_score=_as_float(meta.get("motion_score")),
            width=_as_int(meta.get("width")),
            height=_as_int(meta.get("height")),
            captured_at=_epoch_to_dt(meta.get("captured_at")),
        )

        logger.info("frame: device=%s activity=%s frame=%s uid=%s",
                    device.id, activity.id, frame.id, activity_uid)
        return Response({"activity_id": activity.id, "frame_id": frame.id, "capped": False},
                        status=201)


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
