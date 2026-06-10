"""Device storage-cleanup endpoint.

Lets a field device free SD-card space by deleting local copies of clips that
(1) uploaded successfully (a Video row exists) AND (2) a human explicitly cleared
for deletion on the dashboard (``Video.device_delete_requested``). The device
never decides this on its own.

  GET  /api/v1/devices/cleanup   -> {"video_ids": [...]}  (cleared, still on device)
  POST /api/v1/devices/cleanup   {"deleted": [...]}       -> stamps device_deleted_at

Both device-authenticated (Bearer ``bmk_device_*``). Tiny JSON, so it's cheap to
run over cellular from the telemetry service.
"""
import logging

from django.utils import timezone
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.devices.models import Device
from apps.videos.models import PendingDeviceDeletion, Video

from .authentication import DeviceKeyAuthentication

logger = logging.getLogger(__name__)

# Cap how many ids we hand out per poll so a huge backlog can't bloat one request.
MAX_BATCH = 200


class DeviceCleanupView(APIView):
    """Hand the device the human-cleared clips to delete, and record confirmations."""

    authentication_classes = [DeviceKeyAuthentication]
    throttle_classes: list = []

    def get(self, request):
        device = request.auth
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)
        ids = list(
            Video.objects.filter(
                device=device,
                device_delete_requested=True,
                device_deleted_at__isnull=True,
            )
            .order_by("uploaded_at")
            .values_list("id", flat=True)[:MAX_BATCH]
        )
        # Also include tombstones — clips whose cloud Video was deleted while an
        # on-device copy may still exist (the id still matches the Pi's sidecar).
        if len(ids) < MAX_BATCH:
            tomb = list(
                PendingDeviceDeletion.objects.filter(device=device)
                .order_by("created_at")
                .values_list("video_id", flat=True)[: MAX_BATCH - len(ids)]
            )
            ids = list(dict.fromkeys(ids + tomb))  # de-dup, preserve order
        return Response({"video_ids": ids})

    def post(self, request):
        device = request.auth
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)
        raw = request.data.get("deleted") if isinstance(request.data, dict) else None
        ids = [int(v) for v in raw if str(v).isdigit()] if isinstance(raw, list) else []
        if not ids:
            return Response({"confirmed": 0})
        # Only stamp rows that belong to THIS device and were actually cleared —
        # a device can't mark someone else's videos (or un-cleared ones) deleted.
        n = (
            Video.objects.filter(
                device=device,
                id__in=ids,
                device_delete_requested=True,
                device_deleted_at__isnull=True,
            ).update(device_deleted_at=timezone.now())
        )
        # Clear any tombstones the device just freed (cloud copy already gone).
        t = PendingDeviceDeletion.objects.filter(device=device, video_id__in=ids).delete()[0]
        logger.info("device %s confirmed %d local deletions (+%d tombstoned)", device.id, n, t)
        return Response({"confirmed": n + t})
