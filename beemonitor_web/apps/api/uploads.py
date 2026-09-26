"""Pi-side upload endpoints.

Two-call flow:

  1. ``POST /api/v1/uploads/initiate`` — device-authenticated. Pi declares
     a filename / size / content-type; Django picks a server-side S3 key
     under ``users/<user_id>/devices/<device_id>/<yyyy>/<mm>/<dd>/<uuid>.<ext>``
     and returns a short-lived presigned PUT URL. AWS credentials never
     leave the Django app.
  2. Pi ``PUT`` the bytes directly to S3 with that URL.
  3. ``POST /api/v1/uploads/complete`` — Pi tells Django it succeeded;
     Django creates a ``Video`` row and returns the id. Phase 4 will
     enqueue analysis here.

v1 implements single-PUT only (S3 hard cap: 5 GB per PUT). Multipart
upload + resumable chunks lands in a follow-up.

Full-resolution stills (memory/40) use the same two calls with
``kind: "still"`` (and ``"still_thumb"`` for its 1280 px preview): the key goes
under ``.../devices/<device_id>/stills/`` and ``complete`` creates a
``DeviceStill`` instead of a ``Video``. Same WiFi-only uploader, same retries.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone as dt_timezone
from pathlib import PurePosixPath

from django.utils import timezone
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.devices.models import Device
from apps.videos.models import Video
from config.storage import get_s3_client

from .authentication import DeviceKeyAuthentication

logger = logging.getLogger(__name__)

# S3 hard cap for single-part PUT.
MAX_SINGLE_PUT_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB

# How long the presigned URL is valid. Long enough to upload a multi-GB
# video over a slow cellular link, short enough that a leaked URL expires.
PRESIGNED_URL_TTL_SECONDS = 6 * 60 * 60  # 6 hours


def _safe_extension(filename: str) -> str:
    """Return a lowercased extension (with dot), or empty string."""
    suffix = PurePosixPath(filename).suffix.lower()
    # Whitelist: only the formats the recorder actually produces.
    if suffix in {".mp4", ".h264", ".mov", ".mkv"}:
        return suffix
    return ""


STILL_KINDS = ("still", "still_thumb")


def _build_storage_key(user_id: int, device_id: int, ext: str, recorded_at: datetime,
                       kind: str = "") -> str:
    """``users/<user>/devices/<device>/[stills/]<yyyy>/<mm>/<dd>/<uuid>[.thumb]<ext>``."""
    sub = "stills/" if kind in STILL_KINDS else ""
    suffix = ".thumb" if kind == "still_thumb" else ""
    return (
        f"users/{user_id}/devices/{device_id}/{sub}"
        f"{recorded_at.year:04d}/{recorded_at.month:02d}/{recorded_at.day:02d}/"
        f"{uuid.uuid4().hex}{suffix}{ext}"
    )


def _parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    # Accept trailing 'Z' as UTC for the Pi's convenience.
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


class UploadInitiateView(APIView):
    """Issue a presigned PUT URL for a video the Pi is about to upload."""

    authentication_classes = [DeviceKeyAuthentication]
    # Upload endpoints don't need the TierBasedThrottle — bytes never
    # pass through Django; the 5 GiB single-PUT cap is the real limiter.
    throttle_classes: list = []

    def post(self, request):
        device: Device = request.auth  # set by DeviceKeyAuthentication
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)

        filename = request.data.get("filename", "")
        try:
            size_bytes = int(request.data.get("size_bytes", 0))
        except (TypeError, ValueError):
            return Response({"detail": "size_bytes must be an integer."}, status=400)
        kind = (request.data.get("kind") or "").strip()
        content_type = request.data.get("content_type") or (
            "image/jpeg" if kind in STILL_KINDS else "video/mp4")
        recorded_at = _parse_iso8601(request.data.get("recorded_at")) or timezone.now()

        if not filename:
            return Response({"detail": "filename is required."}, status=400)
        if size_bytes <= 0:
            return Response({"detail": "size_bytes must be positive."}, status=400)
        if size_bytes > MAX_SINGLE_PUT_BYTES:
            return Response(
                {"detail": f"size_bytes exceeds {MAX_SINGLE_PUT_BYTES} (5 GiB single-PUT cap)."},
                status=400,
            )

        if kind in STILL_KINDS:
            ext = ".jpg" if PurePosixPath(filename).suffix.lower() in (".jpg", ".jpeg") else ""
            if not ext:
                return Response({"detail": "a still must be a .jpg."}, status=400)
        else:
            ext = _safe_extension(filename)
        if not ext:
            return Response(
                {"detail": "filename must end in .mp4 / .h264 / .mov / .mkv."},
                status=400,
            )

        # Convert recorded_at to UTC for stable date partitioning.
        recorded_at_utc = recorded_at.astimezone(dt_timezone.utc)
        storage_key = _build_storage_key(
            user_id=device.owner_id,
            device_id=device.id,
            ext=ext,
            recorded_at=recorded_at_utc,
            kind=kind,
        )

        try:
            upload_url = get_s3_client().generate_presigned_url(
                "raw-videos",
                storage_key,
                expiry_hours=PRESIGNED_URL_TTL_SECONDS / 3600,
                permissions="w",
            )
        except Exception as e:
            logger.exception("Failed to presign upload for device %s", device.id)
            return Response({"detail": f"Failed to presign URL: {e}"}, status=500)

        return Response(
            {
                "storage_key": storage_key,
                "upload_url": upload_url,
                "expires_in": PRESIGNED_URL_TTL_SECONDS,
                "method": "PUT",
                "headers": {"Content-Type": content_type},
            },
            status=200,
        )


class UploadCompleteView(APIView):
    """Pi confirms the PUT succeeded; Django creates the Video row."""

    authentication_classes = [DeviceKeyAuthentication]
    throttle_classes: list = []

    def post(self, request):
        device: Device = request.auth
        if device is None or not isinstance(device, Device):
            return Response({"detail": "Device authentication required."}, status=401)

        storage_key = request.data.get("storage_key", "")
        try:
            file_size_bytes = int(request.data.get("file_size_bytes", 0))
        except (TypeError, ValueError):
            return Response({"detail": "file_size_bytes must be an integer."}, status=400)
        recorded_at = _parse_iso8601(request.data.get("recorded_at"))
        title = (request.data.get("title") or "").strip()

        if not storage_key:
            return Response({"detail": "storage_key is required."}, status=400)
        if file_size_bytes <= 0:
            return Response({"detail": "file_size_bytes must be positive."}, status=400)

        # The Pi could theoretically send any storage_key; the prefix check
        # ensures it belongs to *this* device's slice of the bucket.
        expected_prefix = f"users/{device.owner_id}/devices/{device.id}/"
        if not storage_key.startswith(expected_prefix):
            return Response(
                {"detail": "storage_key does not match this device's prefix."},
                status=403,
            )

        # Verify the object actually exists in S3 — defends against a Pi
        # claiming a successful upload when the PUT actually failed.
        s3 = get_s3_client()
        if not s3.blob_exists("raw-videos", storage_key):
            return Response(
                {"detail": "Object not found in S3 — was the PUT successful?"},
                status=404,
            )

        if (request.data.get("kind") or "").strip() == "still":
            return self._complete_still(request, device, storage_key, file_size_bytes,
                                        recorded_at, expected_prefix, s3)

        # Use filename-derived title if the Pi didn't supply one.
        if not title:
            title = PurePosixPath(storage_key).stem

        parsed_site, _parsed_recorded_at = Video.parse_timestamp_from_filename(
            PurePosixPath(storage_key).name,
        )
        final_recorded_at, recorded_at_source = Video.resolve_recorded_at(
            recorded_at, PurePosixPath(storage_key).name)

        metadata = {
            "device_id": device.id,
            "device_name": device.name,
            # Which of the three sources the timestamp came from, so a clip
            # stamped with its upload time (an offline backlog flushed on
            # reconnect) is not mistaken for a measured recording time.
            "recorded_at_source": recorded_at_source,
        }

        video = Video.objects.create(
            user=device.owner,
            device=device,
            title=title,
            storage_key=storage_key,
            file_size_bytes=file_size_bytes,
            status=Video.Status.READY,
            recorded_at=final_recorded_at,
            site_name=parsed_site or device.location or "",
            metadata=metadata,
        )

        # One sampled still for the review grid — background, so the device (or
        # the browser) is not held open for a decode.
        from apps.videos.thumbnails import queue_thumbnail
        queue_thumbnail(video)

        logger.info(
            "Pi upload complete: device=%s user=%s video=%s key=%s size=%d MB",
            device.id, device.owner_id, video.id, storage_key,
            file_size_bytes // (1024 * 1024),
        )

        # Analysis is NOT automatic. The video lands as READY; the user reviews
        # it and analyzes on demand via the existing videos/analysis system. This
        # avoids spawning a job per upload (a snippet backlog would otherwise
        # flood SageMaker + the DB).
        return Response(
            {
                "video_id": video.id,
                "storage_key": storage_key,
                "status": video.status,
                "recorded_at": final_recorded_at.isoformat(),
            },
            status=201,
        )

    def _complete_still(self, request, device, storage_key, file_size_bytes,
                        taken_at, expected_prefix, s3):
        """A full-resolution still: one DeviceStill row. Idempotent on the key,
        so a retry after a lost response does not make a second row."""
        from apps.devices.models import DeviceStill

        thumb_key = (request.data.get("thumb_key") or "").strip()
        if thumb_key and (not thumb_key.startswith(expected_prefix)
                          or not s3.blob_exists("raw-videos", thumb_key)):
            thumb_key = ""  # the full image is what matters; the page falls back

        def _int(name):
            try:
                return max(0, int(request.data.get(name) or 0))
            except (TypeError, ValueError):
                return 0

        try:
            lens = float(request.data.get("lens_position"))
        except (TypeError, ValueError):
            lens = None
        still, created = DeviceStill.objects.get_or_create(
            storage_key=storage_key,
            defaults={
                "device": device,
                "taken_at": taken_at or timezone.now(),
                "thumb_key": thumb_key,
                "width": _int("width"),
                "height": _int("height"),
                "file_size_bytes": file_size_bytes,
                "sensor_mode": str(request.data.get("sensor_mode") or "")[:8],
                "lens_position": lens,
                "source": (request.data.get("source")
                           if request.data.get("source") in ("manual", "burst") else "schedule"),
                "burst_id": str(request.data.get("burst_id") or "")[:40],
                "burst_index": _int("burst_index") if request.data.get("burst_id") else None,
            },
        )
        logger.info("Pi still %s: device=%s still=%s key=%s size=%d MB",
                    "complete" if created else "re-confirmed", device.id, still.id,
                    storage_key, file_size_bytes // (1024 * 1024))
        return Response({"still_id": still.id, "storage_key": storage_key,
                         "taken_at": still.taken_at.isoformat()},
                        status=201 if created else 200)
