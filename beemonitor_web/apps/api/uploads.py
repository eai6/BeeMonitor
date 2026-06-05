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


def _build_storage_key(user_id: int, device_id: int, ext: str, recorded_at: datetime) -> str:
    """``users/<user>/devices/<device>/<yyyy>/<mm>/<dd>/<uuid><ext>``."""
    return (
        f"users/{user_id}/devices/{device_id}/"
        f"{recorded_at.year:04d}/{recorded_at.month:02d}/{recorded_at.day:02d}/"
        f"{uuid.uuid4().hex}{ext}"
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
        content_type = request.data.get("content_type") or "video/mp4"
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

        # Use filename-derived title if the Pi didn't supply one.
        if not title:
            title = PurePosixPath(storage_key).stem

        parsed_site, parsed_recorded_at = Video.parse_timestamp_from_filename(
            PurePosixPath(storage_key).name,
        )
        final_recorded_at = recorded_at or parsed_recorded_at or timezone.now()

        video = Video.objects.create(
            user=device.owner,
            device=device,
            title=title,
            storage_key=storage_key,
            file_size_bytes=file_size_bytes,
            status=Video.Status.READY,
            recorded_at=final_recorded_at,
            site_name=parsed_site or device.location or "",
            metadata={"device_id": device.id, "device_name": device.name},
        )

        logger.info(
            "Pi upload complete: device=%s user=%s video=%s key=%s size=%d MB",
            device.id, device.owner_id, video.id, storage_key,
            file_size_bytes // (1024 * 1024),
        )

        # Phase 4: auto-spawn a SageMaker analysis job. Best-effort —
        # a spawn failure must not fail the upload (the video is safely
        # in S3 either way). The job row carries the failure if SM is down.
        job_id = None
        try:
            job_id = _enqueue_pi_analysis(video, device)
        except Exception as e:
            logger.exception("auto-spawn analysis failed for video %s: %s", video.id, e)

        return Response(
            {
                "video_id": video.id,
                "storage_key": storage_key,
                "status": video.status,
                "recorded_at": final_recorded_at.isoformat(),
                "analysis_job_id": job_id,
            },
            status=201,
        )


def _enqueue_pi_analysis(video, device) -> int | None:
    """Create an analysis Job for a Pi-uploaded video and fire it off.

    Threaded spawn — same pattern as ``JobCreateView`` (analysis/views.py).
    Returns the Job pk if the spawn was enqueued, ``None`` if the SageMaker
    endpoint isn't configured (e.g. local dev / tests).
    """
    from django.conf import settings as _s
    if not getattr(_s, "SAGEMAKER_ENDPOINT_NAME", ""):
        return None  # Phase 4 endpoint not deployed yet — keep upload happy.

    import uuid as _uuid

    from apps.analysis.models import Job
    from apps.analysis.views import spawn_gpu_job_async
    from django.utils import timezone

    job = Job.objects.create(
        user=device.owner,
        video=video,
        config={
            "detection_mode": "yolo",
            "confidence_threshold": 0.25,
            "source": "pi-upload",
            "device_id": device.id,
        },
        status=Job.Status.PROCESSING,
        started_at=timezone.now(),
        modal_job_id=f"pi_{_uuid.uuid4().hex[:12]}",
    )
    # Bounded pool, not a raw thread — a backlog drain must not open one DB
    # connection per upload and exhaust the database.
    spawn_gpu_job_async(job.pk)
    return job.pk
