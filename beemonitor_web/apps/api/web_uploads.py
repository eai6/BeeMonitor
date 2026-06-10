"""Web-form upload endpoints — browser PUTs directly to S3.

Same idea as the Pi flow in ``apps/api/uploads.py`` but auth is the user's
Django session (not a device key) and the storage key has no device id —
it uses the legacy ``{user_pk}/{upload_id}/{filename}`` layout that
``_upload_to_storage`` used to write to.

This removes the App Runner instance from the video data path entirely:
Django signs a presigned URL (~10 ms) and confirms the upload (~50 ms);
the bytes go straight from browser to S3. Without this, a single 1 GB
upload was tying up one of two gunicorn workers for the entire transfer.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone as dt_timezone
from pathlib import PurePosixPath

from django.contrib.auth.mixins import LoginRequiredMixin
from django.utils import timezone
from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.videos.models import Video
from config.storage import get_s3_client

logger = logging.getLogger(__name__)

MAX_SINGLE_PUT_BYTES = 5 * 1024 * 1024 * 1024  # S3 single-PUT cap
PRESIGNED_URL_TTL_SECONDS = 6 * 60 * 60  # same as Pi path


def _safe_extension(filename: str) -> str:
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix in {".mp4", ".h264", ".mov", ".mkv"}:
        return suffix
    return ""


class _CsrfExemptSessionAuth(SessionAuthentication):
    """SessionAuthentication without CSRF check.

    The web uploader uses ``fetch`` with credentials; we send the CSRF token
    in the ``X-CSRFToken`` header, and DRF's SessionAuthentication enforces
    that. But Django's CSRF middleware also runs for the page request; for
    the API POST we explicitly trust the session cookie + LoginRequiredMixin
    redirect chain. Use the regular SessionAuthentication so CSRF is enforced.
    """


class WebUploadInitiateView(APIView):
    """Browser asks for a presigned PUT URL. Returns the URL + storage_key."""

    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]
    # Upload endpoints don't need the TierBasedThrottle (which is for the
    # ~60/min CRUD-style API). Bytes never pass through here anyway — the
    # 5 GiB single-PUT cap is the real rate limit.
    throttle_classes: list = []

    def post(self, request):
        user = request.user
        filename = request.data.get("filename", "")
        try:
            size_bytes = int(request.data.get("size_bytes", 0))
        except (TypeError, ValueError):
            return Response({"detail": "size_bytes must be an integer."}, status=400)
        content_type = request.data.get("content_type") or "video/mp4"

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

        upload_id = uuid.uuid4().hex[:12]
        # Match _upload_to_storage's existing layout so a future audit of
        # raw-videos keys doesn't see a sudden schema break.
        storage_key = f"{user.pk}/{upload_id}/{filename}"

        try:
            upload_url = get_s3_client().generate_presigned_url(
                "raw-videos",
                storage_key,
                expiry_hours=PRESIGNED_URL_TTL_SECONDS / 3600,
                permissions="w",
            )
        except Exception as e:
            logger.exception("Failed to presign web upload for user %s", user.pk)
            return Response({"detail": f"Failed to presign URL: {e}"}, status=500)

        return Response({
            "storage_key": storage_key,
            "upload_url": upload_url,
            "expires_in": PRESIGNED_URL_TTL_SECONDS,
            "method": "PUT",
            "headers": {"Content-Type": content_type},
        }, status=200)


class WebUploadCompleteView(APIView):
    """Browser tells Django the PUT succeeded; Django creates the Video row."""

    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes: list = []

    def post(self, request):
        user = request.user
        storage_key = request.data.get("storage_key", "")
        try:
            file_size_bytes = int(request.data.get("file_size_bytes", 0))
        except (TypeError, ValueError):
            return Response({"detail": "file_size_bytes must be an integer."}, status=400)
        title = (request.data.get("title") or "").strip()
        site_name_override = (request.data.get("site_name") or "").strip()
        device_id = request.data.get("device_id")

        if not storage_key:
            return Response({"detail": "storage_key is required."}, status=400)
        if file_size_bytes <= 0:
            return Response({"detail": "file_size_bytes must be positive."}, status=400)

        # Same defence-in-depth as the Pi path: confirm the key starts
        # with this user's prefix. The presigned URL is already locked to
        # one key, but a forged complete-call shouldn't be able to claim
        # someone else's S3 object.
        expected_prefix = f"{user.pk}/"
        if not storage_key.startswith(expected_prefix):
            return Response(
                {"detail": "storage_key does not match this user's prefix."},
                status=403,
            )

        s3 = get_s3_client()
        if not s3.blob_exists("raw-videos", storage_key):
            return Response(
                {"detail": "Object not found in S3 — was the PUT successful?"},
                status=404,
            )

        filename = PurePosixPath(storage_key).name
        if not title:
            title = filename.rsplit(".", 1)[0] if "." in filename else filename

        # Optional device attribution (e.g. uploading USB-copied clips so they
        # land in the cloud the same as cellular/WiFi uploads). Only devices the
        # user owns; the site name defaults to the device's location.
        device = None
        if device_id:
            from apps.devices.models import Device
            device = Device.objects.filter(pk=device_id, owner=user).first()
            if device is None:
                return Response({"detail": "Unknown device."}, status=400)

        parsed_site, parsed_recorded_at = Video.parse_timestamp_from_filename(filename)
        final_site = site_name_override or (device.location if device else "") or parsed_site or ""
        final_recorded_at = parsed_recorded_at or timezone.now().astimezone(dt_timezone.utc)

        video = Video.objects.create(
            user=user,
            device=device,
            title=title,
            storage_key=storage_key,
            file_size_bytes=file_size_bytes,
            status=Video.Status.READY,
            recorded_at=final_recorded_at,
            site_name=final_site,
        )

        logger.info(
            "Web upload complete: user=%s video=%s key=%s size=%d MB",
            user.pk, video.id, storage_key, file_size_bytes // (1024 * 1024),
        )

        return Response({
            "video_id": video.id,
            "storage_key": storage_key,
            "status": video.status,
            "title": video.title,
        }, status=201)
