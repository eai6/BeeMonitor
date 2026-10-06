"""Browser uploads in parts — any size, resumable (memory/44).

The single presigned PUT in ``web_uploads.py`` caps a file at 5 GiB and a
dropped connection restarts it from zero. Here the browser uploads an S3
multipart upload directly: Django opens it, signs URLs for the parts it asks
for, lists the parts S3 already has (so a re-added file resumes from where it
stopped), and completes it. Bytes still never pass through Django.

    initiate  {filename, size_bytes, content_type}  -> {storage_key, upload_id, part_size, part_count}
    sign      {storage_key, upload_id, part_numbers}  -> {urls: {n: url}}
    parts     {storage_key, upload_id}                -> {parts: [{part_number, etag, size}]}
    complete  {storage_key, upload_id, parts, ...video fields}  -> the Video
    abort     {storage_key, upload_id}
    check     {files: [{name, size}]}                 -> {duplicates: [...]}

The bucket's CORS rule already allows PUT and exposes ``ETag`` (infra/aws),
which is what the browser needs to collect each part's tag.
"""

from __future__ import annotations

import logging
import math
import uuid
from pathlib import PurePosixPath

from rest_framework.authentication import SessionAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.videos.models import Site, Video
from config.storage import get_s3_client

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".h264", ".avi"}
# Photos from any camera become kind=photo rows (memory/45).
PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | PHOTO_EXTENSIONS
MIN_PART = 16 * 1024 * 1024          # S3 minimum is 5 MiB; 16 keeps part counts sane
MAX_PARTS = 9000                     # S3 allows 10,000; leave headroom
SIGN_TTL = 6 * 60 * 60
MAX_SIGN_PER_CALL = 100


def part_size_for(size_bytes: int) -> int:
    """Smallest part size (a whole MiB, >= 16 MiB) that fits in MAX_PARTS."""
    need = math.ceil(size_bytes / MAX_PARTS)
    mib = 1024 * 1024
    return max(MIN_PART, math.ceil(need / mib) * mib)


def _s3():
    client = get_s3_client()
    return client, client._client, client._bucket("raw-videos")


def _own_key(user, key) -> bool:
    return bool(key) and key.startswith(f"{user.pk}/")


class _Base(APIView):
    authentication_classes = [SessionAuthentication]
    permission_classes = [IsAuthenticated]
    throttle_classes: list = []


class MultipartInitiateView(_Base):
    def post(self, request):
        filename = PurePosixPath(str(request.data.get("filename", ""))).name
        try:
            size = int(request.data.get("size_bytes", 0))
        except (TypeError, ValueError):
            return Response({"detail": "size_bytes must be an integer."}, status=400)
        if not filename or size <= 0:
            return Response({"detail": "filename and a positive size_bytes are required."}, status=400)
        if PurePosixPath(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            return Response({"detail": "Upload videos (.mp4, .mov, .mkv, .avi, .h264) or photos "
                                       "(.jpg, .png, .tif, .heic)."}, status=400)

        key = f"{request.user.pk}/{uuid.uuid4().hex[:12]}/{filename}"
        wrapper, s3, bucket = _s3()
        try:
            out = s3.create_multipart_upload(
                Bucket=bucket, Key=wrapper._key(key),
                ContentType=request.data.get("content_type") or "video/mp4")
        except Exception as e:
            logger.exception("multipart initiate failed for user %s", request.user.pk)
            return Response({"detail": f"Could not start the upload: {e}"}, status=500)
        part = part_size_for(size)
        return Response({"storage_key": key, "upload_id": out["UploadId"], "part_size": part,
                         "part_count": math.ceil(size / part)})


class MultipartSignView(_Base):
    def post(self, request):
        key, upload_id = request.data.get("storage_key"), request.data.get("upload_id")
        if not _own_key(request.user, key) or not upload_id:
            return Response({"detail": "Unknown upload."}, status=403)
        try:
            numbers = sorted({int(n) for n in request.data.get("part_numbers") or []})
        except (TypeError, ValueError):
            return Response({"detail": "part_numbers must be integers."}, status=400)
        numbers = [n for n in numbers if 1 <= n <= 10000][:MAX_SIGN_PER_CALL]
        wrapper, s3, bucket = _s3()
        urls = {n: s3.generate_presigned_url(
            "upload_part", ExpiresIn=SIGN_TTL,
            Params={"Bucket": bucket, "Key": wrapper._key(key), "UploadId": upload_id,
                    "PartNumber": n}) for n in numbers}
        return Response({"urls": urls})


class MultipartPartsView(_Base):
    """The parts S3 already holds — what a resumed upload can skip."""

    def post(self, request):
        key, upload_id = request.data.get("storage_key"), request.data.get("upload_id")
        if not _own_key(request.user, key) or not upload_id:
            return Response({"detail": "Unknown upload."}, status=403)
        wrapper, s3, bucket = _s3()
        parts, marker = [], 0
        try:
            while True:
                page = s3.list_parts(Bucket=bucket, Key=wrapper._key(key), UploadId=upload_id,
                                     PartNumberMarker=marker)
                parts += [{"part_number": p["PartNumber"], "etag": p["ETag"], "size": p["Size"]}
                          for p in page.get("Parts", [])]
                if not page.get("IsTruncated"):
                    break
                marker = page.get("NextPartNumberMarker", 0)
        except s3.exceptions.NoSuchUpload:
            return Response({"detail": "That upload expired; it will start again."}, status=404)
        return Response({"parts": parts})


class MultipartAbortView(_Base):
    def post(self, request):
        key, upload_id = request.data.get("storage_key"), request.data.get("upload_id")
        if not _own_key(request.user, key) or not upload_id:
            return Response({"detail": "Unknown upload."}, status=403)
        wrapper, s3, bucket = _s3()
        try:
            s3.abort_multipart_upload(Bucket=bucket, Key=wrapper._key(key), UploadId=upload_id)
        except Exception:
            logger.info("abort of %s ignored", key, exc_info=True)
        return Response({"ok": True})


class MultipartCompleteView(_Base):
    def post(self, request):
        key, upload_id = request.data.get("storage_key"), request.data.get("upload_id")
        if not _own_key(request.user, key) or not upload_id:
            return Response({"detail": "Unknown upload."}, status=403)
        try:
            parts = sorted(({"PartNumber": int(p["part_number"]), "ETag": str(p["etag"])}
                            for p in request.data.get("parts") or []),
                           key=lambda p: p["PartNumber"])
            size = int(request.data.get("file_size_bytes", 0))
        except (TypeError, ValueError, KeyError):
            return Response({"detail": "parts must be [{part_number, etag}]."}, status=400)
        if not parts or size <= 0:
            return Response({"detail": "parts and file_size_bytes are required."}, status=400)

        wrapper, s3, bucket = _s3()
        try:
            s3.complete_multipart_upload(Bucket=bucket, Key=wrapper._key(key), UploadId=upload_id,
                                         MultipartUpload={"Parts": parts})
        except Exception as e:
            logger.exception("multipart complete failed for %s", key)
            return Response({"detail": f"Could not finish the upload: {e}"}, status=400)
        try:
            video = create_uploaded_video(request.user, key, size, request.data)
        except ValueError as e:
            return Response({"detail": str(e)}, status=400)
        return Response({"video_id": video.id, "kind": video.kind, "title": video.title,
                         "recorded_at": video.recorded_at.isoformat() if video.recorded_at else None,
                         "recorded_at_source": video.metadata.get("recorded_at_source"),
                         "site_name": video.site_name}, status=201)


class UploadCheckView(_Base):
    """Files this user already uploaded (same original name and size)."""

    def post(self, request):
        files = request.data.get("files") or []
        out = []
        for f in files[:2000]:
            try:
                name, size = str(f.get("name") or ""), int(f.get("size") or 0)
            except (TypeError, ValueError, AttributeError):
                continue
            if not name or size <= 0:
                continue
            hit = (Video.everything.filter(user=request.user, file_size_bytes=size)
                   .filter(metadata__original_filename=name).order_by("-uploaded_at").first())
            if hit:
                out.append({"name": name, "size": size, "video_id": hit.id,
                            "uploaded_at": hit.uploaded_at.isoformat()})
        return Response({"duplicates": out})


def create_uploaded_video(user, storage_key, size, data):
    """The Video row for a finished browser upload.

    ``data`` may carry: title, original_filename, file_recorded_at (the
    file's own creation time, read in the browser), user_recorded_at (the
    batch start time typed by the uploader), site_id or site_name, device_id,
    batch. Every one is optional; the clip is created whatever is missing.
    """
    from apps.devices.models import Device
    from apps.videos.thumbnails import queue_thumbnail

    original = PurePosixPath(str(data.get("original_filename") or storage_key)).name
    title = (str(data.get("title") or "").strip()
             or (original.rsplit(".", 1)[0] if "." in original else original))

    device = None
    if data.get("device_id"):
        device = Device.objects.filter(pk=data.get("device_id"), owner=user).first()
        if device is None:
            raise ValueError("Unknown device.")

    site = None
    if data.get("site_id"):
        site = Site.objects.filter(pk=data.get("site_id"), user=user).first()
        if site is None:
            raise ValueError("Unknown site.")
    site_name = (site.name if site else "") or str(data.get("site_name") or "").strip() \
        or (device.location if device else "") or Video.parse_timestamp_from_filename(original)[0]

    recorded_at, source = Video.resolve_upload_recorded_at(
        data.get("file_recorded_at"), original, data.get("user_recorded_at"))

    metadata = {"recorded_at_source": source, "original_filename": original,
                "uploaded_via": "web"}
    batch = str(data.get("batch") or "").strip()[:120]
    if batch:
        metadata["batch"] = batch
    suffix = PurePosixPath(original).suffix.lower()
    is_photo = suffix in PHOTO_EXTENSIONS
    if suffix == ".avi":
        metadata["needs_transcode"] = True

    video = Video.everything.create(
        user=user, device=device, site=site, title=title[:300], storage_key=storage_key,
        file_size_bytes=size, status=Video.Status.READY, recorded_at=recorded_at,
        site_name=site_name[:200], metadata=metadata,
        kind=Video.Kind.PHOTO if is_photo else Video.Kind.VIDEO)
    if not is_photo:
        queue_thumbnail(video)          # photos show from the file itself
    logger.info("browser upload: user=%s video=%s %s MB time=%s", user.pk, video.id,
                size // (1024 * 1024), source)
    return video
