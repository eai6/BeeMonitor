"""
Smart Video Re-encoder Proxy.

Downloads annotated video blobs from Azure, checks if they are browser-compatible
(H.264 codec), re-encodes with ffmpeg if needed, caches the re-encoded version
back to Azure with suffix ``_h264.mp4``, and redirects to the SAS URL.
"""

import logging
import os
import subprocess
import tempfile

from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import Http404, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.views import View

from .models import Job, JobResult

logger = logging.getLogger(__name__)


def _generate_sas_url(blob_path: str, container: str = "processed") -> str:
    """Generate a time-limited SAS URL for a blob in Azure Storage."""
    try:
        from datetime import datetime, timedelta, timezone

        from azure.storage.blob import (
            BlobSasPermissions,
            BlobServiceClient,
            generate_blob_sas,
        )

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return ""

        service = BlobServiceClient.from_connection_string(conn_str)
        account_name = service.account_name

        # Extract account key from connection string
        account_key = ""
        for part in conn_str.split(";"):
            if part.startswith("AccountKey="):
                account_key = part.split("=", 1)[1]
                break

        token = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_path,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        return f"https://{account_name}.blob.core.windows.net/{container}/{blob_path}?{token}"
    except Exception as e:
        logger.error("Failed to generate SAS URL for %s: %s", blob_path, e)
        return ""


def _blob_exists(container: str, blob_path: str) -> bool:
    """Check whether a blob already exists in Azure Storage."""
    try:
        from azure.storage.blob import BlobServiceClient

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return False
        service = BlobServiceClient.from_connection_string(conn_str)
        blob_client = service.get_blob_client(container, blob_path)
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False


def _download_blob(container: str, blob_path: str, local_path: str) -> bool:
    """Download a blob from Azure to a local file path."""
    try:
        from azure.storage.blob import BlobServiceClient

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return False
        service = BlobServiceClient.from_connection_string(conn_str)
        blob_client = service.get_blob_client(container, blob_path)
        with open(local_path, "wb") as f:
            data = blob_client.download_blob()
            data.readinto(f)
        return True
    except Exception as e:
        logger.error("Failed to download blob %s/%s: %s", container, blob_path, e)
        return False


def _upload_blob(container: str, blob_path: str, local_path: str) -> bool:
    """Upload a local file to Azure Blob Storage."""
    try:
        from azure.storage.blob import BlobServiceClient

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return False
        service = BlobServiceClient.from_connection_string(conn_str)
        blob_client = service.get_blob_client(container, blob_path)
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=True)
        return True
    except Exception as e:
        logger.error("Failed to upload blob %s/%s: %s", container, blob_path, e)
        return False


def _is_h264(file_path: str) -> bool:
    """Use ffprobe to check if a video file uses the H.264 codec."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        codec = result.stdout.strip().lower()
        return codec == "h264"
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning("ffprobe check failed (will re-encode to be safe): %s", e)
        return False


def _reencode_to_h264(input_path: str, output_path: str) -> bool:
    """Re-encode a video file to H.264 using ffmpeg."""
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", input_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10-minute timeout for long videos
        )
        if result.returncode != 0:
            logger.error("ffmpeg re-encode failed: %s", result.stderr)
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error("ffmpeg re-encode error: %s", e)
        return False


def _h264_blob_path(original_path: str) -> str:
    """Derive the cached H.264 blob path from the original annotated video path."""
    if original_path.endswith(".mp4"):
        return original_path[:-4] + "_h264.mp4"
    return original_path + "_h264.mp4"


class VideoProxyView(LoginRequiredMixin, View):
    """
    Smart proxy for annotated video playback.

    1. Checks if an H.264 cached version already exists on Azure.
    2. If yes, redirects to its SAS URL.
    3. If not, downloads the original blob, checks codec, re-encodes if needed,
       uploads the H.264 version, and redirects.
    """

    def get(self, request, pk):
        job = get_object_or_404(Job, pk=pk, user=request.user)

        try:
            result = job.result
        except JobResult.DoesNotExist:
            raise Http404("No results available for this job.")

        # Determine the annotated video blob path
        user_id = str(job.user.pk)
        modal_id = job.modal_job_id or ""
        prefix = f"{user_id}/{modal_id}"
        annotated_path = result.annotated_video_path or (
            f"{prefix}/annotated_video.mp4" if modal_id else ""
        )

        if not annotated_path:
            raise Http404("No annotated video found for this job.")

        container = "processed"
        h264_path = _h264_blob_path(annotated_path)

        # Step 1: Check if cached H.264 version already exists
        if _blob_exists(container, h264_path):
            sas_url = _generate_sas_url(h264_path, container)
            if sas_url:
                return HttpResponseRedirect(sas_url)

        # Step 2: Download the original annotated video
        with tempfile.TemporaryDirectory() as tmpdir:
            original_local = os.path.join(tmpdir, "original.mp4")
            h264_local = os.path.join(tmpdir, "h264.mp4")

            if not _download_blob(container, annotated_path, original_local):
                # Fallback: redirect to the original SAS URL directly
                sas_url = _generate_sas_url(annotated_path, container)
                if sas_url:
                    return HttpResponseRedirect(sas_url)
                raise Http404("Could not download annotated video.")

            # Step 3: Check codec and re-encode if needed
            if _is_h264(original_local):
                # Already H.264; just redirect to original
                sas_url = _generate_sas_url(annotated_path, container)
                if sas_url:
                    return HttpResponseRedirect(sas_url)
                raise Http404("Could not generate SAS URL.")

            # Step 4: Re-encode to H.264
            if not _reencode_to_h264(original_local, h264_local):
                # Fallback to original if re-encoding fails
                sas_url = _generate_sas_url(annotated_path, container)
                if sas_url:
                    return HttpResponseRedirect(sas_url)
                raise Http404("Re-encoding failed and original is unavailable.")

            # Step 5: Upload re-encoded version back to Azure
            _upload_blob(container, h264_path, h264_local)

        # Step 6: Redirect to the H.264 version
        sas_url = _generate_sas_url(h264_path, container)
        if sas_url:
            return HttpResponseRedirect(sas_url)

        # Ultimate fallback
        sas_url = _generate_sas_url(annotated_path, container)
        if sas_url:
            return HttpResponseRedirect(sas_url)
        raise Http404("Could not serve video.")
