"""
Smart Video Re-encoder Proxy.

Downloads annotated video objects from S3, checks if they are browser-compatible
(H.264 codec), re-encodes with ffmpeg if needed, caches the re-encoded version
back to S3 with suffix ``_h264.mp4``, and redirects to a presigned URL.
"""

import logging
import os
import subprocess
import tempfile

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import Http404, HttpResponseRedirect
from django.shortcuts import get_object_or_404
from django.views import View

from config.storage import get_s3_client
from .models import Job, JobResult

logger = logging.getLogger(__name__)


def _generate_presigned_url(blob_path: str, container: str = "processed") -> str:
    if not blob_path:
        return ""
    try:
        return get_s3_client().generate_presigned_url(container, blob_path)
    except Exception as e:
        logger.error("Failed to presign %s/%s: %s", container, blob_path, e)
        return ""


def _blob_exists(container: str, blob_path: str) -> bool:
    try:
        return get_s3_client().blob_exists(container, blob_path)
    except Exception:
        return False


def _download_blob(container: str, blob_path: str, local_path: str) -> bool:
    try:
        get_s3_client().download_file(container, blob_path, local_path)
        return True
    except Exception as e:
        logger.error("Failed to download %s/%s: %s", container, blob_path, e)
        return False


def _upload_blob(container: str, blob_path: str, local_path: str) -> bool:
    try:
        get_s3_client().upload_file(container, blob_path, local_path)
        return True
    except Exception as e:
        logger.error("Failed to upload %s/%s: %s", container, blob_path, e)
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

    1. Checks if a cached H.264 version already exists in S3.
    2. If yes, redirects to a presigned URL.
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
            sas_url = _generate_presigned_url(h264_path, container)
            if sas_url:
                return HttpResponseRedirect(sas_url)

        # Step 2: Download the original annotated video
        with tempfile.TemporaryDirectory() as tmpdir:
            original_local = os.path.join(tmpdir, "original.mp4")
            h264_local = os.path.join(tmpdir, "h264.mp4")

            if not _download_blob(container, annotated_path, original_local):
                # Fallback: redirect to the original presigned URL
                sas_url = _generate_presigned_url(annotated_path, container)
                if sas_url:
                    return HttpResponseRedirect(sas_url)
                raise Http404("Could not download annotated video.")

            # Step 3: Check codec and re-encode if needed
            if _is_h264(original_local):
                # Already H.264; just redirect to original
                sas_url = _generate_presigned_url(annotated_path, container)
                if sas_url:
                    return HttpResponseRedirect(sas_url)
                raise Http404("Could not generate presigned URL.")

            # Step 4: Re-encode to H.264
            if not _reencode_to_h264(original_local, h264_local):
                # Fallback to original if re-encoding fails
                sas_url = _generate_presigned_url(annotated_path, container)
                if sas_url:
                    return HttpResponseRedirect(sas_url)
                raise Http404("Re-encoding failed and original is unavailable.")

            # Step 5: Upload re-encoded version back to S3
            _upload_blob(container, h264_path, h264_local)

        # Step 6: Redirect to the H.264 version
        sas_url = _generate_presigned_url(h264_path, container)
        if sas_url:
            return HttpResponseRedirect(sas_url)

        # Ultimate fallback
        sas_url = _generate_presigned_url(annotated_path, container)
        if sas_url:
            return HttpResponseRedirect(sas_url)
        raise Http404("Could not serve video.")
