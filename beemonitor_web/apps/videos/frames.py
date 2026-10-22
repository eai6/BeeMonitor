"""One full-resolution frame of a clip, at any time, for drawing a layout on.

Not the thumbnail: that one is downscaled and turned for display when the unit
is mounted upside down, so coordinates drawn on it would not match what the
GPU sees. ffmpeg seeks a presigned URL with range requests, so a frame from
minute 30 of a 2 GB clip costs a few megabytes, not the whole download.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60


def frame_jpeg(video, seconds: float = 0.0) -> "bytes | None":
    """JPEG bytes of the frame at ``seconds``, or None when it can't be read."""
    from config.storage import presigned_get

    key = video.storage_key or ""
    if not key or key.startswith("s3://") or video.raw_deleted_at:
        return None
    url = presigned_get(key)
    if not url:
        return None
    duration = float(video.duration_seconds or 0)
    seconds = max(0.0, min(float(seconds or 0), max(duration - 0.5, 0.0) if duration else float(seconds or 0)))
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", url,
           "-frames:v", "1", "-q:v", "3", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("frame: ffmpeg failed for video %s at %.1fs: %s", video.pk, seconds, e)
        return None
    if out.returncode != 0 or not out.stdout:
        logger.warning("frame: no frame for video %s at %.1fs", video.pk, seconds)
        return None
    return out.stdout
