"""Frames and properties of a clip, read over a URL (no download).

One full-resolution frame at any time, for drawing a layout on.

Not the thumbnail: that one is downscaled and turned for display when the unit
is mounted upside down, so coordinates drawn on it would not match what the
GPU sees. ffmpeg seeks a presigned URL with range requests, so a frame from
minute 30 of a 2 GB clip costs a few megabytes, not the whole download.
"""

import logging
import subprocess

logger = logging.getLogger(__name__)

TIMEOUT_SECONDS = 60


def readable_url(video) -> "str | None":
    """A short-lived URL the clip's bytes can be read from: our bucket, or the
    user's own bucket for a clip kept there (memory/54)."""
    key = video.storage_key or ""
    if not key or video.raw_deleted_at:
        return None
    if key.startswith("s3://"):
        if not video.source_id:
            return None
        try:
            from apps.sources import crypto, s3
            bucket, remote = s3.parse_remote(video)
            return s3.presign(crypto.decrypt(video.source), bucket, remote)
        except Exception:
            logger.info("frame: can't presign linked clip %s", video.pk)
            return None
    from config.storage import presigned_get
    return presigned_get(key)


def frame_jpeg(video, seconds: float = 0.0, width: "int | None" = None) -> "bytes | None":
    """JPEG bytes of the frame at ``seconds``, or None when it can't be read.
    ``width`` scales it down (thumbnails); None keeps full resolution."""
    url = readable_url(video)
    if not url:
        return None
    duration = float(video.duration_seconds or 0)
    seconds = max(0.0, min(float(seconds or 0), max(duration - 0.5, 0.0) if duration else float(seconds or 0)))
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{seconds:.3f}", "-i", url,
           "-frames:v", "1", "-q:v", "3"]
    if width:
        cmd += ["-vf", f"scale='min({int(width)},iw)':-2"]
    cmd += ["-f", "image2", "-c:v", "mjpeg", "pipe:1"]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("frame: ffmpeg failed for video %s at %.1fs: %s", video.pk, seconds, e)
        return None
    if out.returncode != 0 or not out.stdout:
        logger.warning("frame: no frame for video %s at %.1fs", video.pk, seconds)
        return None
    return out.stdout


def probe(video) -> dict:
    """``{fps, duration_seconds, width, height}`` from the container, read
    over a URL (no download). Empty when it can't be read."""
    import json

    url = readable_url(video)
    if not url:
        return {}
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
           "stream=width,height,avg_frame_rate:format=duration", "-of", "json", url]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=TIMEOUT_SECONDS, check=False)
        data = json.loads(out.stdout or b"{}")
    except (OSError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning("probe: failed for video %s: %s", video.pk, e)
        return {}
    stream = (data.get("streams") or [{}])[0]
    props = {}
    try:
        num, _, den = str(stream.get("avg_frame_rate") or "0/1").partition("/")
        fps = float(num) / float(den or 1)
        if fps > 0:
            props["fps"] = round(fps, 3)
    except (ValueError, ZeroDivisionError):
        pass
    for field in ("width", "height"):
        if stream.get(field):
            props[field] = int(stream[field])
    try:
        duration = float((data.get("format") or {}).get("duration") or 0)
        if duration > 0:
            props["duration_seconds"] = round(duration, 3)
    except ValueError:
        pass
    return props
