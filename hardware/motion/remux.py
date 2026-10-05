"""Snippet remux: .h264 elementary stream -> .mp4 via stream copy (no re-encode)."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from motion.config import log, FPS, WORK_DIR, RECORD_DIR


def _remux(h264_path: Path, mp4_path: Path, fps: float = FPS) -> None:
    """ffmpeg stream-copy .h264 / .h265 -> .mp4, then delete the raw file. Cheap
    (no re-encode). `fps` is the rate the clip was encoded at: a raw elementary
    stream carries no timing."""
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first so the uploader never sees a half-muxed .mp4.
    tmp = mp4_path.with_suffix(".part.mp4")
    hevc = h264_path.suffix == ".h265"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-r", str(fps)]
    if hevc:
        cmd += ["-f", "hevc"]          # raw HEVC isn't sniffed reliably by extension
    cmd += ["-i", str(h264_path), "-c", "copy"]
    if hevc:
        cmd += ["-tag:v", "hvc1"]      # the tag Safari / QuickTime need to play HEVC
    cmd += ["-f", "mp4", str(tmp)]
    try:
        subprocess.run(cmd, check=True)
        tmp.replace(mp4_path)          # atomic within the same filesystem
        log.info("snippet ready: %s (%.1f MB)",
                 mp4_path.name, mp4_path.stat().st_size / (1024 * 1024))
    except subprocess.CalledProcessError as e:
        log.error("remux failed for %s: %s", h264_path.name, e)
        tmp.unlink(missing_ok=True)
    finally:
        h264_path.unlink(missing_ok=True)


def _snippet_paths(now: datetime, codec: str = "h264"):
    """Return (work .h264/.h265 path, final .mp4 path) for a clip starting at `now`."""
    stamp = now.strftime("%Y-%m-%d_%H_%M_%S")
    day = now.strftime("%Y-%m-%d")
    h264 = WORK_DIR / f"{stamp}.{'h265' if codec == 'h265' else 'h264'}"
    mp4 = RECORD_DIR / day / f"{stamp}.mp4"   # matches uploader.py's regex
    return h264, mp4
