"""BeeMonitor Pi → S3 uploader.

Runs as a separate systemd service from ``main.py`` (the recorder) so a
network outage cannot interrupt recording — videos accumulate on disk and
the uploader catches up when connectivity returns.

Flow per file:
    1. POST /api/v1/uploads/initiate  → presigned PUT URL
    2. PUT the .mp4 directly to S3 (long-running, GB-scale OK)
    3. POST /api/v1/uploads/complete  → Django creates the Video row
    4. Touch ``<file>.uploaded`` sidecar so we never re-upload it.

Config — set in ``/etc/beemonitor/uploader.env`` (read by the systemd unit):
    BEEMONITOR_API_BASE   = https://mqnafc3ejc.us-east-1.awsapprunner.com
    BEEMONITOR_DEVICE_KEY = bmk_device_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    BEEMONITOR_RECORD_DIR = /home/apis/Desktop/cameraOutput/beeHotel
    BEEMONITOR_POLL_SECONDS = 30
"""

from __future__ import annotations

import logging
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
except ImportError:  # pragma: no cover — Pi image installs it via apt/pip
    print("uploader requires 'requests' — pip install requests", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("BEEMONITOR_API_BASE", "").rstrip("/")
DEVICE_KEY = os.environ.get("BEEMONITOR_DEVICE_KEY", "")
RECORD_DIR = Path(os.environ.get(
    "BEEMONITOR_RECORD_DIR", "/home/apis/Desktop/cameraOutput/beeHotel",
))
POLL_SECONDS = int(os.environ.get("BEEMONITOR_POLL_SECONDS", "30"))
PUT_TIMEOUT_SECONDS = int(os.environ.get("BEEMONITOR_PUT_TIMEOUT", "7200"))  # 2h


# Backoff for transient API / network failures.
INITIAL_BACKOFF = 5
MAX_BACKOFF = 5 * 60  # 5 minutes

# Files smaller than this are skipped (likely still being written by the recorder).
MIN_FILE_BYTES = 1024  # 1 KiB

# Recording filename pattern from `main.py`: site_YYYY-MM-DD_HH_MM_SS.mp4
RECORDED_AT_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})[_\-](\d{2})[_\-](\d{2})[_\-](\d{2})\.\w+$"
)


logging.basicConfig(
    format="%(asctime)s %(levelname)s uploader %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("uploader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    missing = []
    if not API_BASE:
        missing.append("BEEMONITOR_API_BASE")
    if not DEVICE_KEY:
        missing.append("BEEMONITOR_DEVICE_KEY")
    if not DEVICE_KEY.startswith("bmk_device_"):
        log.error("BEEMONITOR_DEVICE_KEY must start with 'bmk_device_'")
        sys.exit(2)
    if missing:
        log.error("Missing required env: %s", ", ".join(missing))
        sys.exit(2)
    if not RECORD_DIR.is_dir():
        log.error("Recording directory does not exist: %s", RECORD_DIR)
        sys.exit(2)


def _parse_recorded_at(filename: str) -> datetime | None:
    """Pull a UTC datetime out of the recorder's filename convention."""
    m = RECORDED_AT_RE.search(filename)
    if not m:
        return None
    date, hh, mm, ss = m.groups()
    try:
        return datetime.fromisoformat(f"{date}T{hh}:{mm}:{ss}+00:00")
    except ValueError:
        return None


def _list_pending(record_dir: Path) -> list[Path]:
    """Find .mp4 files that don't yet have a .uploaded sidecar."""
    pending: list[Path] = []
    for mp4 in record_dir.rglob("*.mp4"):
        if mp4.with_suffix(mp4.suffix + ".uploaded").exists():
            continue
        # Skip files still being written: heuristic = size > min AND mtime > 60s ago.
        try:
            stat = mp4.stat()
        except FileNotFoundError:
            continue
        if stat.st_size < MIN_FILE_BYTES:
            continue
        if time.time() - stat.st_mtime < 60:
            continue
        pending.append(mp4)
    # Oldest first so we drain the backlog in chronological order.
    pending.sort(key=lambda p: p.stat().st_mtime)
    return pending


# ---------------------------------------------------------------------------
# Upload flow
# ---------------------------------------------------------------------------

def _api_post(path: str, json: dict) -> dict:
    """POST to the Django API with the device key. Raises on non-2xx."""
    url = urljoin(API_BASE + "/", path.lstrip("/"))
    r = requests.post(
        url,
        json=json,
        headers={"Authorization": f"Bearer {DEVICE_KEY}"},
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
    return r.json()


def _put_to_s3(presigned_url: str, file_path: Path, content_type: str) -> None:
    """Stream the file to S3 via the presigned URL."""
    size = file_path.stat().st_size
    with open(file_path, "rb") as fh:
        r = requests.put(
            presigned_url,
            data=fh,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(size),
            },
            timeout=PUT_TIMEOUT_SECONDS,
        )
    if not r.ok:
        raise RuntimeError(f"S3 PUT -> {r.status_code}: {r.text[:300]}")


def _upload_one(file_path: Path) -> None:
    """End-to-end upload of a single .mp4. Touches .uploaded on success."""
    size = file_path.stat().st_size
    recorded_at = _parse_recorded_at(file_path.name) or datetime.now(timezone.utc)
    content_type = "video/mp4"

    log.info(
        "uploading %s (%.1f MB, recorded_at=%s)",
        file_path.name, size / (1024 * 1024), recorded_at.isoformat(),
    )

    # 1. Ask Django for a presigned URL.
    initiate = _api_post("/api/v1/uploads/initiate", {
        "filename": file_path.name,
        "size_bytes": size,
        "content_type": content_type,
        "recorded_at": recorded_at.isoformat(),
    })
    storage_key = initiate["storage_key"]
    upload_url = initiate["upload_url"]

    # 2. Stream the bytes to S3.
    _put_to_s3(upload_url, file_path, content_type)

    # 3. Tell Django the PUT succeeded.
    complete = _api_post("/api/v1/uploads/complete", {
        "storage_key": storage_key,
        "file_size_bytes": size,
        "recorded_at": recorded_at.isoformat(),
    })

    # 4. Mark as done so we never re-upload.
    file_path.with_suffix(file_path.suffix + ".uploaded").write_text(
        f"video_id={complete['video_id']}\nstorage_key={storage_key}\n",
    )
    log.info(
        "uploaded video_id=%s key=%s",
        complete["video_id"], storage_key,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_running = True


def _handle_signal(signum, frame):  # noqa: ARG001
    global _running
    log.info("signal %s received, draining current upload then exiting", signum)
    _running = False


def main() -> int:
    _validate_config()
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    log.info("uploader started — API=%s, record_dir=%s", API_BASE, RECORD_DIR)

    backoff = INITIAL_BACKOFF
    while _running:
        try:
            pending = _list_pending(RECORD_DIR)
        except OSError as e:
            log.error("listing recording dir failed: %s", e)
            time.sleep(POLL_SECONDS)
            continue

        if not pending:
            time.sleep(POLL_SECONDS)
            backoff = INITIAL_BACKOFF
            continue

        log.info("found %d pending file(s)", len(pending))
        for mp4 in pending:
            if not _running:
                break
            try:
                _upload_one(mp4)
                backoff = INITIAL_BACKOFF  # reset on success
            except Exception as e:
                log.exception("upload failed for %s: %s", mp4.name, e)
                # Don't touch the .uploaded marker — we'll retry next loop.
                log.info("backing off %ds", backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
                # Move to next file rather than spin on the failing one.

    log.info("uploader stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
