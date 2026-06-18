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
    BEEMONITOR_RECORD_DIR = /home/beemonitor/Desktop/cameraOutput/beeHotel
    BEEMONITOR_POLL_SECONDS = 30
"""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
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
    "BEEMONITOR_RECORD_DIR", "/home/beemonitor/Desktop/cameraOutput/beeHotel",
))
POLL_SECONDS = int(os.environ.get("BEEMONITOR_POLL_SECONDS", "30"))
PUT_TIMEOUT_SECONDS = int(os.environ.get("BEEMONITOR_PUT_TIMEOUT", "7200"))  # 2h
# Video is expensive over cellular — only upload when WiFi is available. The
# hourly telemetry beat is what flows over cellular; bulk video waits for WiFi.
# Set BEEMONITOR_WIFI_ONLY_VIDEO=false to upload over any link.
WIFI_ONLY_VIDEO = os.environ.get(
    "BEEMONITOR_WIFI_ONLY_VIDEO", "true").strip().lower() in {"1", "true", "yes", "on"}


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


def _wifi_connected() -> bool:
    """True if a WiFi link is up (so bulk video upload is cheap).

    Prefers NetworkManager; falls back to checking for a connected wl* iface
    with an IPv4 address. Cellular (wwan0) does not count.
    """
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "TYPE,STATE", "device"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        if out.strip():
            for line in out.splitlines():
                typ, _, state = line.partition(":")
                if typ == "wifi" and state.startswith("connected"):
                    return True
            return False  # nmcli answered and no wifi is connected
    except (OSError, subprocess.SubprocessError):
        pass
    # Fallback: any wl* interface holding an IPv4 address.
    try:
        out = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1].startswith("wl"):
                return True
    except (OSError, subprocess.SubprocessError):
        pass
    return False


def _list_pending_frames(record_dir: Path) -> list[Path]:
    """Find durable activity-frame groups awaiting upload: ``<day>/frames/*.json``
    (the JSON is written last, so its presence means the crop+source are complete).
    Skip very-fresh files in case a write is still in flight."""
    out: list[Path] = []
    for meta in record_dir.glob("*/frames/*.json"):
        try:
            if time.time() - meta.stat().st_mtime < 30:
                continue
        except OSError:
            continue
        out.append(meta)
    out.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    return out


def _list_pending(record_dir: Path) -> list[Path]:
    """Find .mp4 files that don't yet have a .uploaded sidecar."""
    pending: list[Path] = []
    for mp4 in record_dir.rglob("*.mp4"):
        if ".part." in mp4.name:
            continue
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


def _read_bee_tag(file_path: Path) -> "dict | None":
    """The recorder's bee-confirmation tag (`<clip>.mp4.bee.json`), if present.

    Rides with the clip so the cloud can flag unconfirmed clips and skip the
    expensive automated video analysis on them. ALL clips upload regardless."""
    try:
        return json.loads(
            file_path.with_suffix(file_path.suffix + ".bee.json").read_text())
    except (OSError, ValueError):
        return None


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

    # 3. Tell Django the PUT succeeded (carrying the bee-confirmation tag, if any,
    #    so the cloud can flag the clip + gate its own automated analysis).
    complete_payload = {
        "storage_key": storage_key,
        "file_size_bytes": size,
        "recorded_at": recorded_at.isoformat(),
    }
    bee = _read_bee_tag(file_path)
    if bee:
        complete_payload["bee"] = bee
    complete = _api_post("/api/v1/uploads/complete", complete_payload)

    # 4. Mark as done so we never re-upload.
    file_path.with_suffix(file_path.suffix + ".uploaded").write_text(
        f"video_id={complete['video_id']}\nstorage_key={storage_key}\n",
    )
    log.info(
        "uploaded video_id=%s key=%s",
        complete["video_id"], storage_key,
    )


def _post_frame(jpg_path: Path, meta: dict) -> None:
    """Multipart-POST one activity frame (crop or source) to the frames endpoint."""
    url = urljoin(API_BASE + "/", "api/v1/devices/frames")
    with open(jpg_path, "rb") as fh:
        r = requests.post(
            url,
            headers={"Authorization": f"Bearer {DEVICE_KEY}"},
            data={"meta": json.dumps(meta)},
            files={"frame": (jpg_path.name, fh, "image/jpeg")},
            timeout=120,
        )
    if not r.ok:
        raise RuntimeError(f"POST frames -> {r.status_code}: {r.text[:200]}")


def _upload_frame_group(meta_path: Path) -> None:
    """Upload a durable frame group over WiFi: the crop (kind 'crop') + the full
    source frame (kind 'wide'). The cloud dedups by frame_uid (a crop also sent
    over cellular collapses to one row). Delete the local copies once both are in.
    """
    base = str(meta_path)[:-5]  # strip ".json"
    crop_p, src_p = Path(base + ".crop.jpg"), Path(base + ".src.jpg")
    meta = json.loads(meta_path.read_text())
    uid, idx = meta.get("activity_uid"), meta.get("index", 0)
    shared = {k: meta.get(k) for k in
              ("activity_uid", "started_at", "captured_at", "bbox",
               "motion_score", "peak_motion")}
    if crop_p.exists():
        _post_frame(crop_p, dict(shared, kind="crop", frame_uid=f"{uid}_{idx}_crop",
                                 width=meta.get("crop_width"), height=meta.get("crop_height")))
    if src_p.exists():
        _post_frame(src_p, dict(shared, kind="wide", frame_uid=f"{uid}_{idx}_src",
                                width=meta.get("src_width"), height=meta.get("src_height")))
    # Both are in the cloud (system of record) — drop the local archive copies.
    for p in (crop_p, src_p, meta_path):
        try:
            p.unlink()
        except OSError:
            pass
    log.info("uploaded activity frame group %s_%s (crop+source)", uid, idx)


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

    log.info("uploader started — API=%s, record_dir=%s, wifi_only=%s",
             API_BASE, RECORD_DIR, WIFI_ONLY_VIDEO)

    backoff = INITIAL_BACKOFF
    holding_logged = False
    while _running:
        try:
            pending = _list_pending(RECORD_DIR)
            pending_frames = _list_pending_frames(RECORD_DIR)
        except OSError as e:
            log.error("listing recording dir failed: %s", e)
            time.sleep(POLL_SECONDS)
            continue

        if not pending and not pending_frames:
            time.sleep(POLL_SECONDS)
            backoff = INITIAL_BACKOFF
            continue

        # Hold bulk uploads (video + the durable crop/source-frame archive) off
        # cellular — only drain the backlog when WiFi is up.
        if WIFI_ONLY_VIDEO and not _wifi_connected():
            if not holding_logged:
                log.info("no WiFi — holding %d video(s) + %d frame group(s) until WiFi",
                         len(pending), len(pending_frames))
                holding_logged = True
            time.sleep(POLL_SECONDS)
            continue
        holding_logged = False

        if pending:
            log.info("found %d pending video(s)", len(pending))
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

        # Drain the durable activity-frame archive (crop + source frame) over WiFi.
        # Idempotent: a failed group is retried next loop (the cloud dedups), so we
        # just move on rather than spin on it.
        for meta_path in pending_frames:
            if not _running:
                break
            try:
                _upload_frame_group(meta_path)
                backoff = INITIAL_BACKOFF
            except Exception as e:
                log.exception("frame upload failed for %s: %s", meta_path.name, e)
                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)

    log.info("uploader stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
