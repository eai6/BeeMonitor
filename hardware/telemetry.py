"""BeeMonitor Pi -> cloud telemetry beat (cheap, cellular).

Runs once per invocation (driven hourly by ``beemonitor-telemetry.timer``) and
POSTs a small health beat + one image to the API. This is the *cheap* channel —
telemetry JSON + a ~250 KB JPEG — that tells the dashboard the unit is alive,
kept separate from the WiFi-gated bulk-video upload (``uploader.py``).

See design: ``memory/10_cellular_telemetry_design.md``.

Flow:
    1. Collect health metrics (storage, uptime, CPU temp, service health,
       cellular signal, video counts).
    2. Attach the latest image the recorder dropped in the telemetry queue.
    3. POST multipart to ``/api/v1/devices/heartbeat`` with the device key.
    4. On success, delete the sent image (and prune older queued ones).

It POSTs even if the recorder is dead — a missing image plus
``recorder_active=false`` is itself the alert.

Config — ``/etc/beemonitor/uploader.env`` (shared with the uploader):
    BEEMONITOR_API_BASE        = https://...
    BEEMONITOR_DEVICE_KEY      = bmk_device_...
    BEEMONITOR_RECORD_DIR      = /home/beemonitor/Desktop/cameraOutput/beeHotel
    BEEMONITOR_TELEMETRY_QUEUE = <RECORD_DIR>/../telemetry   (default)
    BEEMONITOR_SCHEDULE_WINDOW = "07:50-18:45"               (optional, WittyPi)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

try:
    import requests
except ImportError:  # pragma: no cover
    print("telemetry requires 'requests' — pip install requests", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("BEEMONITOR_API_BASE", "").rstrip("/")
DEVICE_KEY = os.environ.get("BEEMONITOR_DEVICE_KEY", "")
RECORD_DIR = Path(os.environ.get(
    "BEEMONITOR_RECORD_DIR", "/home/beemonitor/Desktop/cameraOutput/beeHotel"))
QUEUE_DIR = Path(os.environ.get(
    "BEEMONITOR_TELEMETRY_QUEUE", str(RECORD_DIR.parent / "telemetry")))
SCHEDULE_WINDOW = os.environ.get("BEEMONITOR_SCHEDULE_WINDOW", "")
POST_TIMEOUT = int(os.environ.get("BEEMONITOR_TELEMETRY_TIMEOUT", "120"))
# Seconds between beats. 3600 in production; set 60 for testing. Also the
# trailing window used for the snippets-per-period activity proxy.
INTERVAL = int(os.environ.get("BEEMONITOR_TELEMETRY_INTERVAL", "3600"))

# systemd unit names to report health for.
RECORDER_UNIT = os.environ.get("BEEMONITOR_RECORDER_UNIT", "beemonitor-recorder.service")
UPLOADER_UNIT = os.environ.get("BEEMONITOR_UPLOADER_UNIT", "beemonitor-uploader.service")
CELLULAR_UNIT = os.environ.get("BEEMONITOR_CELLULAR_UNIT", "cellular.service")
QMI_DEV = os.environ.get("BEEMONITOR_QMI_DEV", "/dev/cdc-wdm0")

logging.basicConfig(
    format="%(asctime)s %(levelname)s telemetry %(message)s", level=logging.INFO,
)
log = logging.getLogger("telemetry")


# ---------------------------------------------------------------------------
# Metric collection (all best-effort — a single failure must not drop the beat)
# ---------------------------------------------------------------------------

def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _human_duration(seconds: float) -> str:
    s = int(seconds)
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, _ = divmod(s, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def _uptime_seconds():
    try:
        with open("/proc/uptime") as fh:
            return float(fh.read().split()[0])
    except (OSError, ValueError):
        return None


def _cpu_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as fh:
            return round(int(fh.read().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return None


def _service_active(unit: str) -> bool:
    try:
        return subprocess.run(
            ["systemctl", "is-active", "--quiet", unit], timeout=10,
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _cellular_signal():
    """Best-effort RSSI via qmicli. Returns a string like '-71 dBm' or None."""
    try:
        out = subprocess.run(
            ["qmicli", "-p", "-d", QMI_DEV, "--nas-get-signal-strength"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        if "dBm" in line and ("RSSI" in line or "Network" in line):
            return line.split(":", 1)[-1].strip()
    return None


def _video_stats(window_seconds: int) -> dict:
    """Single pass over the recordings dir.

    Returns total snippet count, pending (un-uploaded) count + bytes, the count
    recorded within the trailing ``window_seconds`` (the activity proxy — a
    snippet only exists because motion was detected), and the newest snippet's
    timestamp.
    """
    now = time.time()
    total = pending = pending_bytes = recent = 0
    newest = 0.0
    if RECORD_DIR.is_dir():
        for mp4 in RECORD_DIR.rglob("*.mp4"):
            total += 1
            try:
                st = mp4.stat()
            except OSError:
                continue
            if st.st_mtime > newest:
                newest = st.st_mtime
            if st.st_mtime >= now - window_seconds:
                recent += 1
            if not mp4.with_suffix(mp4.suffix + ".uploaded").exists():
                pending += 1
                pending_bytes += st.st_size
    return {
        "videos_recorded": total,
        "pending_uploads": pending,
        "bytes_pending_upload": pending_bytes,
        "snippets_last_period": recent,
        "newest_mtime": newest,
    }


def collect_metrics() -> dict:
    m: dict = {}

    if RECORD_DIR.is_dir():
        try:
            usage = shutil.disk_usage(RECORD_DIR)
            m["storage_pct"] = round(usage.used / usage.total * 100, 1)
            m["storage_free_human"] = _human_bytes(usage.free)
            m["storage_used_bytes"] = usage.used
            m["storage_free_bytes"] = usage.free
        except OSError:
            pass

    up = _uptime_seconds()
    if up is not None:
        m["uptime_seconds"] = int(up)
        m["uptime_human"] = _human_duration(up)

    temp = _cpu_temp_c()
    if temp is not None:
        m["cpu_temp_c"] = temp

    vs = _video_stats(INTERVAL)
    m["videos_recorded"] = vs["videos_recorded"]
    m["pending_uploads"] = vs["pending_uploads"]
    m["bytes_pending_upload"] = vs["bytes_pending_upload"]
    # Activity proxy: snippets recorded in the trailing telemetry window.
    m["snippets_last_period"] = vs["snippets_last_period"]
    m["telemetry_period_seconds"] = INTERVAL
    m["telemetry_period_human"] = _human_duration(INTERVAL)
    if vs["newest_mtime"]:
        m["last_activity_at"] = datetime.fromtimestamp(
            vs["newest_mtime"], tz=timezone.utc).isoformat()

    m["recorder_active"] = _service_active(RECORDER_UNIT)
    m["uploader_active"] = _service_active(UPLOADER_UNIT)
    m["cellular_active"] = _service_active(CELLULAR_UNIT)

    sig = _cellular_signal()
    if sig:
        m["cellular_signal"] = sig
    if SCHEDULE_WINDOW:
        m["schedule_window"] = SCHEDULE_WINDOW

    return m


# ---------------------------------------------------------------------------
# Image queue
# ---------------------------------------------------------------------------

def _latest_image() -> Path | None:
    if not QUEUE_DIR.is_dir():
        return None
    imgs = sorted(QUEUE_DIR.glob("*.jpg"), key=lambda p: p.stat().st_mtime)
    return imgs[-1] if imgs else None


def _prune_queue(up_to: Path) -> None:
    """Delete the sent image and any older queued ones (keep nothing stale)."""
    try:
        cutoff = up_to.stat().st_mtime
    except OSError:
        cutoff = time.time()
    for img in QUEUE_DIR.glob("*.jpg"):
        try:
            if img.stat().st_mtime <= cutoff:
                img.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------

def _validate_config() -> None:
    if not API_BASE or not DEVICE_KEY:
        log.error("missing BEEMONITOR_API_BASE / BEEMONITOR_DEVICE_KEY")
        sys.exit(2)
    if not DEVICE_KEY.startswith("bmk_device_"):
        log.error("BEEMONITOR_DEVICE_KEY must start with 'bmk_device_'")
        sys.exit(2)


def send_beat() -> int:
    metrics = collect_metrics()
    image = _latest_image()
    url = urljoin(API_BASE + "/", "api/v1/devices/heartbeat")
    headers = {"Authorization": f"Bearer {DEVICE_KEY}"}
    data = {"metrics": json.dumps(metrics)}

    log.info("beat: storage=%s%% videos=%s pending=%s snippets/%s=%s rec=%s up=%s cell=%s image=%s",
             metrics.get("storage_pct"), metrics.get("videos_recorded"),
             metrics.get("pending_uploads"), metrics.get("telemetry_period_human"),
             metrics.get("snippets_last_period"), metrics.get("recorder_active"),
             metrics.get("uploader_active"), metrics.get("cellular_active"),
             image.name if image else None)

    try:
        if image is not None:
            with open(image, "rb") as fh:
                r = requests.post(
                    url, headers=headers, data=data,
                    files={"image": (image.name, fh, "image/jpeg")},
                    timeout=POST_TIMEOUT,
                )
        else:
            r = requests.post(url, headers=headers, data=data, timeout=POST_TIMEOUT)
    except requests.RequestException as e:
        log.error("heartbeat POST failed: %s", e)
        return 1

    if not r.ok:
        log.error("heartbeat -> %s: %s", r.status_code, r.text[:300])
        return 1

    log.info("heartbeat ok: %s", r.json())
    if image is not None:
        _prune_queue(image)
    return 0


_running = True


def _handle_signal(signum, frame):  # noqa: ARG001
    global _running
    _running = False


def main() -> int:
    _validate_config()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    # --once: single beat (handy for cron/manual test). Default: loop forever
    # at BEEMONITOR_TELEMETRY_INTERVAL (3600 prod, 60 for testing).
    if "--once" in sys.argv:
        return send_beat()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    log.info("telemetry loop started — interval=%ds, queue=%s", INTERVAL, QUEUE_DIR)

    while _running:
        try:
            send_beat()
        except Exception as e:  # pragma: no cover - never let the loop die
            log.exception("beat raised: %s", e)
        # Interruptible sleep so SIGTERM stops us promptly.
        slept = 0
        while _running and slept < INTERVAL:
            step = min(5, INTERVAL - slept)
            time.sleep(step)
            slept += step
    log.info("telemetry stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
