"""BeeMonitor Pi -> cloud telemetry beat (cheap, cellular).

Loops at ``BEEMONITOR_TELEMETRY_INTERVAL`` (60s by default; pass ``--once`` for a
single beat) and POSTs a small health beat + one image to the API each beat.
This is the *cheap* channel —
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
# Seconds between beats. Defaults to 60 so an offline unit is noticed within a
# minute. Telemetry is JSON-only (no image) — cheap enough to send this often.
INTERVAL = int(os.environ.get("BEEMONITOR_TELEMETRY_INTERVAL", "60"))
# Trailing window for the snippets-per-period activity proxy. Decoupled from the
# beat cadence: 60s of activity is too coarse to be useful, so default to 1h.
ACTIVITY_PERIOD = int(os.environ.get("BEEMONITOR_ACTIVITY_PERIOD", "3600"))

# systemd unit names to report health for.
RECORDER_UNIT = os.environ.get("BEEMONITOR_RECORDER_UNIT", "beemonitor-recorder.service")
UPLOADER_UNIT = os.environ.get("BEEMONITOR_UPLOADER_UNIT", "beemonitor-uploader.service")
CELLULAR_UNIT = os.environ.get("BEEMONITOR_CELLULAR_UNIT", "cellular.service")
# WiFi interface used for the WiFi on/off/connect commands and state reporting.
WIFI_IFACE = os.environ.get("BEEMONITOR_WIFI_IFACE", "wlan0")
# How often (s) to poll for an on-demand command between beats — keeps the
# "Take photo" latency low without raising the health-beat cadence.
COMMAND_POLL_SECONDS = int(os.environ.get("BEEMONITOR_COMMAND_POLL_SECONDS", "8"))

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


def _run(cmd: list[str], timeout: int = 30):
    """Run a command best-effort; return the CompletedProcess or None."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("command failed %s: %s", cmd[:2], e)
        return None


def _nmcli(args: list[str], timeout: int = 30):
    """nmcli wrapper. State changes (radio/connect) need root, so prefix sudo
    when we aren't already root (the install adds a NOPASSWD sudoers rule for
    nmcli). Read-only queries work as the service user without it."""
    base = ["nmcli"] if os.geteuid() == 0 else ["sudo", "-n", "nmcli"]
    return _run(base + args, timeout=timeout)


def _wifi_state() -> dict:
    """Best-effort current WiFi status for the dashboard (radio/SSID/IP)."""
    out: dict = {}
    r = _run(["nmcli", "-t", "-f", "WIFI", "radio"])
    if r and r.returncode == 0:
        out["wifi_enabled"] = r.stdout.strip().endswith("enabled")
    r = _run(["nmcli", "-t", "-f", "DEVICE,STATE,CONNECTION", "device", "status"])
    if r and r.returncode == 0:
        for line in r.stdout.splitlines():
            parts = line.split(":")
            if len(parts) >= 3 and parts[0] == WIFI_IFACE:
                out["wifi_state"] = parts[1]
                out["wifi_ssid"] = parts[2] if parts[1] == "connected" else ""
                break
    r = _run(["nmcli", "-t", "-f", "IP4.ADDRESS", "device", "show", WIFI_IFACE])
    if r and r.returncode == 0:
        for line in r.stdout.splitlines():
            if line.startswith("IP4.ADDRESS") and ":" in line:
                out["wifi_ip"] = line.split(":", 1)[1].split("/")[0]
                break
    return out


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

    vs = _video_stats(ACTIVITY_PERIOD)
    m["videos_recorded"] = vs["videos_recorded"]
    m["pending_uploads"] = vs["pending_uploads"]
    m["bytes_pending_upload"] = vs["bytes_pending_upload"]
    # Activity proxy: snippets recorded in the trailing ACTIVITY_PERIOD window
    # (decoupled from the 60s beat — 1h by default).
    m["snippets_last_period"] = vs["snippets_last_period"]
    m["telemetry_period_seconds"] = ACTIVITY_PERIOD
    m["telemetry_period_human"] = _human_duration(ACTIVITY_PERIOD)
    if vs["newest_mtime"]:
        m["last_activity_at"] = datetime.fromtimestamp(
            vs["newest_mtime"], tz=timezone.utc).isoformat()

    m["recorder_active"] = _service_active(RECORDER_UNIT)
    m["uploader_active"] = _service_active(UPLOADER_UNIT)
    m["cellular_active"] = _service_active(CELLULAR_UNIT)

    # Current WiFi state so the dashboard can show on/off + connected network.
    m.update(_wifi_state())

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


def send_beat(image: "Path | None" = None):
    """Send one telemetry beat; return the parsed JSON response (or None on error).

    JSON-only by default (cheap over cellular). ``image`` is only passed for
    on-demand captures (picture/live-view); the regular 60s beat sends no image.
    The response may carry a pending ``command`` for the device to act on.
    """
    metrics = collect_metrics()
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
        return None

    if not r.ok:
        log.error("heartbeat -> %s: %s", r.status_code, r.text[:300])
        return None

    resp = r.json()
    log.info("heartbeat ok: %s", resp)
    if image is not None:
        _prune_queue(image)
    return resp


_running = True


def _safe_mtime(p) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _capture_now(timeout: float = 12.0):
    """Ask the recorder (via a sentinel file) for one fresh still; return its Path.

    The recorder owns the camera, so telemetry can't grab a frame directly — it
    drops ``capture.request`` in the queue, the recorder writes a JPEG there, and
    we return the newest one. None if nothing arrives in ``timeout`` s.
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    req = QUEUE_DIR / "capture.request"
    t0 = time.time()
    try:
        req.write_text(str(t0))
    except OSError:
        return None
    deadline = time.time() + timeout
    while time.time() < deadline:
        fresh = [p for p in QUEUE_DIR.glob("*.jpg") if _safe_mtime(p) >= t0 - 0.5]
        if fresh:
            return max(fresh, key=_safe_mtime)
        time.sleep(0.3)
    try:
        req.unlink()  # clear stale request so it can't fire a late capture
    except OSError:
        pass
    return None


def _capture_and_upload() -> bool:
    img = _capture_now()
    if img is None:
        log.warning("on-demand capture produced no still (is the recorder running?)")
        return False
    return send_beat(image=img) is not None


def _wifi_connect(params: dict) -> None:
    """Join a network and persist it (autoconnect) via NetworkManager."""
    ssid = (params.get("ssid") or "").strip()
    password = params.get("password") or ""
    if not ssid:
        log.warning("wifi_connect: missing ssid")
        return
    _nmcli(["radio", "wifi", "on"])  # connecting is pointless with the radio off
    args = ["device", "wifi", "connect", ssid]
    if password:
        args += ["password", password]
    args += ["ifname", WIFI_IFACE]
    r = _nmcli(args, timeout=60)
    # Never log the password — only the SSID + result.
    if r is not None and r.returncode == 0:
        log.info("wifi_connect: joined '%s'", ssid)
    else:
        log.warning("wifi_connect: failed for '%s' (rc=%s) %s", ssid,
                    getattr(r, "returncode", None),
                    (getattr(r, "stderr", "") or "").strip()[:200])


def _handle_command(cmd: str, params: dict) -> None:
    if cmd == "capture_image":
        log.info("command: capture_image")
        _capture_and_upload()
    elif cmd == "wifi_on":
        log.info("command: wifi_on")
        _nmcli(["radio", "wifi", "on"])
    elif cmd == "wifi_off":
        log.info("command: wifi_off")
        _nmcli(["radio", "wifi", "off"])
    elif cmd == "wifi_connect":
        log.info("command: wifi_connect ssid=%s", (params.get("ssid") or "").strip())
        _wifi_connect(params)
    elif cmd == "wifi_forget":
        ssid = (params.get("ssid") or "").strip()
        log.info("command: wifi_forget ssid=%s", ssid)
        if ssid:
            _nmcli(["connection", "delete", ssid])
    else:
        log.warning("ignoring unknown command: %s", cmd)


def _poll_command() -> None:
    """Lightweight check for a pending on-demand command between beats.

    Cheap GET (no metrics/image), so we can poll it every few seconds — a
    requested photo arrives in ~COMMAND_POLL_SECONDS instead of waiting for the
    next 60s beat.
    """
    try:
        r = requests.get(
            urljoin(API_BASE + "/", "api/v1/devices/command"),
            headers={"Authorization": f"Bearer {DEVICE_KEY}"},
            timeout=30,
        )
        if r.ok:
            d = r.json()
            if d.get("command"):
                _handle_command(d["command"], d.get("params") or {})
    except requests.RequestException:
        pass


def _handle_signal(signum, frame):  # noqa: ARG001
    global _running
    _running = False


def main() -> int:
    _validate_config()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    # --once: single beat (handy for cron/manual test). Default: loop forever
    # at BEEMONITOR_TELEMETRY_INTERVAL (60 default; raise via env to slow down).
    if "--once" in sys.argv:
        send_beat()
        return 0

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    log.info("telemetry loop started — interval=%ds, queue=%s", INTERVAL, QUEUE_DIR)

    while _running:
        try:
            resp = send_beat()
            # The beat may also carry a command (belt-and-suspenders).
            if resp and resp.get("command"):
                _handle_command(resp["command"], resp.get("params") or {})
        except Exception as e:  # pragma: no cover - never let the loop die
            log.exception("beat raised: %s", e)
        # Between beats, poll for on-demand commands every COMMAND_POLL_SECONDS so
        # a requested photo lands in seconds, not up to a full INTERVAL.
        slept = 0
        while _running and slept < INTERVAL:
            step = min(COMMAND_POLL_SECONDS, INTERVAL - slept)
            time.sleep(step)
            slept += step
            if _running and slept < INTERVAL:
                _poll_command()
    log.info("telemetry stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
