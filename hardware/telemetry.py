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
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
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
# Cellular interface (for transport reporting).
CELL_IFACE = os.environ.get("BEEMONITOR_CELL_IFACE", "wwan0")
# When WiFi is up, pin its default-route metric below cellular's (cellular-up.sh
# uses 700) so ALL traffic — incl. telemetry — rides WiFi, with cellular as the
# fallback. Saves metered mobile data whenever WiFi is connected.
WIFI_ROUTE_METRIC = int(os.environ.get("BEEMONITOR_WIFI_ROUTE_METRIC", "100"))
# Lowest beat interval the dashboard may set (seconds). Floored at 5s. WiFi can
# usually sustain 5s; over cellular a beat's own work (nmcli/systemctl + the
# POST) often takes longer, so the loop just beats as fast as it can — it never
# beats faster than this, but may be slower on a slow link.
MIN_INTERVAL = int(os.environ.get("BEEMONITOR_MIN_INTERVAL", "5"))
# How often (s) to poll for an on-demand command between beats — keeps the
# "Take photo" latency low without raising the health-beat cadence.
COMMAND_POLL_SECONDS = int(os.environ.get("BEEMONITOR_COMMAND_POLL_SECONDS", "8"))
# Slow command-poll floor for CELLULAR (the WiFi poll runs at COMMAND_POLL_SECONDS).
# 0 = beat-only on cellular. A small value (default 120s) makes dashboard commands
# land within ~2 min on cellular — and keeps last_seen fresh so the unit stays
# "online" — without cranking the heavier full beat. ~720 lightweight GETs/day.
CELL_COMMAND_POLL_SECONDS = int(os.environ.get("BEEMONITOR_CELL_COMMAND_POLL_SECONDS", "120"))

# Remote software update (the "update" command). REPO_DIR is the git checkout this
# file lives in; the updater and the status file it writes live there/below.
REPO_DIR = Path(__file__).resolve().parent.parent
UPDATE_SCRIPT = REPO_DIR / "hardware" / "update.sh"
USB_SCRIPT = REPO_DIR / "hardware" / "usb-transfer.sh"
# Cellular egress firewall. Normally gates wwan0 to telemetry-only; the dashboard
# can open it briefly for remote debugging (rpi-connect), which auto-reverts.
FIREWALL_SCRIPT = REPO_DIR / "hardware" / "cellular" / "cellular-firewall.sh"
# State marker the firewall script writes ("gated"/"open"), reported in the beat.
CELL_FIREWALL_STATE = Path(os.environ.get(
    "BEEMONITOR_CELL_STATE", "/run/beemonitor-cell-firewall"))
# Default auto-revert window when cellular is opened for debugging (minutes).
CELL_DEBUG_DEFAULT_MIN = int(os.environ.get("BEEMONITOR_CELL_DEBUG_MINUTES", "30"))
STATE_DIR = Path(os.environ.get("BEEMONITOR_STATE_DIR", "/home/beemonitor/.beemonitor"))
UPDATE_STATUS_FILE = STATE_DIR / "update-status.json"
# Last USB-transfer result (written by usb-transfer.sh), reported in the beat so
# the dashboard can show "copied N" / "no USB detected".
USB_STATUS_FILE = STATE_DIR / "usb-status.json"
# Device's dashboard location, cached from the heartbeat so usb-transfer.sh can
# name the USB folder after the hive (one stick, several hives, clear folders).
LOCATION_FILE = STATE_DIR / "location"
# Dashboard motion-tuning overrides; the recorder applies these on top of
# calibration.json (main_motion.py). Lives beside calibration.json.
MOTION_TUNING_FILE = RECORD_DIR.parent / "motion_tuning.json"
# Dashboard ROI editor outputs (normalized): hotel ROI override + nest layout.
ROI_OVERRIDE_FILE = RECORD_DIR.parent / "roi_override.json"
NEST_LAYOUT_FILE = RECORD_DIR.parent / "nest_layout.json"
# Dashboard-pushed bee-confirmation mode (off|tag|gate); the recorder hot-reloads
# it over its env default. Lets a no-shell unit be switched between observe (tag)
# and filter (gate) remotely.
BEE_CONFIRM_MODE_FILE = RECORD_DIR.parent / "bee_confirm_mode.json"
# Dashboard-pushed crop-upload toggle the recorder hot-reloads ({"enabled": bool}).
# Off = stop sampling/sending BioCLIP review crops over cellular (the recorder
# stops sampling, so no SD/CPU/cellular spend). Same dir as the mode file above.
ACTIVITY_FRAMES_FILE = RECORD_DIR.parent / "activity_frames.json"

# Activity-frame upload (taxonomic monitoring). The recorder queues mover crops
# here; we ship them over cellular under a daily cap. See main_motion.py +
# memory/15_monitoring_agent_design.md.
ACTIVITY_FRAMES_QUEUE = Path(os.environ.get(
    "BEEMONITOR_ACTIVITY_FRAMES_QUEUE", str(RECORD_DIR.parent / "activity_frames")))
# Max frames to send per UTC day (cellular budget guard). 0 disables uploading.
FRAME_DAILY_CAP = int(os.environ.get("BEEMONITOR_FRAME_DAILY_CAP", "60"))
# How many queued frames to ship per beat so a backlog drains gradually.
FRAME_BATCH_PER_BEAT = int(os.environ.get("BEEMONITOR_FRAME_BATCH_PER_BEAT", "6"))
# Hard cap on crops queued on disk. Frames merely over the daily cap wait for the
# next UTC day's budget; only when the queue exceeds THIS do we drop the oldest
# (a unit offline for days, or a hive far over cap) so disk can't fill.
FRAME_QUEUE_MAX = int(os.environ.get("BEEMONITOR_FRAME_QUEUE_MAX", "1000"))
FRAME_SENT_FILE = STATE_DIR / "frames-sent.json"  # {"date": "YYYY-MM-DD", "count": N}
# Dashboard-pushed override of the daily cap (so a no-shell unit can be throttled
# — or set to 0 to stop crop upload entirely — without editing uploader.env). The
# heartbeat may carry ``frame_daily_cap``; when present it wins over the env value.
FRAME_CAP_FILE = STATE_DIR / "frame-cap.json"  # {"cap": N}

# Storage cleanup: delete local clips a human cleared for deletion on the
# dashboard (GET/POST /api/v1/devices/cleanup). Cheap JSON, runs over cellular.
# Seconds between cleanup passes; 0 disables.
CLEANUP_INTERVAL = int(os.environ.get("BEEMONITOR_CLEANUP_INTERVAL", "600"))

# --- Remote WittyPi power schedule (Phase 2) ---------------------------------
# Where the UUGear WittyPi software lives (utilities.sh / runScript.sh / schedules).
# First existing path wins. See memory/16_remote_scheduling_design.md.
WITTYPI_DIRS = [Path(p) for p in os.environ.get(
    "BEEMONITOR_WITTYPI_DIR",
    "/home/beemonitor/wittypi:/home/beemonitor/Desktop/wittypi:/home/pi/wittypi"
).split(":") if p]
# Where we record the schedule we last successfully applied, so the beat can
# report active_schedule for the dashboard's desired-vs-active reconcile.
WAKE_SCHEDULE_STATE = STATE_DIR / "wake-schedule.json"
# SAFETY FLOOR: the device never lets a schedule keep the unit OFF longer than
# this (hours). A spec whose off-stretch exceeds it is rejected, so a fat-fingered
# schedule can't strand a unit for a whole day+. (memory/16 §5.)
WAKE_FLOOR_HOURS = float(os.environ.get("BEEMONITOR_WAKE_FLOOR_HOURS", "24"))
# Phase 2 is APPLY-GATED. Default OFF: the device only READS the WittyPi and
# reports next boot/shutdown (real dashboard reconcile, zero stranding risk). It
# does NOT change the WittyPi schedule until this is set, after the read path is
# verified on real hardware. Enable with BEEMONITOR_WAKE_SCHEDULE_APPLY=1.
WAKE_SCHEDULE_APPLY = os.environ.get(
    "BEEMONITOR_WAKE_SCHEDULE_APPLY", "0") in ("1", "true", "True")

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


def _mac_address() -> dict:
    """The unit's primary network MAC (stable per board) for identification on the
    dashboard — wlan0's, falling back to eth0's. Read straight from sysfs, so it
    needs no root and no extra tools, and works whether or not WiFi is connected."""
    out: dict = {}
    for iface in (WIFI_IFACE, "eth0"):
        try:
            mac = Path("/sys/class/net", iface, "address").read_text().strip()
        except OSError:
            continue
        if mac and mac != "00:00:00:00:00:00":
            out["mac"], out["mac_iface"] = mac, iface
            break
    return out


def _active_transport() -> str:
    """Which interface telemetry is actually leaving on right now.

    Reads the kernel's chosen route to the internet, so it reflects the live
    WiFi-vs-cellular decision the dashboard wants to see.
    """
    r = _run(["ip", "route", "get", "1.1.1.1"])
    if r and r.returncode == 0 and "dev" in r.stdout:
        parts = r.stdout.split()
        try:
            dev = parts[parts.index("dev") + 1]
        except (ValueError, IndexError):
            return ""
        if dev == WIFI_IFACE:
            return "wifi"
        if dev == CELL_IFACE or dev.startswith("wwan") or dev.startswith("ppp"):
            return "cellular"
        return dev
    return ""


def _prefer_wifi_route() -> None:
    """Ensure the active WiFi connection outranks cellular as the default route.

    cellular-up.sh sets wwan0's metric to 700; we pin the live WiFi connection's
    ipv4.route-metric to WIFI_ROUTE_METRIC (100 by default) so WiFi always wins
    while it's up. nmcli applies route-metric on (re)activation, so we modify +
    bring the connection up only when its metric is wrong (a brief, rare blip).
    """
    # Find the connection name currently active on the WiFi interface.
    r = _nmcli(["-t", "-f", "DEVICE,STATE,CONNECTION", "device", "status"])
    if not (r and r.returncode == 0):
        return
    conn = ""
    for line in r.stdout.splitlines():
        parts = line.split(":")
        if len(parts) >= 3 and parts[0] == WIFI_IFACE and parts[1] == "connected":
            conn = parts[2]
            break
    if not conn:
        return  # WiFi not connected — nothing to prefer; cellular carries it.
    # Already at the desired metric? Then don't disrupt the live connection.
    cur = _nmcli(["-t", "-f", "ipv4.route-metric", "connection", "show", conn])
    if cur and cur.returncode == 0 and cur.stdout.strip().endswith(str(WIFI_ROUTE_METRIC)):
        return
    log.info("prefer-wifi: pinning '%s' route-metric=%d (was %s)", conn,
             WIFI_ROUTE_METRIC, (cur.stdout.strip() if cur else "?"))
    _nmcli(["connection", "modify", conn, "ipv4.route-metric", str(WIFI_ROUTE_METRIC)])
    _nmcli(["connection", "up", conn], timeout=45)


# Recording time from the clip filename: site_YYYY-MM-DD_HH_MM_SS.mp4 (the
# HH-MM-SS separators may be '_' or '-'). The hour is all we need to bucket.
_CLIP_RE = re.compile(r"_(\d{4})-(\d{2})-(\d{2})_(\d{2})[_\-.]")


def _clip_hour_key(name: str, mtime: float) -> str:
    """Local-hour key 'YYYY-MM-DDTHH' for a clip, from its FILENAME (stable
    across restarts/mtime changes). Falls back to mtime if the name doesn't
    parse."""
    m = _CLIP_RE.search(name)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}"
    return time.strftime("%Y-%m-%dT%H", time.localtime(mtime))


def _video_stats(window_seconds: int) -> dict:
    """Single pass over the recordings dir.

    Returns total snippet count, pending (un-uploaded) count + bytes, the count
    recorded within the trailing ``window_seconds`` (the activity proxy — a
    snippet only exists because motion was detected), and the newest snippet's
    timestamp.
    """
    now = time.time()
    total = pending = pending_bytes = recent = 0
    recordings_bytes = 0
    usb_transferred = 0
    newest = 0.0
    # Per-LOCAL-hour histogram of clips on the card, so the dashboard can show
    # activity the moment a clip is RECORDED — no waiting for the (WiFi-gated)
    # cloud upload. Keyed by the recording hour parsed from the FILENAME (the Pi
    # names clips site_YYYY-MM-DD_HH_MM_SS.mp4), so the count is a pure function
    # of the files on disk — unaffected by mtime changes or service restarts.
    # Bounded to ~8 days to cover the dashboard's 7d view without a huge payload.
    by_hour: dict = {}
    hist_cutoff = now - 8 * 86400
    cur_key = time.strftime("%Y-%m-%dT%H", time.localtime(now))
    if RECORD_DIR.is_dir():
        for mp4 in RECORD_DIR.rglob("*.mp4"):
            total += 1
            try:
                st = mp4.stat()
            except OSError:
                continue
            recordings_bytes += st.st_size  # footprint of ALL clips on the card
            if st.st_mtime > newest:
                newest = st.st_mtime
            # Bee confirmation (gate mode): a clip the on-device YOLO couldn't
            # confirm is marked `<clip>.mp4.unconfirmed`. It still records +
            # uploads (counted in total/pending/bytes), but is NOT counted as a
            # bee ACTIVITY — so the dashboard/telemetry activity proxy isn't
            # inflated by shadows. Untagged clips (off mode / confirmed) count.
            is_activity = not mp4.with_suffix(mp4.suffix + ".unconfirmed").exists()
            if is_activity and st.st_mtime >= now - window_seconds:
                recent += 1
            if is_activity and st.st_mtime >= hist_cutoff:
                key = _clip_hour_key(mp4.name, st.st_mtime)
                by_hour[key] = by_hour.get(key, 0) + 1
            if not mp4.with_suffix(mp4.suffix + ".uploaded").exists():
                pending += 1
                pending_bytes += st.st_size
            if mp4.with_suffix(mp4.suffix + ".usb").exists():
                usb_transferred += 1  # copied to a field USB (see usb-transfer.sh)
    return {
        "videos_recorded": total,
        "pending_uploads": pending,
        "bytes_pending_upload": pending_bytes,
        "recordings_bytes": recordings_bytes,
        "usb_transferred": usb_transferred,
        "snippets_last_period": recent,
        "newest_mtime": newest,
        "activity_by_hour": by_hour,
        "clips_this_hour": by_hour.get(cur_key, 0),
    }


def _git_commit() -> str:
    """Deployed-code version for the dashboard: the edge-artifact manifest version
    on a release (symlink) layout, else the short git SHA on a git-clone layout."""
    try:
        v = json.loads((REPO_DIR / ".edge-manifest.json").read_text()).get("version")
        if v:
            return str(v)
    except (OSError, ValueError):
        pass
    r = _run(["git", "-C", str(REPO_DIR), "rev-parse", "--short", "HEAD"])
    return r.stdout.strip() if r and r.returncode == 0 else ""


def _update_status() -> dict | None:
    """Last remote-update result (written by update.sh), reported in the beat."""
    try:
        return json.loads(UPDATE_STATUS_FILE.read_text())
    except (OSError, ValueError):
        return None


def _usb_status() -> dict | None:
    """Last USB-transfer result (written by usb-transfer.sh)."""
    try:
        return json.loads(USB_STATUS_FILE.read_text())
    except (OSError, ValueError):
        return None


def _usb_present() -> bool:
    """Is a USB drive plugged into the Pi right now? Detected by transport
    (tran==usb disk), excluding the SD card / OS-root disk — same signal
    usb-transfer.sh uses. Drives the dashboard's 'USB connected' indicator."""
    r = _run(["lsblk", "-rno", "NAME,TYPE,TRAN"])
    if not (r and r.returncode == 0):
        return False
    root_disk = ""
    rs = _run(["findmnt", "-no", "SOURCE", "/"])
    if rs and rs.returncode == 0 and rs.stdout.strip():
        pk = _run(["lsblk", "-no", "PKNAME", rs.stdout.strip()])
        if pk and pk.returncode == 0 and pk.stdout.strip():
            root_disk = pk.stdout.strip().splitlines()[0]
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[1] == "disk" and parts[2] == "usb" \
                and parts[0] != root_disk:
            return True
    return False


# --- WittyPi power: battery telemetry + schedule (read side) -----------------
# All of this goes through the WittyPi's own utilities.sh functions rather than
# raw I2C registers, so it stays correct across WittyPi firmware/board variants
# (the register map differs between WittyPi 3/4/L3V7). Every call is best-effort:
# no WittyPi, no software, or a missing sudo rule -> the value is simply omitted
# from the beat. Reads are safe to run every beat; nothing here changes state.
#
# NOTE: the utilities.sh function NAMES (get_input_voltage / get_output_current /
# get_power_mode / get_startup_time / get_shutdown_time) are the UUGear standard
# but should be confirmed once on the real unit (rpi-connect shell).


@lru_cache(maxsize=1)
def _wittypi_dir() -> "Path | None":
    """The installed WittyPi software directory (has utilities.sh), or None."""
    for d in WITTYPI_DIRS:
        try:
            if (d / "utilities.sh").is_file():
                return d
        except OSError:
            continue
    return None


def _wittypi_sh(snippet: str, timeout: int = 8):
    """Run a shell snippet with WittyPi's utilities.sh sourced (its functions talk
    to the board over i2c).

    The i2c reads/writes work as any user in the ``i2c`` group — no root needed —
    so we try the call DIRECTLY first. Only if that fails do we retry under
    ``sudo -n`` (for setups that gate i2c or the scripts behind root, which needs
    a NOPASSWD rule). None if there's no WittyPi or both attempts fail."""
    d = _wittypi_dir()
    if d is None:
        return None
    cmd = f"cd '{d}' && . ./utilities.sh >/dev/null 2>&1; {snippet}"
    r = _run(["bash", "-c", cmd], timeout=timeout)
    if r is not None and r.returncode == 0:
        return r
    if os.geteuid() != 0:
        r2 = _run(["sudo", "-n", "bash", "-c", cmd], timeout=timeout)
        if r2 is not None and r2.returncode == 0:
            return r2
    return r  # surface the direct attempt's result (incl. failures) for logging


def _wittypi_value(snippet: str) -> "str | None":
    """stdout of a WittyPi utilities call, stripped; None unless it succeeded."""
    r = _wittypi_sh(snippet)
    if r is None or r.returncode != 0:
        return None
    return r.stdout.strip()


def _to_float(s) -> "float | None":
    try:
        return float(str(s).strip())
    except (TypeError, ValueError):
        return None


def _wittypi_power() -> dict:
    """Battery/input voltage + load current + power source for the beat. {} if no
    WittyPi. This is the missing outage diagnostic — battery_voltage trending down
    before a no-wake points at power, not a board fault. Disable with
    BEEMONITOR_WITTYPI_BATTERY=0."""
    if os.environ.get("BEEMONITOR_WITTYPI_BATTERY", "1") not in ("1", "true", "True"):
        return {}
    vin = _to_float(_wittypi_value("get_input_voltage"))
    if vin is None or vin <= 0:
        return {}  # no WittyPi / unreadable -> omit power telemetry entirely
    out: dict = {"battery_voltage": round(vin, 2)}
    iout = _to_float(_wittypi_value("get_output_current"))
    if iout is not None:
        out["output_current"] = round(iout, 2)
    mode = _wittypi_value("get_power_mode")
    if mode is not None:
        out["power_source"] = "DC input" if mode.strip() == "1" else "USB 5V"
    return out


def _alarm_is_set(s: str) -> bool:
    """A WittyPi alarm string with any non-zero digit is a real alarm; the unset
    sentinel is all zeros (e.g. '0 00:00:00')."""
    return bool(s) and any(c in "123456789" for c in s)


def _wittypi_alarms() -> dict:
    """The WittyPi's currently-programmed next startup/shutdown, as display
    strings — the 'when is this unit next on/off?' the dashboard was missing.
    Read-only."""
    out: dict = {}
    boot = _wittypi_value("get_startup_time")
    if boot and _alarm_is_set(boot):
        out["next_boot"] = boot
    shut = _wittypi_value("get_shutdown_time")
    if shut and _alarm_is_set(shut):
        out["next_shutdown"] = shut
    return out


def _read_applied_spec() -> "dict | None":
    """The wake-schedule spec the device last successfully applied, or None."""
    try:
        spec = json.loads(WAKE_SCHEDULE_STATE.read_text())
        return spec if isinstance(spec, dict) else None
    except (OSError, ValueError):
        return None


def _schedule_metrics() -> dict:
    """Power-schedule telemetry for the beat: the WittyPi's real next boot/shutdown
    plus active_schedule (what the device last applied) for the dashboard's
    desired-vs-active reconcile. All read-only."""
    out = _wittypi_alarms()
    applied = _read_applied_spec()
    if applied:
        out["active_schedule"] = applied
    return out


_wpi_diag_done = False


def _wittypi_diag_once() -> None:
    """Log, once, why WittyPi power telemetry is or isn't working — so a blank
    Power card is diagnosable from `journalctl -u beemonitor-telemetry | grep WittyPi`
    without guessing (dir not found vs. read failed vs. function missing)."""
    global _wpi_diag_done
    if _wpi_diag_done:
        return
    _wpi_diag_done = True
    d = _wittypi_dir()
    if d is None:
        log.warning("WittyPi: no software dir in %s — power telemetry off; set "
                    "BEEMONITOR_WITTYPI_DIR to the install path",
                    [str(x) for x in WITTYPI_DIRS])
        return
    r = _wittypi_sh("get_input_voltage")
    log.info("WittyPi diag: dir=%s get_input_voltage rc=%s out=%r err=%r", d,
             (r.returncode if r else None),
             (r.stdout.strip() if r else None),
             ((r.stderr or "").strip()[:160] if r else None))


def _timezone_info() -> dict:
    """The device's local timezone (IANA name + current UTC offset). Lets the
    dashboard bucket activity by the hive's local clock-hour wherever it's
    deployed. Best-effort — both fields are optional."""
    out: dict = {}
    try:
        out["tz"] = Path("/etc/timezone").read_text().strip()
    except OSError:
        pass
    try:
        off = datetime.now().astimezone().utcoffset()
        if off is not None:
            out["tz_offset_min"] = int(off.total_seconds() // 60)
    except (OSError, ValueError):
        pass
    return out


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
    # Footprint of the recorded clips themselves (responds to device cleanup,
    # unlike whole-card storage_pct).
    m["recordings_bytes"] = vs["recordings_bytes"]
    m["recordings_human"] = _human_bytes(vs["recordings_bytes"])
    # Clips copied to a field USB (usb-transfer.sh writes a .usb sidecar).
    m["usb_transferred"] = vs["usb_transferred"]
    # Activity proxy: snippets recorded in the trailing ACTIVITY_PERIOD window
    # (decoupled from the 60s beat — 1h by default).
    m["snippets_last_period"] = vs["snippets_last_period"]
    # Creation-based activity — clips ON THE CARD per local hour, so the dashboard
    # updates the moment a clip is recorded, not when it's uploaded.
    m["clips_this_hour"] = vs["clips_this_hour"]
    m["activity_by_hour"] = vs["activity_by_hour"]
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
    # Primary network MAC (stable per board) for identification on the dashboard.
    m.update(_mac_address())
    # Which link this beat is actually leaving on (wifi when connected, else
    # cellular) — lets the dashboard confirm telemetry rode WiFi.
    transport = _active_transport()
    if transport:
        m["active_transport"] = transport

    if SCHEDULE_WINDOW:
        m["schedule_window"] = SCHEDULE_WINDOW

    # Cellular egress gate state ("gated"/"open") so the dashboard shows whether
    # debug access is currently open. Absent file = gated (the default at boot).
    try:
        state = CELL_FIREWALL_STATE.read_text().strip()
        if state:
            m["cell_firewall"] = state
    except OSError:
        pass

    # Deployed code version + last remote-update result, for the dashboard.
    commit = _git_commit()
    if commit:
        m["code_commit"] = commit
    upd = _update_status()
    if upd:
        m["update"] = upd
    usb = _usb_status()
    if usb:
        m["usb"] = usb
    # Reap a finished background copy (so it's not left a zombie) and report
    # whether a USB is currently connected.
    if _usb_proc is not None:
        _usb_proc.poll()
    m["usb_present"] = _usb_present()
    m.update(_timezone_info())
    # WittyPi battery/input voltage + load (best-effort; omitted if no WittyPi).
    _wittypi_diag_once()  # one-time log of why power telemetry is/ isn't working
    m.update(_wittypi_power())
    # WittyPi next boot/shutdown + the schedule the device last applied, for the
    # dashboard's power-schedule reconcile (read-only).
    m.update(_schedule_metrics())

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
# Activity-frame upload (mover crops -> /api/v1/devices/frames, daily-capped)
# ---------------------------------------------------------------------------

def _frames_sent_today() -> int:
    """Frames already sent this UTC day (resets at midnight UTC)."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        d = json.loads(FRAME_SENT_FILE.read_text())
        if d.get("date") == today:
            return int(d.get("count") or 0)
    except (OSError, ValueError, TypeError):
        pass
    return 0


def _bump_frames_sent(n: int) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = _frames_sent_today() + n
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        FRAME_SENT_FILE.write_text(json.dumps({"date": today, "count": count}))
    except OSError:
        pass


def _drop_frame(jpg: Path, meta_path: Path) -> None:
    for p in (jpg, meta_path):
        try:
            p.unlink()
        except OSError:
            pass


def _send_one_frame(jpg: Path, meta_raw: str) -> "bool | None":
    """POST one activity frame. True = accepted; False = transient (retry later);
    None = permanent reject (drop it)."""
    url = urljoin(API_BASE + "/", "api/v1/devices/frames")
    headers = {"Authorization": f"Bearer {DEVICE_KEY}"}
    try:
        with open(jpg, "rb") as fh:
            r = requests.post(
                url, headers=headers, data={"meta": meta_raw},
                files={"frame": (jpg.name, fh, "image/jpeg")},
                timeout=POST_TIMEOUT,
            )
    except requests.RequestException as e:
        log.warning("frame POST failed (%s): %s", jpg.name, e)
        return False
    if r.ok:
        return True
    log.warning("frame -> %s: %s", r.status_code, r.text[:200])
    # 4xx is permanent (bad meta) -> drop; 5xx is transient -> keep + retry.
    return None if 400 <= r.status_code < 500 else False


def _effective_frame_cap() -> int:
    """Daily crop-upload cap in force: a dashboard-pushed value wins over the env
    default. Lets the fleet be throttled (or stopped, cap=0) without shell access."""
    try:
        d = json.loads(FRAME_CAP_FILE.read_text())
        cap = d.get("cap")
        if isinstance(cap, int) and not isinstance(cap, bool) and cap >= 0:
            return cap
    except (OSError, ValueError, AttributeError):
        pass
    return FRAME_DAILY_CAP


def _apply_frame_cap(value) -> None:
    """Persist a dashboard-pushed daily crop cap (only when changed). Ignores
    absent/invalid values so a cloud that doesn't send the field never clobbers a
    locally-set cap. Set 0 to stop crop upload over cellular entirely."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return
    try:
        cur = _effective_frame_cap()
        if value != cur:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            FRAME_CAP_FILE.write_text(json.dumps({"cap": value}))
            log.info("frame daily cap set to %d (was %d)", value, cur)
    except OSError as e:
        log.warning("could not write frame cap: %s", e)


def _apply_bee_mode(value) -> None:
    """Persist a dashboard-pushed bee-confirmation mode (off|tag|gate) to the file
    the recorder hot-reloads. Ignores absent/invalid values so a cloud that
    doesn't send the field never clobbers a locally-set mode."""
    if not isinstance(value, str):
        return
    mode = value.strip().lower()
    if mode not in ("off", "tag", "gate"):
        return
    try:
        new = json.dumps({"mode": mode}, sort_keys=True)
        cur = (BEE_CONFIRM_MODE_FILE.read_text().strip()
               if BEE_CONFIRM_MODE_FILE.exists() else "")
        if new != cur:
            BEE_CONFIRM_MODE_FILE.parent.mkdir(parents=True, exist_ok=True)
            BEE_CONFIRM_MODE_FILE.write_text(new)
            log.info("bee confirm mode set to %s", mode)
    except OSError as e:
        log.warning("could not write bee confirm mode: %s", e)


def _apply_activity_frames(value) -> None:
    """Persist the dashboard's crop-upload toggle to the file the recorder hot-
    reloads. value is a bool (send BioCLIP review crops or not). Ignores
    absent/non-bool values so a cloud that doesn't send the field never clobbers a
    locally-set toggle."""
    if not isinstance(value, bool):
        return
    try:
        new = json.dumps({"enabled": value}, sort_keys=True)
        cur = (ACTIVITY_FRAMES_FILE.read_text().strip()
               if ACTIVITY_FRAMES_FILE.exists() else "")
        if new != cur:
            ACTIVITY_FRAMES_FILE.parent.mkdir(parents=True, exist_ok=True)
            ACTIVITY_FRAMES_FILE.write_text(new)
            log.info("activity-frame crop upload set to %s", "on" if value else "off")
    except OSError as e:
        log.warning("could not write activity-frame toggle: %s", e)


def _drain_activity_frames() -> None:
    """Ship queued mover crops over cellular, under the daily cap. Best-effort:
    never raises, and stops on the first transient error to retry next beat."""
    if not ACTIVITY_FRAMES_QUEUE.is_dir():
        return
    metas = sorted(ACTIVITY_FRAMES_QUEUE.glob("*.json"), key=_safe_mtime)
    if not metas:
        return

    # Disk safety: bound the on-disk queue FIRST — before the cap check — so even
    # when the cap is 0 (crop upload off) a still-sampling recorder can't grow the
    # queue without bound. Only when it exceeds FRAME_QUEUE_MAX do we drop the
    # OLDEST crops, so frames merely over today's cap wait for tomorrow's budget.
    if FRAME_QUEUE_MAX > 0 and len(metas) > FRAME_QUEUE_MAX:
        overflow = metas[: len(metas) - FRAME_QUEUE_MAX]
        for meta_path in overflow:
            _drop_frame(meta_path.with_suffix(".jpg"), meta_path)
        log.warning("frames: queue over %d; dropped %d oldest crop(s) to bound disk",
                    FRAME_QUEUE_MAX, len(overflow))
        metas = metas[len(metas) - FRAME_QUEUE_MAX:]

    cap = _effective_frame_cap()
    if cap <= 0:
        return  # crop upload off — queue already bounded above

    sent_today = _frames_sent_today()
    sent_now = 0
    for meta_path in metas[:FRAME_BATCH_PER_BEAT]:
        if sent_today + sent_now >= cap:
            break  # daily budget spent — keep the rest for the next UTC day
        jpg = meta_path.with_suffix(".jpg")
        if not jpg.exists():
            _drop_frame(jpg, meta_path)  # orphan sidecar from a crashed write
            continue
        try:
            meta_raw = meta_path.read_text()
        except OSError:
            _drop_frame(jpg, meta_path)
            continue
        result = _send_one_frame(jpg, meta_raw)
        if result is True:
            _drop_frame(jpg, meta_path)
            sent_now += 1
        elif result is None:
            _drop_frame(jpg, meta_path)  # permanent reject
        else:
            break  # transient — leave the rest for the next beat
    if sent_now:
        _bump_frames_sent(sent_now)
        log.info("frames: sent %d (%d/%d today)", sent_now,
                 sent_today + sent_now, cap)


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

# Handle of a background USB copy, so beats keep flowing (and report progress)
# while it runs, and we can avoid launching a second copy on top of it.
_usb_proc = None

# Live beat interval — starts at the env default but can be changed from the
# dashboard (the heartbeat + command responses carry telemetry_interval).
_interval = INTERVAL


def _apply_interval(value) -> None:
    """Adopt a dashboard-set beat interval (clamped to >= MIN_INTERVAL)."""
    global _interval
    try:
        v = int(value)
    except (TypeError, ValueError):
        return
    if v < MIN_INTERVAL:
        v = MIN_INTERVAL
    if v != _interval:
        log.info("telemetry interval changed %ds -> %ds (from dashboard)", _interval, v)
        _interval = v


# --- Remote WittyPi power schedule: apply side (gated) -----------------------
# The desired schedule rides the heartbeat response (resp["wake_schedule"]). We
# validate + clamp it locally (the SERVER is never trusted for safety), and only
# when WAKE_SCHEDULE_APPLY do we write it to the WittyPi. Safety (memory/16 §5):
#   * reject any spec whose off-stretch exceeds the wake floor — can't strand it;
#   * keep the WittyPi low-voltage recovery in every mode (we never disable it);
#   * 'daylight' leaves the operator's installed schedule untouched (no regression).
# The whole apply path is unproven on hardware -> gated off by default.


def _parse_hhmm(s) -> "int | None":
    try:
        h, m = str(s).split(":")
        h, m = int(h), int(m)
    except (TypeError, ValueError):
        return None
    return h * 60 + m if (0 <= h <= 23 and 0 <= m <= 59) else None


def _fmt_hhmm(mins: int) -> str:
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _clean_schedule(spec) -> "dict | None":
    """Validate a desired spec and enforce the wake floor; None if unusable."""
    if not isinstance(spec, dict):
        return None
    mode = spec.get("mode")
    floor_min = WAKE_FLOOR_HOURS * 60
    if mode in ("daylight", "always_on"):
        return {"mode": mode}
    if mode == "window":
        on, off = _parse_hhmm(spec.get("on")), _parse_hhmm(spec.get("off"))
        if on is None or off is None or on == off:
            return None
        on_dur = (off - on) % 1440           # minutes ON
        off_dur = 1440 - on_dur              # minutes OFF (overnight stretch)
        if on_dur <= 0 or off_dur > floor_min:   # off stretch must respect the floor
            return None
        return {"mode": "window", "on": _fmt_hhmm(on), "off": _fmt_hhmm(off)}
    if mode == "interval":
        try:
            every, on_min = int(spec.get("wake_every_min")), int(spec.get("on_minutes"))
        except (TypeError, ValueError):
            return None
        if not (5 <= every <= 1440) or not (1 <= on_min < every):
            return None
        if (every - on_min) > floor_min:
            return None
        return {"mode": "interval", "wake_every_min": every, "on_minutes": on_min}
    return None


def _schedule_to_wpi(spec: dict) -> "str | None":
    """Translate a window/interval spec into a WittyPi .wpi script (BEGIN/END +
    repeating ON/OFF durations). None for modes that don't use a .wpi."""
    end = "END   2099-12-31 23:59:59"
    if spec["mode"] == "interval":
        on_min = spec["on_minutes"]
        off_min = spec["wake_every_min"] - on_min
        begin = datetime.now().strftime("BEGIN %Y-%m-%d 00:00:00")
        return f"{begin}\n{end}\nON    M{on_min}\nOFF   M{off_min}\n"
    if spec["mode"] == "window":
        on = _parse_hhmm(spec["on"])
        on_dur = (_parse_hhmm(spec["off"]) - on) % 1440
        off_dur = 1440 - on_dur
        begin = datetime.now().strftime("BEGIN %Y-%m-%d ") + f"{spec['on']}:00"
        return (f"{begin}\n{end}\n"
                f"ON    H{on_dur // 60} M{on_dur % 60}\n"
                f"OFF   H{off_dur // 60} M{off_dur % 60}\n")
    return None


def _write_applied_spec(spec: dict) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        WAKE_SCHEDULE_STATE.write_text(json.dumps(spec))
    except OSError as e:
        log.warning("could not record applied wake_schedule: %s", e)


def _disable_schedule_file(d: Path) -> None:
    """Move the recurring .wpi aside so the daemon stops re-arming a shutdown
    (used by always_on). Best-effort."""
    sched = d / "schedule.wpi"
    try:
        if sched.exists():
            sched.rename(d / "schedule.wpi.disabled")
    except OSError:
        pass


def _write_schedule_file(d: Path, wpi_text: str) -> bool:
    """Write our generated schedule.wpi into the WittyPi dir. Falls back to a
    sudo'd write if the dir is root-owned (telemetry runs as beemonitor)."""
    target = d / "schedule.wpi"
    try:
        target.write_text(wpi_text)
        return True
    except OSError as e:
        log.info("wake_schedule: direct write of %s failed (%s); trying sudo", target, e)
    cmd = f"printf '%s' {shlex.quote(wpi_text)} > {shlex.quote(str(target))}"
    r = _run(["sudo", "-n", "bash", "-c", cmd], timeout=10)
    if r is not None and r.returncode == 0:
        return True
    log.warning("wake_schedule: could not write %s (perms?): %s", target,
                (r.stderr or "").strip()[:160] if r else "no result")
    return False


def _write_and_run_wpi(d: Path, wpi_text: str) -> bool:
    """Install our generated schedule where runScript.sh expects it and apply it
    (runScript.sh parses schedule.wpi and programs the next startup/shutdown)."""
    if not _write_schedule_file(d, wpi_text):
        return False
    # Run runScript.sh DIRECTLY — do NOT pre-source utilities.sh here: runScript.sh
    # sources it itself, and utilities.sh declares `readonly` constants, so a second
    # source errors and aborts the script. Capture output so a failure is visible in
    # the journal (the old version swallowed it with >/dev/null).
    cmd = f"cd {shlex.quote(str(d))} && ./runScript.sh"
    r = _run(["bash", "-c", cmd], timeout=25)
    if (r is None or r.returncode != 0) and os.geteuid() != 0:
        r = _run(["sudo", "-n", "bash", "-c", cmd], timeout=25)
    if r is not None and r.returncode == 0:
        log.info("wake_schedule: runScript.sh applied schedule.wpi")
        return True
    log.warning("wake_schedule: runScript.sh failed rc=%s err=%r out=%r",
                (r.returncode if r else None),
                ((r.stderr or "").strip()[:200] if r else None),
                ((r.stdout or "").strip()[:200] if r else None))
    return False


def _do_apply(spec: dict) -> bool:
    """Reconcile the WittyPi to a cleaned spec. Returns True on success."""
    d = _wittypi_dir()
    if d is None:
        return False
    mode = spec["mode"]
    if mode == "daylight":
        return True  # keep the operator's installed schedule untouched
    if mode == "always_on":
        # Never power off on schedule: drop the recurring .wpi and clear the
        # pending shutdown alarm. The low-voltage cutoff stays active.
        _disable_schedule_file(d)
        r = _wittypi_sh("clear_shutdown_time")
        return r is not None and r.returncode == 0
    wpi = _schedule_to_wpi(spec)
    return bool(wpi) and _write_and_run_wpi(d, wpi)


_TZ_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._+-]*(?:/[A-Za-z0-9._+-]+)*$")


def _current_tz() -> str:
    """The Pi's current system timezone (IANA name), best-effort."""
    try:
        return Path("/etc/timezone").read_text().strip()
    except OSError:
        try:  # /etc/localtime -> .../zoneinfo/<Area>/<City>
            return os.path.realpath("/etc/localtime").split("zoneinfo/", 1)[1]
        except (OSError, IndexError):
            return ""


def _apply_device_tz(tz) -> None:
    """Set the Pi's clock timezone to the dashboard's location-derived value.

    Why it matters: the WittyPi wake window is stamped in LOCAL time
    (``_schedule_to_wpi``), so a wrong system tz fires the window hours off. We keep
    the Pi's tz matching its GPS location. Idempotent; strictly validated; needs the
    ``beemonitor-timedatectl`` sudoers entry. On a real change we re-arm the wake
    schedule so its .wpi is re-stamped in the corrected local time. The actual UTC
    clock (NTP) is untouched — this only relabels local time, so TLS/cellular are
    unaffected."""
    if not tz or not isinstance(tz, str):
        return
    tz = tz.strip()
    if ".." in tz or not _TZ_RE.match(tz):
        log.warning("ignoring invalid device_tz from dashboard: %r", tz)
        return
    if not Path("/usr/share/zoneinfo", tz).exists():
        log.warning("device_tz %r not in /usr/share/zoneinfo — ignoring", tz)
        return
    if _current_tz() == tz:
        return  # already correct
    r = _run(["sudo", "-n", "timedatectl", "set-timezone", tz])
    if not r or r.returncode != 0:
        log.warning("timedatectl set-timezone %s failed: %s", tz,
                    (r.stderr.strip() if r and r.stderr else "no result"))
        return
    log.info("device timezone -> %s (from dashboard/GPS)", tz)
    try:
        time.tzset()  # make THIS process pick up the new localtime immediately
    except Exception:  # pragma: no cover
        pass
    # Force the wake schedule to re-arm in the new tz (its .wpi BEGIN was stamped
    # in the old one). Dropping the applied-spec cache makes the next
    # _apply_schedule re-write it; harmless when apply is off (we manage no .wpi).
    try:
        if WAKE_SCHEDULE_STATE.exists():
            WAKE_SCHEDULE_STATE.unlink()
    except OSError:
        pass


def _apply_schedule(spec, apply_flag=None) -> None:
    """Adopt the dashboard's desired WittyPi schedule from the heartbeat response.

    Validates + clamps (wake floor) locally; idempotent; and writes it to the
    WittyPi only when apply is enabled — either per-device from the dashboard
    (``apply_flag``, the wake_schedule_apply field) OR the global
    BEEMONITOR_WAKE_SCHEDULE_APPLY env override. Off by default: report-only, so it
    can't strand a unit while unproven. The wake floor still bounds any applied
    schedule's off-stretch, so even with apply on a bad spec can't strand it.
    """
    if spec is None:
        return
    cleaned = _clean_schedule(spec)
    if cleaned is None:
        log.warning("wake_schedule rejected (invalid or off-stretch > %sh floor): %s",
                    WAKE_FLOOR_HOURS, spec)
        return
    apply_on = WAKE_SCHEDULE_APPLY or bool(apply_flag)
    if _read_applied_spec() == cleaned:
        return  # already applied — nothing to do
    if not apply_on:
        log.info("wake_schedule '%s' received — apply disabled (report-only); enable "
                 "it on the device page (or BEEMONITOR_WAKE_SCHEDULE_APPLY=1)",
                 cleaned.get("mode"))
        return
    if _do_apply(cleaned):
        _write_applied_spec(cleaned)
        log.info("wake_schedule applied to WittyPi: %s", cleaned)
    else:
        log.warning("wake_schedule apply failed (leaving previous schedule): %s", cleaned)


def _safe_mtime(p) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _capture_now(timeout: float = 12.0, roi: bool = False):
    """Ask the recorder (via a sentinel file) for one fresh still; return its Path.

    The recorder owns the camera, so telemetry can't grab a frame directly — it
    drops ``capture.request`` in the queue, the recorder writes a JPEG there, and
    we return the newest one. None if nothing arrives in ``timeout`` s. When
    ``roi`` is set the request asks the recorder to overlay the motion ROI +
    nest-hole detections (a debug aid; the recorder runs the model, so allow
    more time).
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    req = QUEUE_DIR / "capture.request"
    t0 = time.time()
    try:
        req.write_text("roi" if roi else str(t0))
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


def _capture_and_upload(params: dict | None = None) -> bool:
    roi = bool((params or {}).get("roi"))
    # The ROI overlay makes the recorder run the nest model — give it longer.
    img = _capture_now(timeout=30.0 if roi else 12.0, roi=roi)
    if img is None:
        log.warning("on-demand capture produced no still (is the recorder running?)")
        return False
    return send_beat(image=img) is not None


def _open_cellular(params: dict) -> None:
    """DEBUG: drop the cellular egress gate so rpi-connect (and anything else) can
    use mobile data for remote shell access — then AUTO-REVERT to gated after a
    timeout so an open metered link can't quietly burn the SIM. The revert is a
    one-shot systemd timer; a reboot also re-gates (cellular-firewall.service)."""
    try:
        minutes = int(params.get("minutes") or CELL_DEBUG_DEFAULT_MIN)
    except (TypeError, ValueError):
        minutes = CELL_DEBUG_DEFAULT_MIN
    minutes = max(1, min(minutes, 240))  # clamp 1..240 min
    # The script (run as root) drops the gate AND schedules its own auto-revert,
    # so telemetry needs only one tightly-scoped sudoers entry for this script.
    r = _run(["sudo", "-n", str(FIREWALL_SCRIPT), "open", str(minutes)], timeout=30)
    if r is None or r.returncode != 0:
        log.warning("cellular_open: could not drop the gate: %s",
                    (r.stderr or "").strip()[:160] if r else "no result")
    else:
        log.info("cellular_open: wwan0 ungated for %d min (debug)", minutes)


def _gate_cellular() -> None:
    """Re-gate cellular egress to telemetry-only now (and cancel any auto-revert)."""
    r = _run(["sudo", "-n", str(FIREWALL_SCRIPT), "telemetry"], timeout=30)
    if r is None or r.returncode != 0:
        log.warning("cellular_gate: could not re-gate: %s",
                    (r.stderr or "").strip()[:160] if r else "no result")
    else:
        log.info("cellular_gate: wwan0 re-gated to telemetry-only")


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
    # nmcli's `device wifi connect` only consults the cached scan list, so a
    # just-enabled or just-renamed AP (e.g. a phone hotspot that wasn't
    # broadcasting at the last scan) fails with rc=10 'No network with SSID ...
    # found'. Force a rescan and retry once — but only for that case, since a bad
    # password or other error won't be fixed by scanning again.
    if (r is None or r.returncode != 0) and \
            "No network with SSID" in (getattr(r, "stderr", "") or ""):
        log.info("wifi_connect: '%s' not in scan — rescanning and retrying", ssid)
        _nmcli(["device", "wifi", "rescan"], timeout=20)  # rate-limit error is harmless
        time.sleep(6)  # let scan results populate before retrying
        r = _nmcli(args, timeout=60)

    # Never log the password — only the SSID + result.
    if r is not None and r.returncode == 0:
        log.info("wifi_connect: joined '%s'", ssid)
        _prefer_wifi_route()  # make telemetry ride this WiFi, not cellular
    else:
        log.warning("wifi_connect: failed for '%s' (rc=%s) %s", ssid,
                    getattr(r, "returncode", None),
                    (getattr(r, "stderr", "") or "").strip()[:200])


def _start_update(params: dict) -> None:
    """Spawn the updater's fetch phase as a detached child.

    The child stays in THIS service's cgroup, so its git/pip traffic is allowed
    through the cellular firewall. We return immediately (don't block the beat
    loop while it fetches); the updater hands off to beemonitor-update.service to
    restart + health-check + roll back. See hardware/update.sh.
    """
    if not UPDATE_SCRIPT.exists():
        log.warning("command: update — %s missing, ignoring", UPDATE_SCRIPT)
        return
    # Artifact update (feature 18): the cloud sent a signed-bundle descriptor
    # {version,url,sha256,sig,reqs_hash}. Write it where update.sh reads it and run
    # the artifact fetch path. Otherwise fall back to the git fetch (params.ref).
    if params.get("url") and params.get("version") and params.get("sha256"):
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            desc = {k: params.get(k) for k in
                    ("version", "url", "sha256", "sig", "reqs_hash")}
            (STATE_DIR / "update-descriptor.json").write_text(json.dumps(desc))
        except OSError as e:
            log.warning("command: update — cannot write descriptor: %s", e)
            return
        log.info("command: update version=%s — spawning artifact fetch", params.get("version"))
        cmd = ["bash", str(UPDATE_SCRIPT), "fetch-artifact"]
    else:
        ref = (params.get("ref") or "origin/main").strip() or "origin/main"
        log.info("command: update ref=%s — spawning git fetch", ref)
        cmd = ["bash", str(UPDATE_SCRIPT), "fetch", ref]
    try:
        subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, cwd=str(REPO_DIR),
        )
    except OSError as e:
        log.warning("command: update — failed to spawn updater: %s", e)


def _apply_motion_tuning(tuning) -> None:
    """Persist dashboard motion-tuning overrides to motion_tuning.json (only when
    they change). The recorder hot-reloads it on top of calibration."""
    if not isinstance(tuning, dict):
        return
    try:
        new = json.dumps(tuning, sort_keys=True)
        cur = MOTION_TUNING_FILE.read_text().strip() if MOTION_TUNING_FILE.exists() else ""
        if new != cur:
            MOTION_TUNING_FILE.parent.mkdir(parents=True, exist_ok=True)
            MOTION_TUNING_FILE.write_text(new)
            log.info("motion tuning updated: %s", tuning or "(cleared)")
    except OSError as e:
        log.warning("could not write motion tuning: %s", e)


def _apply_override_file(path, value) -> None:
    """Sync a dashboard override to a file the recorder hot-reloads. ``None``
    removes it (revert to auto); a value is written as JSON only when changed."""
    try:
        if value is None:
            if path.exists():
                path.unlink()
                log.info("cleared %s", path.name)
            return
        new = json.dumps(value, sort_keys=True)
        cur = path.read_text().strip() if path.exists() else ""
        if new != cur:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new)
            log.info("updated %s", path.name)
    except OSError as e:
        log.warning("could not write %s: %s", path.name, e)


def _cache_location(value) -> None:
    """Persist the device's dashboard location so usb-transfer.sh can name the
    USB folder after the hive. Written only when it changes (cheap, idempotent),
    and BEFORE any command runs so an immediate usb_transfer uses a fresh label.
    """
    if not isinstance(value, str):
        return
    loc = value.strip()
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        cur = LOCATION_FILE.read_text().strip() if LOCATION_FILE.exists() else ""
        if loc != cur:
            LOCATION_FILE.write_text(loc)
            log.info("device location cached: %r", loc)
    except OSError as e:
        log.debug("could not cache location: %s", e)


def _handle_command(cmd: str, params: dict) -> None:
    if cmd == "capture_image":
        log.info("command: capture_image roi=%s", bool(params.get("roi")))
        _capture_and_upload(params)
    elif cmd == "update":
        _start_update(params)
    elif cmd == "usb_transfer":
        # Run the copy in the BACKGROUND so beats keep flowing — the script
        # writes progress to usb-status.json each step, which we report every
        # beat, so the dashboard shows a live percentage. Needs root to mount the
        # USB (NOPASSWD sudoers rule). Skip if a copy is already running.
        global _usb_proc
        if _usb_proc is not None and _usb_proc.poll() is None:
            log.info("command: usb_transfer — a copy is already running; ignoring")
        else:
            log.info("command: usb_transfer — starting background copy")
            try:
                _usb_proc = subprocess.Popen(
                    ["sudo", "-n", str(USB_SCRIPT)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except OSError as e:
                log.warning("command: usb_transfer — failed to start: %s", e)
    elif cmd == "usb_eject":
        log.info("command: usb_eject")
        # Quick (just unmounts); fine to run inline.
        _run(["sudo", "-n", str(USB_SCRIPT), "--eject"], timeout=60)
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
    elif cmd == "cellular_open":
        log.info("command: cellular_open minutes=%s", params.get("minutes"))
        _open_cellular(params)
    elif cmd == "cellular_gate":
        log.info("command: cellular_gate")
        _gate_cellular()
    else:
        log.warning("ignoring unknown command: %s", cmd)


def _poll_command() -> "dict | None":
    """Lightweight check for a pending on-demand command between beats.

    Cheap GET (no metrics/image), so we can poll it every few seconds — a
    requested photo arrives in ~COMMAND_POLL_SECONDS instead of waiting for the
    next 60s beat. Returns the parsed response (carries telemetry_interval too,
    so a rate change applies within seconds even on a slow beat cadence).
    """
    try:
        r = requests.get(
            urljoin(API_BASE + "/", "api/v1/devices/command"),
            headers={"Authorization": f"Bearer {DEVICE_KEY}"},
            timeout=30,
        )
        if r.ok:
            d = r.json()
            _cache_location(d.get("location"))  # before the command, so usb_transfer sees it
            if d.get("command"):
                _handle_command(d["command"], d.get("params") or {})
            return d
    except requests.RequestException:
        pass
    return None


def _handle_signal(signum, frame):  # noqa: ARG001
    global _running
    _running = False


def _scan_uploaded_video_ids() -> dict:
    """Map server video_id -> local .mp4 Path, read from the .uploaded sidecars.

    The uploader writes ``video_id=<id>`` into ``<clip>.mp4.uploaded`` on success,
    so a sidecar's presence proves the clip is on the server — the first of the
    two deletion keys.
    """
    out: dict = {}
    try:
        sidecars = RECORD_DIR.rglob("*.uploaded")
    except OSError:
        return out
    for sidecar in sidecars:
        try:
            for line in sidecar.read_text().splitlines():
                if line.startswith("video_id="):
                    vid = line.split("=", 1)[1].strip()
                    if vid.isdigit():
                        out[int(vid)] = sidecar.with_suffix("")  # strip ".uploaded"
                    break
        except OSError:
            continue
    return out


def _run_cleanup() -> None:
    """Delete local clips a human cleared for deletion on the dashboard.

    Two-key safety: a clip is eligible only if it uploaded (it has an .uploaded
    sidecar with a server video_id) AND the server returns its id (a human
    requested deletion). We delete only those, then POST a confirmation so the
    server stamps device_deleted_at and stops listing them.
    """
    base = API_BASE + "/"
    hdrs = {"Authorization": f"Bearer {DEVICE_KEY}"}
    try:
        r = requests.get(urljoin(base, "api/v1/devices/cleanup"), headers=hdrs,
                         timeout=POST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("cleanup: list failed: %s", e)
        return
    if r.status_code != 200:
        log.warning("cleanup: list -> %s %s", r.status_code, r.text[:200])
        return
    ids = (r.json() or {}).get("video_ids") or []
    if not ids:
        return

    local = _scan_uploaded_video_ids()
    deleted, freed = [], 0
    for vid in ids:
        vid = int(vid)
        mp4 = local.get(vid)
        if mp4 and mp4.exists():
            try:
                freed += mp4.stat().st_size
                mp4.unlink()
                sidecar = mp4.with_suffix(mp4.suffix + ".uploaded")
                if sidecar.exists():
                    sidecar.unlink()
                deleted.append(vid)
            except OSError as e:
                log.warning("cleanup: could not delete %s: %s", mp4, e)
        else:
            # Not here anymore — confirm anyway so the server stops listing it.
            deleted.append(vid)
    if not deleted:
        return
    try:
        requests.post(urljoin(base, "api/v1/devices/cleanup"), headers=hdrs,
                      json={"deleted": deleted}, timeout=POST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("cleanup: confirm failed (will retry next pass): %s", e)
        return
    log.info("cleanup: freed %s across %d clip(s)", _human_bytes(freed), len(deleted))


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
    # Make telemetry ride WiFi (not metered cellular) whenever WiFi is connected.
    _prefer_wifi_route()
    log.info("telemetry loop started — interval=%ds, queue=%s", _interval, QUEUE_DIR)

    last_cleanup = 0.0
    while _running:
        try:
            resp = send_beat()
            if resp:
                _cache_location(resp.get("location"))  # keep the USB-folder label fresh
                # The beat may also carry a command (belt-and-suspenders)...
                if resp.get("command"):
                    _handle_command(resp["command"], resp.get("params") or {})
                # ...and the dashboard-set beat interval.
                _apply_interval(resp.get("telemetry_interval"))
                # ...and any dashboard motion-tuning overrides.
                _apply_motion_tuning(resp.get("motion_tuning"))
                # ...and a dashboard-pushed crop daily cap (0 = stop crop upload).
                _apply_frame_cap(resp.get("frame_daily_cap"))
                # ...and the bee-confirmation mode (off|tag|gate).
                _apply_bee_mode(resp.get("bee_confirm_mode"))
                # ...and the crop-upload toggle (send BioCLIP review crops or not).
                _apply_activity_frames(resp.get("activity_frames"))
                # ...and the GPS-derived clock timezone (keeps the local-time wake
                # window correct). Applied before the schedule so a tz change
                # re-arms the .wpi in the corrected local time this same beat.
                _apply_device_tz(resp.get("device_tz"))
                # ...and the ROI-editor outputs (hotel ROI + nest layout).
                _apply_override_file(ROI_OVERRIDE_FILE, resp.get("roi_override"))
                _apply_override_file(NEST_LAYOUT_FILE, resp.get("nest_layout"))
                # ...and the desired WittyPi power schedule (validated + clamped
                # locally; only written to the WittyPi when apply is enabled,
                # per-device via wake_schedule_apply from the dashboard).
                _apply_schedule(resp.get("wake_schedule"), resp.get("wake_schedule_apply"))
                # The link is up (the beat landed) — ship a batch of queued
                # activity-frame crops for taxonomic ID, under the daily cap.
                _drain_activity_frames()
            # Free SD space: delete clips a human cleared on the dashboard.
            if CLEANUP_INTERVAL > 0 and time.time() - last_cleanup >= CLEANUP_INTERVAL:
                _run_cleanup()
                last_cleanup = time.time()
        except Exception as e:  # pragma: no cover - never let the loop die
            log.exception("beat raised: %s", e)
        # Between beats, poll for on-demand commands. On WiFi: every
        # COMMAND_POLL_SECONDS (free). On metered cellular: at a slow floor every
        # CELL_COMMAND_POLL_SECONDS (a lightweight GET — no metrics/image — that
        # rides the firewall-allowed telemetry cgroup), so dashboard commands land
        # within ~2 min and last_seen stays fresh, WITHOUT cranking the heavy beat.
        # Set CELL_COMMAND_POLL_SECONDS=0 for beat-only on cellular. The transport
        # check is a local `ip route get` (no traffic); ``_interval`` is read live
        # so lowering it breaks the wait early.
        slept = 0
        last_cell_poll = 0  # seconds-into-window of the last cellular poll
        while _running and slept < _interval:
            step = min(COMMAND_POLL_SECONDS, _interval - slept)
            time.sleep(step)
            slept += step
            if not (_running and slept < _interval):
                break
            transport = _active_transport()
            if transport == "wifi":
                _apply_interval((_poll_command() or {}).get("telemetry_interval"))
            elif (transport == "cellular" and CELL_COMMAND_POLL_SECONDS > 0
                    and slept - last_cell_poll >= CELL_COMMAND_POLL_SECONDS):
                last_cell_poll = slept
                _apply_interval((_poll_command() or {}).get("telemetry_interval"))
    log.info("telemetry stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
