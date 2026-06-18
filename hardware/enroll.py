#!/usr/bin/env python3
"""Zero-touch device enrollment (runs once, as root, before the app services).

A unit flashed from a pre-baked image carries an *enrollment token* but no device
key. On first boot this script:
  1. reads the per-card provisioning values (BEEMONITOR_ENROLL_TOKEN,
     BEEMONITOR_API_BASE) from the FAT *boot* partition if present
     (/boot/firmware/beemonitor.conf — written by prepare-card.sh from any
     laptop), else from /etc/beemonitor/uploader.env,
  2. if BEEMONITOR_DEVICE_KEY is already set -> nothing to do,
  3. else if a token is present -> POST it (+ this Pi's hardware id, hostname,
     timezone) to /api/v1/devices/enroll, get a fresh device key, write it into
     uploader.env, and clear the reusable token from the boot partition.

Stdlib only (urllib), so it runs under the system python3 as root without the
venv. Idempotent: re-running once a key exists is a no-op; the server maps a
re-flashed Pi (same hardware id) back to the same Device.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ENV_FILE = os.environ.get("BEEMONITOR_ENV", "/etc/beemonitor/uploader.env")

# The per-card provisioning file lives on the FAT boot partition so it can be
# written from any laptop (macOS/Windows/Linux) without ext4 tooling. Bookworm
# mounts the boot partition at /boot/firmware; older images used /boot.
BOOT_CONF_CANDIDATES = [
    os.environ.get("BEEMONITOR_BOOT_CONF"),
    "/boot/firmware/beemonitor.conf",
    "/boot/beemonitor.conf",
]


def _parse_env(path):
    env = {}
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    except OSError:
        pass
    return env


def _boot_conf():
    """Return (path, dict) for the first boot-partition provisioning file found.

    ('', {}) if none exists — a manually set up unit just uses uploader.env.
    """
    for path in BOOT_CONF_CANDIDATES:
        if path and os.path.exists(path):
            return path, _parse_env(path)
    return "", {}


def _clear_boot_lines(path, prefixes):
    """Remove lines starting with any of ``prefixes`` from the boot-partition conf,
    so secrets (the enrollment token, the WiFi password) don't linger on an
    easily-read FAT partition once they've been consumed. Best-effort, never fatal."""
    try:
        with open(path) as fh:
            lines = [ln for ln in fh
                     if not any(ln.strip().startswith(p) for p in prefixes)]
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.writelines(lines)
        os.replace(tmp, path)
    except OSError as e:
        print(f"enroll: could not scrub boot conf (non-fatal): {e}", file=sys.stderr)


def _configure_wifi(boot, boot_path):
    """Join WiFi from the boot conf (BEEMONITOR_WIFI_SSID / _PASSWORD) on first
    boot, so a freshly-flashed unit gets online BEFORE enrollment (which needs a
    network). Creates a persistent autoconnect NetworkManager profile, brings it
    up, then scrubs the password from the boot partition. Best-effort — cellular
    still carries enrollment if this fails. Runs as root, so nmcli needs no sudo."""
    ssid = (boot.get("BEEMONITOR_WIFI_SSID") or "").strip()
    if not ssid:
        return
    psk = (boot.get("BEEMONITOR_WIFI_PASSWORD") or "").strip()
    try:
        shown = subprocess.run(["nmcli", "-t", "-f", "NAME", "connection", "show"],
                               capture_output=True, text=True, timeout=15)
        if shown.returncode == 0 and ssid in shown.stdout.split("\n"):
            print(f"enroll: wifi '{ssid}' already configured")
        else:
            print(f"enroll: configuring wifi '{ssid}'")
            subprocess.run(["nmcli", "radio", "wifi", "on"],
                           capture_output=True, timeout=15)
            cmd = ["nmcli", "connection", "add", "type", "wifi", "con-name", ssid,
                   "ssid", ssid, "connection.autoconnect", "yes"]
            if psk:
                cmd += ["wifi-sec.key-mgmt", "wpa-psk", "wifi-sec.psk", psk]
            add = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if add.returncode != 0:
                print(f"enroll: wifi add failed: {add.stderr.strip()}", file=sys.stderr)
        # Try to bring it up now; enrollment's own retries cover a slow join.
        subprocess.run(["nmcli", "connection", "up", ssid],
                       capture_output=True, timeout=45)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"enroll: wifi setup error (non-fatal): {e}", file=sys.stderr)
    if boot_path:
        _clear_boot_lines(boot_path, ["BEEMONITOR_WIFI_PASSWORD="])


def _hw_id():
    """Stable hardware id: the Pi serial, else the machine-id."""
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("Serial"):
                    val = line.split(":", 1)[1].strip()
                    if val and set(val) != {"0"}:
                        return val
    except OSError:
        pass
    try:
        return open("/etc/machine-id").read().strip()
    except OSError:
        return ""


def _timezone():
    out = {}
    try:
        out["tz"] = open("/etc/timezone").read().strip()
    except OSError:
        pass
    try:
        off = datetime.now().astimezone().utcoffset()
        if off is not None:
            out["tz_offset_min"] = int(off.total_seconds() // 60)
    except (OSError, ValueError):
        pass
    return out


def _write_device_key(path, env, key):
    """Set BEEMONITOR_DEVICE_KEY in the env file, preserving other lines."""
    lines, replaced = [], False
    try:
        with open(path) as fh:
            for line in fh:
                if line.strip().startswith("BEEMONITOR_DEVICE_KEY="):
                    lines.append(f"BEEMONITOR_DEVICE_KEY={key}\n")
                    replaced = True
                else:
                    lines.append(line)
    except OSError:
        lines = [f"{k}={v}\n" for k, v in env.items()]
    if not replaced:
        lines.append(f"BEEMONITOR_DEVICE_KEY={key}\n")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.writelines(lines)
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def main():
    # Per-card provisioning lives on the FAT boot partition (written by
    # prepare-card.sh / the browser installer from any laptop) or in uploader.env;
    # the boot partition wins. Read it FIRST and bring up WiFi from it before
    # anything else — enrollment (and a WiFi-only unit) needs a network up.
    boot_path, boot = _boot_conf()
    _configure_wifi(boot, boot_path)

    env = _parse_env(ENV_FILE)
    if env.get("BEEMONITOR_DEVICE_KEY"):
        print("enroll: device key already present — nothing to do")
        return 0

    token = (boot.get("BEEMONITOR_ENROLL_TOKEN")
             or env.get("BEEMONITOR_ENROLL_TOKEN", "")).strip()
    if not token:
        print("enroll: no BEEMONITOR_ENROLL_TOKEN — skipping (manual setup)")
        return 0
    api_base = (boot.get("BEEMONITOR_API_BASE")
                or env.get("BEEMONITOR_API_BASE", "")).rstrip("/")
    if not api_base:
        print("enroll: BEEMONITOR_API_BASE missing", file=sys.stderr)
        return 1

    hw_id = _hw_id()
    if not hw_id:
        print("enroll: could not determine a hardware id", file=sys.stderr)
        return 1

    payload = {"token": token, "hw_id": hw_id, "hostname": socket.gethostname()}
    payload.update(_timezone())
    body = json.dumps(payload).encode()
    url = api_base + "/api/v1/devices/enroll"

    # Retry a few times — the network/cellular link may still be coming up.
    for attempt in range(1, 8):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            key = data.get("device_key")
            if key:
                _write_device_key(ENV_FILE, env, key)
                if boot_path:
                    _clear_boot_lines(boot_path, ["BEEMONITOR_ENROLL_TOKEN="])
                    print("enroll: cleared enrollment token from boot partition")
                print(f"enroll: provisioned device '{data.get('name')}' "
                      f"(id={data.get('device_id')}) — key written")
                return 0
            print("enroll: response had no device_key", file=sys.stderr)
            return 1
        except Exception as e:  # noqa: BLE001 - keep retrying on any transport error
            print(f"enroll: attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(min(30, 5 * attempt))
    print("enroll: gave up after retries (will try again next boot)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
