#!/usr/bin/env python3
"""Zero-touch device enrollment (runs once, as root, before the app services).

A unit flashed from a pre-baked image carries an *enrollment token* but no device
key. On first boot this script:
  1. reads /etc/beemonitor/uploader.env,
  2. if BEEMONITOR_DEVICE_KEY is already set -> nothing to do,
  3. else if BEEMONITOR_ENROLL_TOKEN is set -> POST it (+ this Pi's hardware id,
     hostname, timezone) to /api/v1/devices/enroll, get a fresh device key, and
     write it back into uploader.env.

Stdlib only (urllib), so it runs under the system python3 as root without the
venv. Idempotent: re-running once a key exists is a no-op; the server maps a
re-flashed Pi (same hardware id) back to the same Device.
"""

import json
import os
import socket
import sys
import time
import urllib.request
from datetime import datetime

ENV_FILE = os.environ.get("BEEMONITOR_ENV", "/etc/beemonitor/uploader.env")


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
    env = _parse_env(ENV_FILE)
    if env.get("BEEMONITOR_DEVICE_KEY"):
        print("enroll: device key already present — nothing to do")
        return 0
    token = env.get("BEEMONITOR_ENROLL_TOKEN", "").strip()
    if not token:
        print("enroll: no BEEMONITOR_ENROLL_TOKEN — skipping (manual setup)")
        return 0
    api_base = env.get("BEEMONITOR_API_BASE", "").rstrip("/")
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
