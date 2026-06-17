"""Live validation checks for the walkthrough.

Each check reads the device's REAL state — from telemetry we already collect —
and returns a structured verdict so the wizard can gate "Continue" on the
device actually working, not a self-reported checkbox. No Pi-side code is needed:
"online" is derived from last_seen_at, and service/wifi/cellular flags come from
the latest heartbeat metrics.

A check returns: {"ok": bool, "status": "pass"|"fail"|"waiting", "detail": str}
"""

from django.utils import timezone


def _latest_metrics(device):
    hb = device.heartbeats.first() if device else None
    return (hb.metrics or {}) if hb else {}


def _is_online(device) -> bool:
    # Online window scales to the device's beat cadence (Device.is_online).
    return bool(device) and device.is_online()


def _waiting(detail):
    return {"ok": False, "status": "waiting", "detail": detail}


def _pass(detail):
    return {"ok": True, "status": "pass", "detail": detail}


def _fail(detail):
    return {"ok": False, "status": "fail", "detail": detail}


def check_device_online(device) -> dict:
    if device is None:
        return _fail("No device is linked to this setup yet.")
    if _is_online(device):
        secs = int((timezone.now() - device.last_seen_at).total_seconds())
        return _pass(f"Device is online — last checked in {secs}s ago.")
    if device.last_seen_at:
        return _fail("Device has checked in before but is now stale. Check "
                     "power/network and `journalctl -u beemonitor-telemetry`.")
    return _waiting("Waiting for the first heartbeat… start the telemetry "
                    "service and give it up to a minute.")


def check_services_running(device) -> dict:
    if not _is_online(device):
        return _waiting("Device offline — can't read service status yet.")
    m = _latest_metrics(device)
    rec, up = m.get("recorder_active"), m.get("uploader_active")
    if rec and up:
        return _pass("Recorder and uploader are both active.")
    missing = [n for n, v in (("recorder", rec), ("uploader", up)) if not v]
    return _fail("Not active: " + ", ".join(missing) +
                 ". `systemctl status beemonitor-" + missing[0] + "`.")


def check_wifi_connected(device) -> dict:
    if not _is_online(device):
        return _waiting("Device offline — can't read WiFi state yet.")
    m = _latest_metrics(device)
    if m.get("wifi_state") == "connected":
        ssid = m.get("wifi_ssid") or "(unknown SSID)"
        return _pass(f"WiFi connected to {ssid}.")
    return _fail("WiFi not connected. Use the device's WiFi controls or "
                 "`nmcli device wifi connect <ssid>`.")


def check_cellular_up(device) -> dict:
    if not _is_online(device):
        return _waiting("Device offline — waiting for it to check in (over "
                        "cellular this confirms the link itself).")
    m = _latest_metrics(device)
    if m.get("cellular_active"):
        return _pass("Cellular service is active and the unit is checking in.")
    return _fail("cellular.service isn't active. `systemctl status "
                 "cellular.service` and check the modem/APN.")


def check_camera_ok(device) -> dict:
    """A recent on-demand image having arrived is proof the camera works."""
    if not _is_online(device):
        return _waiting("Device offline — request an image once it's online.")
    hb = device.heartbeats.first()
    if hb and hb.image_storage_key:
        return _pass("A camera image has been received from this device.")
    return _waiting("No image received yet. Use 'Request image' on the device "
                    "page to confirm the camera.")


def check_storage_ok(device) -> dict:
    if not _is_online(device):
        return _waiting("Device offline — can't read storage yet.")
    m = _latest_metrics(device)
    sp = m.get("storage_pct")
    if sp is None:
        return _waiting("No storage reading yet.")
    if sp >= 95:
        return _fail(f"Storage almost full ({sp}%). Free space before recording.")
    return _pass(f"Storage healthy ({sp}% used).")


# check id (used in content.STEPS["verify"]) -> function
CHECKS = {
    "device_online": check_device_online,
    "services_running": check_services_running,
    "wifi_connected": check_wifi_connected,
    "cellular_up": check_cellular_up,
    "camera_ok": check_camera_ok,
    "storage_ok": check_storage_ok,
}


def run_check(check_id: str, device) -> dict:
    fn = CHECKS.get(check_id)
    if not fn:
        return {"ok": False, "status": "fail", "detail": f"Unknown check '{check_id}'."}
    return fn(device)
