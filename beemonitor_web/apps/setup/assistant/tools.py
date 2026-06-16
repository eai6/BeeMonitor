"""Read-only tools the assistant can call to debug *with* the user.

Every tool is read-only and scoped to devices the requesting user can access —
the assistant can look, never touch. Anything that would change device state is
surfaced to the user as text for explicit confirmation (see safety.classify_command),
never executed here.
"""

from django.conf import settings
from django.utils import timezone

from apps.devices.models import Device

from .. import content

# Tool schemas sent to the Claude API.
TOOL_DEFS = [
    {
        "name": "get_device_status",
        "description": "Current health of one device: online/offline, seconds "
                       "since last check-in, recorder/uploader/cellular service "
                       "state, WiFi, storage %, and deployed code commit. Use "
                       "this first when a user reports a problem.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer", "description": "The device's numeric id."},
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "get_recent_heartbeats",
        "description": "The last N telemetry beats (timestamp + key metrics) for "
                       "a device, newest first. Use to spot when something "
                       "changed or stopped.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer"},
                "limit": {"type": "integer", "description": "How many beats (1-20)."},
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "get_device_config",
        "description": "The REMOTE CONFIG the dashboard has pushed to a device — "
                       "telemetry beat rate, bee-confirmation mode, cellular crop "
                       "daily cap, motion-tuning overrides, hotel ROI + nest-tube "
                       "layout, WittyPi wake schedule (and whether it's actually "
                       "applied to hardware), and display timezone. Use to explain "
                       "how a unit is set up or to debug why it's sending little/no "
                       "data (e.g. crop cap 0, gate mode, a slow beat, report-only "
                       "schedule). Omit device_id to get EVERY accessible device's "
                       "config at once; pass one to scope to a single device. "
                       "Read-only — it reports settings, it can't change them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer",
                              "description": "Optional — omit for all devices."},
            },
        },
    },
    {
        "name": "lookup_troubleshooting",
        "description": "Search the BeeMonitor setup guide for steps and known "
                       "error fixes matching a symptom or keyword. Use to ground "
                       "advice in the official docs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Symptom or keyword."},
            },
            "required": ["query"],
        },
    },
]


def _device_for(user, device_id):
    return Device.accessible(user).filter(pk=device_id).first()


def _status_payload(device) -> dict:
    hb = device.heartbeats.first()
    m = (hb.metrics or {}) if hb else {}
    online = False
    age = None
    if device.last_seen_at:
        age = int((timezone.now() - device.last_seen_at).total_seconds())
        online = age <= settings.DEVICE_ONLINE_GRACE_SECONDS
    return {
        "device_id": device.pk,
        "name": device.name,
        "online": online,
        "seconds_since_last_seen": age,
        "recorder_active": m.get("recorder_active"),
        "uploader_active": m.get("uploader_active"),
        "cellular_active": m.get("cellular_active"),
        "wifi_state": m.get("wifi_state"),
        "wifi_ssid": m.get("wifi_ssid"),
        "storage_pct": m.get("storage_pct"),
        "code_commit": m.get("code_commit"),
        "last_error": m.get("update", {}).get("error") if isinstance(m.get("update"), dict) else None,
    }


def _config_payload(device) -> dict:
    """Remote config for one device + live online status."""
    online = False
    age = None
    if device.last_seen_at:
        age = int((timezone.now() - device.last_seen_at).total_seconds())
        online = age <= settings.DEVICE_ONLINE_GRACE_SECONDS
    return {**device.remote_config_summary(),
            "online": online, "seconds_since_last_seen": age}


def run_tool(name: str, tool_input: dict, user) -> dict:
    """Execute a tool by name; returns a JSON-serialisable dict."""
    try:
        if name == "get_device_config":
            # Optional device_id: scope to one device, or report the whole fleet.
            did = tool_input.get("device_id")
            if did is not None:
                device = _device_for(user, did)
                if device is None:
                    return {"error": "No such device, or you don't have access to it."}
                return _config_payload(device)
            return {"devices": [
                _config_payload(d) for d in Device.accessible(user).order_by("name")
            ]}

        if name in ("get_device_status", "get_recent_heartbeats"):
            device = _device_for(user, tool_input.get("device_id"))
            if device is None:
                return {"error": "No such device, or you don't have access to it."}
            if name == "get_device_status":
                return _status_payload(device)
            limit = max(1, min(int(tool_input.get("limit", 5)), 20))
            beats = []
            for hb in device.heartbeats.all()[:limit]:
                m = hb.metrics or {}
                beats.append({
                    "at": hb.created_at.isoformat(),
                    "recorder_active": m.get("recorder_active"),
                    "uploader_active": m.get("uploader_active"),
                    "cellular_active": m.get("cellular_active"),
                    "wifi_ssid": m.get("wifi_ssid"),
                    "storage_pct": m.get("storage_pct"),
                    "snippets_last_period": m.get("snippets_last_period"),
                })
            return {"device_id": device.pk, "heartbeats": beats}

        if name == "lookup_troubleshooting":
            q = (tool_input.get("query") or "").lower()
            hits = []
            for st in content.STEPS:
                hay = (st["title"] + " " + st["concept"] + " " +
                       " ".join(ce["symptom"] + " " + ce["fix"]
                                for ce in st["common_errors"])).lower()
                if q and q in hay:
                    hits.append({
                        "step": st["title"],
                        "concept": st["concept"],
                        "fixes": st["common_errors"],
                    })
            return {"query": tool_input.get("query"), "matches": hits[:5]}

        return {"error": f"Unknown tool '{name}'."}
    except Exception as e:  # never let a tool error kill the turn
        return {"error": f"Tool failed: {e}"}
