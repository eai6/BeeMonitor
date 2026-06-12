"""Read-only tools for the taxonomic monitoring agent (Phase 3).

Every tool is read-only and scoped to devices the requesting user can access
(via ``Device.accessible``) — the agent reads the structured Observations +
telemetry/GPS, it never changes state. Mirrors ``apps/setup/assistant/tools.py``.
"""

from django.conf import settings
from django.db.models import Count, Max, Sum
from django.utils import timezone

from apps.devices.models import Device

from ..models import Activity, Observation
from ..priors import region_taxa

TOOL_DEFS = [
    {
        "name": "list_devices",
        "description": "List the devices (hives) the user can access — id, name, "
                       "location, GPS, and online status. Call this first to know "
                       "which device_id to use.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "species_summary",
        "description": "Taxa observed at a device over the last N days, with how "
                       "many activities each appeared in and when last seen. The "
                       "go-to tool for 'what visited site X this week?'.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer"},
                "days": {"type": "integer", "description": "Look-back window (1-90)."},
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "recent_activities",
        "description": "Recent motion events for a device, newest first: id, time, "
                       "best taxon + confidence, status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer"},
                "limit": {"type": "integer", "description": "How many (1-30)."},
            },
            "required": ["device_id"],
        },
    },
    {
        "name": "get_activity",
        "description": "One activity in detail: its observations (taxon, count, "
                       "confidence), frame count, peak motion, GPS, and time.",
        "input_schema": {
            "type": "object",
            "properties": {"activity_id": {"type": "integer"}},
            "required": ["activity_id"],
        },
    },
    {
        "name": "device_context",
        "description": "A device's deployment context: name, location label, GPS, "
                       "online status, last check-in, storage. Use to ground "
                       "answers in where/how the hive is deployed.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "integer"}},
            "required": ["device_id"],
        },
    },
    {
        "name": "region_species",
        "description": "Insects KNOWN TO OCCUR near a device's location (the "
                       "iNaturalist/GBIF prior used to constrain BioCLIP). Use to "
                       "sanity-check whether an identified species is plausible "
                       "for the site, or what could be expected there.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "integer"}},
            "required": ["device_id"],
        },
    },
]


def _device_for(user, device_id):
    return Device.accessible(user).filter(pk=device_id).first()


def _online(device) -> "dict":
    age = None
    online = False
    if device.last_seen_at:
        age = int((timezone.now() - device.last_seen_at).total_seconds())
        online = age <= settings.DEVICE_ONLINE_GRACE_SECONDS
    return {"online": online, "seconds_since_last_seen": age}


def run_tool(name: str, tool_input: dict, user) -> dict:
    """Execute a tool by name; returns a JSON-serialisable dict."""
    try:
        if name == "list_devices":
            return {"devices": [
                {"device_id": d.pk, "name": d.name, "location": d.location,
                 "lat": d.lat, "lon": d.lon, **_online(d)}
                for d in Device.accessible(user).order_by("name")
            ]}

        if name == "get_activity":
            act = (Activity.objects
                   .filter(device__in=Device.accessible(user))
                   .select_related("device", "best_taxon")
                   .filter(pk=tool_input.get("activity_id")).first())
            if act is None:
                return {"error": "No such activity, or you don't have access."}
            obs = [{"taxon": o.taxon.name if o.taxon else None,
                    "common_name": o.taxon.common_name if o.taxon else None,
                    "individuals": o.individual_count,
                    "confidence": o.confidence}
                   for o in act.observations.select_related("taxon")]
            return {
                "activity_id": act.pk, "device": act.device.name,
                "started_at": act.started_at.isoformat() if act.started_at else None,
                "status": act.status, "peak_motion": act.peak_motion,
                "lat": act.lat, "lon": act.lon, "frame_count": act.frames.count(),
                "observations": obs,
            }

        # Everything below is device-scoped.
        device = _device_for(user, tool_input.get("device_id"))
        if device is None:
            return {"error": "No such device, or you don't have access to it."}

        if name == "device_context":
            return {"device_id": device.pk, "name": device.name,
                    "location": device.location, "lat": device.lat, "lon": device.lon,
                    "tz": device.tz_name, **_online(device)}

        if name == "species_summary":
            days = max(1, min(int(tool_input.get("days", 7)), 90))
            since = timezone.now() - timezone.timedelta(days=days)
            rows = (Observation.objects
                    .filter(activity__device=device, activity__started_at__gte=since,
                            taxon__isnull=False)
                    .values("taxon__name", "taxon__common_name")
                    .annotate(activities=Count("activity", distinct=True),
                              individuals=Sum("individual_count"),
                              last_seen=Max("activity__started_at"))
                    .order_by("-activities"))
            return {"device_id": device.pk, "days": days, "taxa": [
                {"taxon": r["taxon__name"], "common_name": r["taxon__common_name"],
                 "activities": r["activities"], "individuals": r["individuals"],
                 "last_seen": r["last_seen"].isoformat() if r["last_seen"] else None}
                for r in rows
            ]}

        if name == "recent_activities":
            limit = max(1, min(int(tool_input.get("limit", 10)), 30))
            acts = (device.activities.select_related("best_taxon")[:limit])
            return {"device_id": device.pk, "activities": [
                {"activity_id": a.pk,
                 "started_at": a.started_at.isoformat() if a.started_at else None,
                 "best_taxon": a.best_taxon.name if a.best_taxon else None,
                 "confidence": a.best_confidence, "status": a.status}
                for a in acts
            ]}

        if name == "region_species":
            taxa = region_taxa(device.lat, device.lon)
            return {"device_id": device.pk, "lat": device.lat, "lon": device.lon,
                    "expected_species": taxa[:100], "count": len(taxa)}

        return {"error": f"Unknown tool '{name}'."}
    except Exception as e:  # never let a tool error kill the turn
        return {"error": f"Tool failed: {e}"}
