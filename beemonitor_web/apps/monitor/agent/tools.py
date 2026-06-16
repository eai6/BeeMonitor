"""Read-only tools for the taxonomic monitoring agent (Phase 3).

Every tool is read-only and scoped to devices the requesting user can access
(via ``Device.accessible``) — the agent reads the structured Observations +
telemetry/GPS, it never changes state. Mirrors ``apps/setup/assistant/tools.py``.
"""

from django.conf import settings
from django.db.models import Avg, Count, Max, Sum
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
        "name": "device_config",
        "description": "The REMOTE CONFIG the dashboard has pushed to a device — "
                       "telemetry beat rate, bee-confirmation mode, cellular crop "
                       "daily cap, motion-tuning overrides, hotel ROI + nest-tube "
                       "layout, WittyPi wake schedule (and whether it's actually "
                       "applied), and display timezone. Use to explain how a unit is "
                       "currently set up, why it might be sending little/no data "
                       "(e.g. crop cap 0, gate mode, slow beat), or what a setting "
                       "does. Read-only — it reports settings, it can't change them. "
                       "Omit device_id to get the current config of EVERY accessible "
                       "device at once (the whole fleet's state); pass one to scope "
                       "to a single device.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer",
                              "description": "Optional — omit for all devices."},
            },
        },
    },
    {
        "name": "wake_window_status",
        "description": "Whether a device is, RIGHT NOW, inside its configured wake "
                       "window — resolves the unit's local time and today's window "
                       "(daylight sunrise/sunset, fixed on/off, etc.). Use this to "
                       "tell whether an OFFLINE unit is simply ASLEEP as scheduled "
                       "(expected, no action) versus genuinely down (a fault). "
                       "Returns true/false/'unknown' (unknown = no GPS for a daylight "
                       "schedule, or a cyclic interval whose phase isn't tracked).",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "integer"}},
            "required": ["device_id"],
        },
    },
    {
        "name": "tracking_summary",
        "description": "Processed TRACKING results for a device — foraging trips, "
                       "interactions, unique tracks, entries/exits — aggregated "
                       "across its analyzed videos. NOTE: this data only exists "
                       "after analysis is run on the videos (the Processing page); "
                       "the tool says so if nothing has been analyzed yet.",
        "input_schema": {
            "type": "object",
            "properties": {"device_id": {"type": "integer"}},
            "required": ["device_id"],
        },
    },
    {
        "name": "weather",
        "description": "Recent weather at a device's GPS location (daily highs/lows "
                       "+ precipitation, and the last 24h hourly) — to correlate "
                       "activity/foraging with conditions. Needs the device to have "
                       "a location set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "device_id": {"type": "integer"},
                "days": {"type": "integer", "description": "Look-back days (1-14)."},
            },
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


def _device_config(device) -> dict:
    """The remote config the dashboard has pushed to one device (+ online status).

    Delegates to ``Device.remote_config_summary`` (the single source of truth,
    shared with the device support assistant) and adds live online status.
    """
    return {**device.remote_config_summary(), **_online(device)}


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

        if name == "device_config":
            # Optional device_id: scope to one device, or report the whole fleet.
            did = tool_input.get("device_id")
            if did is not None:
                device = _device_for(user, did)
                if device is None:
                    return {"error": "No such device, or you don't have access to it."}
                return _device_config(device)
            return {"devices": [
                _device_config(d) for d in Device.accessible(user).order_by("name")
            ]}

        # Everything below is device-scoped.
        device = _device_for(user, tool_input.get("device_id"))
        if device is None:
            return {"error": "No such device, or you don't have access to it."}

        if name == "device_context":
            return {"device_id": device.pk, "name": device.name,
                    "location": device.location, "lat": device.lat, "lon": device.lon,
                    "tz": device.tz_name, **_online(device)}

        if name == "wake_window_status":
            from apps.devices.wake_status import wake_window_status
            return {**wake_window_status(device), **_online(device)}

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

        if name == "tracking_summary":
            from apps.analysis.models import JobResult
            from apps.videos.models import Video
            results = JobResult.objects.filter(
                job__video__device=device, job__status="completed")
            analyzed = (Video.objects.filter(device=device, jobs__status="completed")
                        .distinct().count())
            total_videos = Video.objects.filter(device=device).count()
            if analyzed == 0:
                return {"device_id": device.pk, "analyzed_videos": 0,
                        "total_videos": total_videos,
                        "note": "No processed tracking data yet for this device. "
                                "Tracking, foraging trips and interactions are "
                                "produced by running analysis on the device's "
                                "videos (the Processing page) — not from the device "
                                f"itself. {total_videos} video(s) uploaded, 0 analyzed."}
            agg = results.aggregate(
                foraging_trips=Sum("foraging_trip_count"),
                interactions=Sum("interaction_count"),
                events=Sum("total_events"), entries=Sum("entry_count"),
                exits=Sum("exit_count"), unique_tracks=Sum("unique_tracks"),
                nests=Sum("nest_count"), avg_trip_duration_sec=Avg("avg_trip_duration_sec"))
            totals = {k: (round(v, 1) if isinstance(v, float) else v)
                      for k, v in agg.items()}
            return {"device_id": device.pk, "analyzed_videos": analyzed,
                    "total_videos": total_videos, "totals": totals}

        if name == "weather":
            if device.lat is None or device.lon is None:
                return {"device_id": device.pk,
                        "error": "No GPS set for this device, so weather isn't "
                                 "available. Set its location to enable weather."}
            from apps.analysis.views import _fetch_weather_data
            days = max(1, min(int(tool_input.get("days", 3)), 14))
            end = timezone.now().date()
            start = end - timezone.timedelta(days=days)
            wx = _fetch_weather_data(start.isoformat(), end.isoformat(),
                                     device.lat, device.lon)
            return {"device_id": device.pk, "lat": device.lat, "lon": device.lon,
                    "days": days, "daily": wx.get("daily", []),
                    "recent_hourly": (wx.get("hourly") or [])[-24:]}

        if name == "region_species":
            taxa = region_taxa(device.lat, device.lon)
            return {"device_id": device.pk, "lat": device.lat, "lon": device.lon,
                    "expected_species": taxa[:100], "count": len(taxa)}

        return {"error": f"Unknown tool '{name}'."}
    except Exception as e:  # never let a tool error kill the turn
        return {"error": f"Tool failed: {e}"}
