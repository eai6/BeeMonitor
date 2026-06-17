"""Daily digest generation for the monitoring agent (Phase 3).

For one device and one UTC day, gather the structured signal (taxa seen, which
are new for the site, activity count, telemetry anomalies), then have the agent
narrate it into a short paragraph — falling back to a deterministic sentence when
no ANTHROPIC_API_KEY is set. Idempotent per (device, date). The scheduled
``manage.py generate_digests`` command drives this. See memory/15 §12.3.
"""

import json
import logging
from datetime import datetime, time, timedelta, timezone as dt_timezone

from django.conf import settings
from django.db.models import Count, Sum

from ..models import Activity, DailyDigest, Observation
from .client import is_enabled

logger = logging.getLogger(__name__)

_DIGEST_SYSTEM = """You are the BeeMonitor monitoring agent writing a short daily \
digest for ONE hive from the JSON data provided. 2-4 sentences, plain prose: which \
taxa visited and how often, call out any new-for-site taxa explicitly, and flag any \
anomalies. Ground strictly in the data — never invent a species, count, or event. \
If there was no activity, say so briefly. Lead with the headline."""


def _window(day):
    start = datetime.combine(day, time.min, tzinfo=dt_timezone.utc)
    return start, start + timedelta(days=1)


def _anomalies(device, activity_count) -> list:
    out = []
    hb = device.heartbeats.first()
    m = (hb.metrics or {}) if hb else {}
    if not device.is_online():  # window scales to the device's beat cadence
        out.append("device offline (no recent check-in)")
    if hb and m.get("recorder_active") is False:
        out.append("recorder not running")
    sp = m.get("storage_pct")
    if isinstance(sp, (int, float)) and sp >= 90:
        out.append(f"storage {int(sp)}% full")
    if activity_count == 0:
        out.append("no activity recorded")
    return out


def compute_digest_data(device, day) -> dict:
    """Structured signal for one device/day (the data the summary is built from)."""
    start, end = _window(day)
    obs = Observation.objects.filter(
        activity__device=device, activity__started_at__gte=start,
        activity__started_at__lt=end, taxon__isnull=False)
    taxa = [
        {"taxon_id": r["taxon_id"], "taxon": r["taxon__name"],
         "common_name": r["taxon__common_name"], "activities": r["activities"],
         "individuals": r["individuals"]}
        for r in obs.values("taxon_id", "taxon__name", "taxon__common_name")
        .annotate(activities=Count("activity", distinct=True),
                  individuals=Sum("individual_count"))
        .order_by("-activities")
    ]
    # New-for-site: a taxon with no observation on this device before the window.
    prior = set(Observation.objects.filter(
        activity__device=device, activity__started_at__lt=start, taxon__isnull=False)
        .values_list("taxon_id", flat=True))
    new_for_site = [t["taxon"] for t in taxa if t["taxon_id"] not in prior]
    activity_count = Activity.objects.filter(
        device=device, started_at__gte=start, started_at__lt=end).count()
    return {
        "device": device.name, "date": day.isoformat(),
        "activity_count": activity_count,
        "taxa": [{k: t[k] for k in ("taxon", "common_name", "activities", "individuals")}
                 for t in taxa],
        "new_for_site": new_for_site,
        "anomalies": _anomalies(device, activity_count),
    }


def _fallback_text(data) -> str:
    """Deterministic summary when the LLM isn't configured."""
    n = data["activity_count"]
    parts = [f"{data['device']} recorded {n} "
             f"{'activity' if n == 1 else 'activities'} on {data['date']}."]
    if data["taxa"]:
        top = ", ".join(f"{t['taxon']} ({t['activities']})" for t in data["taxa"][:5])
        parts.append(f"Taxa: {top}.")
    if data["new_for_site"]:
        parts.append("New for site: " + ", ".join(data["new_for_site"]) + ".")
    if data["anomalies"]:
        parts.append("Anomalies: " + "; ".join(data["anomalies"]) + ".")
    return " ".join(parts)


def narrate(data) -> str:
    if not is_enabled():
        return _fallback_text(data)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model=settings.ASSISTANT_MODEL,
            max_tokens=min(settings.ASSISTANT_MAX_TOKENS, 600),
            system=_DIGEST_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(data)}],
        )
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
        return text or _fallback_text(data)
    except Exception as e:  # pragma: no cover - API hiccup must not block the row
        logger.warning("digest narrate failed for %s: %s", data.get("device"), e)
        return _fallback_text(data)


def generate_digest(device, day) -> DailyDigest:
    """Compute + narrate + upsert the digest for one device/day."""
    data = compute_digest_data(device, day)
    summary = narrate(data)
    digest, _ = DailyDigest.objects.update_or_create(
        device=device, date=day, defaults={"summary": summary, "stats": data})
    return digest
