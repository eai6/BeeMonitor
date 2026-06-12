"""Location priors for BioCLIP (Phase 2).

``region_taxa(lat, lon, month)`` returns the scientific names of insects that
occur near a location, so BioCLIP zero-shot can be constrained to plausible
candidates instead of the whole Tree of Life — the single biggest accuracy
lever (see ``memory/15_monitoring_agent_design.md`` §11).

Source: iNaturalist ``species_counts`` (research-grade, ranked by local
observation frequency); GBIF occurrence search as a fallback. Cached in the
Django cache (fauna lists drift slowly). Returns ``[]`` on any failure — the
pipeline then falls back to unconstrained ToL, so an ID is never blocked.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from django.conf import settings
from django.core.cache import cache

logger = logging.getLogger(__name__)

INAT_INSECTA_TAXON_ID = 47158   # iNaturalist taxon id for class Insecta
GBIF_INSECTA_TAXON_KEY = 216    # GBIF taxonKey for class Insecta


def region_taxa(lat, lon, month: "int | None" = None) -> list:
    """Scientific names of insects near (lat, lon), most-frequent first.

    ``month`` (1-12) adds a phenology filter. Cached; ``[]`` if there's no
    location or both sources fail.
    """
    if lat is None or lon is None:
        return []
    radius = getattr(settings, "MONITOR_PRIOR_RADIUS_KM", 50)
    cap = getattr(settings, "MONITOR_PRIOR_MAX_TAXA", 300)
    key = f"priors:{lat:.2f}:{lon:.2f}:{month or 'all'}:{radius}"
    cached = cache.get(key)
    if cached is not None:
        return cached

    taxa = _inat_species(lat, lon, radius, month, cap) or _gbif_species(lat, lon, radius, cap)
    cache.set(key, taxa, getattr(settings, "MONITOR_PRIOR_TTL_SECONDS", 60 * 60 * 24 * 30))
    return taxa


def _inat_species(lat, lon, radius, month, cap) -> list:
    params = {
        "lat": f"{lat:.4f}", "lng": f"{lon:.4f}", "radius": radius,
        "taxon_id": INAT_INSECTA_TAXON_ID, "quality_grade": "research",
        "rank": "species", "per_page": min(500, cap),
    }
    if month:
        params["month"] = int(month)
    url = "https://api.inaturalist.org/v1/observations/species_counts?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # pragma: no cover - network hiccup must not 500
        logger.warning("iNat species_counts failed (%.3f,%.3f): %s", lat, lon, e)
        return []
    out = []
    for r in data.get("results", []):
        name = ((r.get("taxon") or {}).get("name") or "").strip()
        if name:
            out.append(name)
        if len(out) >= cap:
            break
    return out


def _gbif_species(lat, lon, radius, cap) -> list:
    """Fallback: distinct species from GBIF occurrence records in a bbox."""
    d = max(0.05, radius / 111.0)  # km -> degrees (rough)
    params = {
        "taxonKey": GBIF_INSECTA_TAXON_KEY,
        "decimalLatitude": f"{lat - d:.4f},{lat + d:.4f}",
        "decimalLongitude": f"{lon - d:.4f},{lon + d:.4f}",
        "hasCoordinate": "true",
        "limit": min(300, cap),
    }
    url = "https://api.gbif.org/v1/occurrence/search?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # pragma: no cover - network hiccup must not 500
        logger.warning("GBIF occurrence search failed (%.3f,%.3f): %s", lat, lon, e)
        return []
    seen, out = set(), []
    for r in data.get("results", []):
        name = (r.get("species") or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
        if len(out) >= cap:
            break
    return out
