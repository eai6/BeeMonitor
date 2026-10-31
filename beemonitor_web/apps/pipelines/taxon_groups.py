"""Insect groups from species names, so tracks can be filtered by what they are
(memory/65).

The species models answer at species rank from small crops, and they are often
wrong in an unhelpful way: an ant comes back as a cockroach, or as one of four
different ants. What a study needs is coarser — bee, wasp, ant — so every name a
crop was given is looked up to its family (GBIF's backbone taxonomy, cached),
mapped to a group, and the crops vote on the group instead of the species.

The pipeline owns the grouping: the Identify Species node's ``group_rules``
holds the groups added by hand and the rules that override the family default,
and ``leave_out`` says which groups are not counted. A track with no clear group
is always counted (the user's call, 2026-10-31): a blurry bee must not be lost.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

BEES, WASPS, ANTS = "Bees", "Wasps", "Ants"
FLIES, BEETLES, MOTHS, CRAWLERS = "Flies", "Beetles", "Moths & butterflies", "Other crawlers"
UNCLEAR = "Unclear"
BUILTIN = [BEES, WASPS, ANTS, FLIES, BEETLES, MOTHS, CRAWLERS]
# Counted by default: bees and wasps (the user's call, 2026-10-31).
DEFAULT_LEAVE_OUT = [ANTS, FLIES, BEETLES, MOTHS, CRAWLERS]
DEFAULT_SHARE = 0.5
# Fewer crops than this can't call a group: the track is unclear, so counted.
MIN_CROPS = 3

BEE_FAMILIES = {"Apidae", "Halictidae", "Megachilidae", "Andrenidae", "Colletidae",
                "Melittidae", "Stenotritidae"}
ORDER_GROUPS = {"Diptera": FLIES, "Coleoptera": BEETLES, "Lepidoptera": MOTHS,
                "Blattodea": CRAWLERS, "Hemiptera": CRAWLERS, "Dermaptera": CRAWLERS,
                "Orthoptera": CRAWLERS, "Isopoda": CRAWLERS}
CRAWLER_CLASSES = {"Arachnida", "Chilopoda", "Diplopoda", "Collembola", "Malacostraca"}
# BeeMachine's coarse non-bee classes, and its "Genus_species" spelling.
BEEMACHINE = {"Wasp": {"order": "Hymenoptera"},
              "Syrphidae": {"order": "Diptera", "family": "Syrphidae"},
              "Diptera": {"order": "Diptera"}, "Coleoptera": {"order": "Coleoptera"},
              "Lepidoptera": {"order": "Lepidoptera"}}

RULE_KINDS = ("track", "name", "genus", "family", "order")
GBIF_URL = "https://api.gbif.org/v1/species/match"
_CACHE_SECONDS = 60 * 60 * 24 * 90


# ── Lineage ─────────────────────────────────────────────────────────────────

def _clean(name) -> str:
    return " ".join(str(name or "").replace("_", " ").split())


def _gbif(name: str) -> dict | None:
    """``{class, order, family, genus}`` for a name, ``{}`` when GBIF has no
    match, or None when GBIF couldn't be asked (not cached, so it's retried)."""
    url = GBIF_URL + "?" + urllib.parse.urlencode({"name": name, "verbose": "false"})
    try:
        with urllib.request.urlopen(url, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - a lookup must never block a count
        logger.warning("GBIF lookup failed for %r: %s", name, exc)
        return None
    if (data.get("matchType") or "NONE") == "NONE":
        return {}
    return {k: data.get(k) or "" for k in ("class", "order", "family", "genus")}


def _cache_key(name) -> str:
    import hashlib

    return "lineage:v1:" + hashlib.sha1(_clean(name).lower().encode()).hexdigest()


def lineages(names) -> dict:
    """``{name: {class, order, family, genus}}`` for every name, ``{}`` for one
    nobody could place. Cached for 90 days; misses are looked up in parallel."""
    from django.core.cache import cache

    out, todo = {}, []
    for name in {n for n in names if n}:
        if name in BEEMACHINE:
            out[name] = dict(BEEMACHINE[name])
            continue
        hit = cache.get(_cache_key(name))
        if hit is not None:
            out[name] = hit
        else:
            todo.append(name)
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            found = list(pool.map(lambda n: _gbif(_clean(n)), todo))
        for name, lin in zip(todo, found):
            if lin is None:
                out[name] = {}
                continue
            if not lin.get("genus"):
                lin["genus"] = _clean(name).split(" ")[0] if " " in _clean(name) else ""
            cache.set(_cache_key(name), lin, _CACHE_SECONDS)
            out[name] = lin
    return out


def default_group(lin: dict) -> str:
    """The group a lineage falls in before any rule: bees by family, ants by
    Formicidae, any other Hymenopteran a wasp, then by order or class."""
    lin = lin or {}
    family, order = lin.get("family") or "", lin.get("order") or ""
    if family == "Formicidae":
        return ANTS
    if family in BEE_FAMILIES:
        return BEES
    if order == "Hymenoptera":
        return WASPS
    if order in ORDER_GROUPS:
        return ORDER_GROUPS[order]
    if (lin.get("class") or "") in CRAWLER_CLASSES:
        return CRAWLERS
    return UNCLEAR


# ── The node's settings ─────────────────────────────────────────────────────

def _json(value, default):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def settings(config) -> dict:
    """The Identify Species node's group settings, parsed:
    ``{on, leave_out, share, custom, rules}``. The editor stores the JSON
    fields as strings; a config saved before groups existed gets the defaults."""
    cfg = config or {}
    rules_cfg = _json(cfg.get("group_rules"), {})
    if not isinstance(rules_cfg, dict):
        rules_cfg = {}
    custom = [str(g).strip() for g in rules_cfg.get("groups") or [] if str(g).strip()]
    rules = [r for r in rules_cfg.get("rules") or []
             if isinstance(r, dict) and r.get("kind") in RULE_KINDS and r.get("group")]
    leave_out = _json(cfg.get("leave_out"), None)
    if not isinstance(leave_out, list):
        leave_out = list(DEFAULT_LEAVE_OUT)
    try:
        share = float(cfg.get("agree_share") if cfg.get("agree_share") not in (None, "")
                      else DEFAULT_SHARE)
    except (TypeError, ValueError):
        share = DEFAULT_SHARE
    if share > 1:
        share /= 100.0  # typed as a percentage
    return {
        "on": str(cfg.get("group_filter") or "on").lower() != "off",
        "leave_out": [g for g in leave_out if g and g != UNCLEAR],
        "share": min(max(share, 0.0), 1.0),
        "custom": custom,
        "rules": rules,
    }


def all_groups(custom=()) -> list:
    return BUILTIN + [g for g in custom if g not in BUILTIN and g != UNCLEAR]


def _rule_index(rules):
    index = {}
    for r in rules:
        if r["kind"] == "track":
            key = ("track", str(r.get("job") or ""), str(r.get("track") or ""))
        else:
            key = (r["kind"], str(r.get("value") or "").strip().lower())
        index[key] = r["group"]
    return index


def name_group(name, lin, index) -> tuple[str, str]:
    """``(group, why)`` for one species name: a rule on the name, its genus,
    family or order (most specific first), else the family default."""
    for kind, value in (("name", name), ("genus", (lin or {}).get("genus")),
                        ("family", (lin or {}).get("family")),
                        ("order", (lin or {}).get("order"))):
        if value and (kind, str(value).strip().lower()) in index:
            return index[(kind, str(value).strip().lower())], f"rule on {kind} {value}"
    return default_group(lin), "default"


# ── Tracks ──────────────────────────────────────────────────────────────────

def _tid(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def crop_readings(result, tracking_df=None) -> dict:
    """``{track_id: [species name, ...]}``, one name per crop read.

    From the per-crop votes the worker saved (``track_votes.csv``); a run
    without that file falls back to each track's voted species, repeated as
    many times as it won votes, so a firm call still reads as firm."""
    from . import ops

    stats = (result or {}).get("summary_stats") or {}
    votes = ops._read_csv(stats.get("track_votes_csv") or "")
    out: dict = {}
    if votes is not None and len(votes) and "species" in votes.columns:
        for tid, name in zip(votes["track_id"], votes["species"]):
            t = _tid(tid)
            if t is not None and isinstance(name, str) and name.strip():
                out.setdefault(t, []).append(name.strip())
        return out
    df = tracking_df
    if df is None or not len(df):
        return out
    id_col = ops._pick(df, ops._ID_COLS)
    if id_col is None or "taxon" not in df.columns or "taxon_votes" not in df.columns:
        return out
    for tid, grp in df.groupby(id_col):
        t = _tid(tid)
        names = grp["taxon"].dropna()
        votes_n = int(grp["taxon_votes"].fillna(0).max() or 0)
        if t is None or names.empty or votes_n <= 0:
            continue
        crops = int(grp["taxon_crops"].fillna(0).max() or votes_n) if "taxon_crops" in grp else votes_n
        out[t] = [str(names.iloc[0])] * votes_n + [""] * max(crops - votes_n, 0)
    return out


def track_groups(readings: dict, cfg: dict, job_id=None) -> dict:
    """``{track_id: {group, share, crops, said, top, family, why, counted}}``.

    Each crop's name becomes a group (rules first), the crops vote, and the
    track's group is called when at least ``share`` of its crops agree and
    ``MIN_CROPS`` were read; otherwise it is unclear. A rule on the track
    itself decides outright. ``counted`` is whether the filter keeps it."""
    index = _rule_index(cfg["rules"])
    names = {n for crops in readings.values() for n in crops if n}
    lin = lineages(names)
    per_name = {n: name_group(n, lin.get(n), index) for n in names}
    leave_out = set(cfg["leave_out"]) if cfg["on"] else set()

    out = {}
    for tid, crops in readings.items():
        tally, species = {}, {}
        for n in crops:
            g = per_name[n][0] if n else UNCLEAR
            tally[g] = tally.get(g, 0) + 1
            if n:
                species[n] = species.get(n, 0) + 1
        read = len(crops)
        named = {g: c for g, c in tally.items() if g != UNCLEAR}
        best, votes = (max(named.items(), key=lambda kv: kv[1]) if named else (UNCLEAR, 0))
        share = round(votes / read, 3) if read else 0.0
        group = best if read >= MIN_CROPS and share >= cfg["share"] and best != UNCLEAR else UNCLEAR
        top = max(species.items(), key=lambda kv: kv[1])[0] if species else ""
        why = "crops"
        if read < MIN_CROPS:
            why = f"only {read} crop{'s' if read != 1 else ''}"
        elif group == UNCLEAR and named:
            why = f"{round(share * 100)}% {best.lower()}"
        rule = index.get(("track", str(job_id or ""), str(tid)))
        if rule:
            group, why = rule, "set by hand"
        out[tid] = {
            "group": group, "share": share, "crops": read,
            "said": ", ".join(f"{c} {g.lower()}" for g, c in
                              sorted(tally.items(), key=lambda kv: -kv[1])),
            "top": top, "family": (lin.get(top) or {}).get("family", "") if top else "",
            "why": why,
            "counted": group not in leave_out,
        }
    return out


def names_seen(readings: dict) -> list:
    return sorted({n for crops in readings.values() for n in crops if n})


# ── Changing a rule ─────────────────────────────────────────────────────────

def with_rule(config: dict, rule: dict) -> dict:
    """A copy of the node's config with ``rule`` in its rules (replacing one on
    the same thing). An empty group removes the rule. A group not yet known is
    added to the node's own groups."""
    cfg = dict(config or {})
    rules_cfg = _json(cfg.get("group_rules"), {})
    if not isinstance(rules_cfg, dict):
        rules_cfg = {}
    groups = list(rules_cfg.get("groups") or [])
    rules = [r for r in rules_cfg.get("rules") or [] if isinstance(r, dict)]

    def same(r):
        if r.get("kind") != rule["kind"]:
            return False
        if rule["kind"] == "track":
            return (str(r.get("job")), str(r.get("track"))) == (str(rule.get("job")),
                                                                 str(rule.get("track")))
        return str(r.get("value", "")).strip().lower() == str(rule.get("value", "")).strip().lower()

    rules = [r for r in rules if not same(r)]
    if rule.get("group"):
        rules.append(rule)
        if rule["group"] not in BUILTIN and rule["group"] != UNCLEAR and rule["group"] not in groups:
            groups.append(rule["group"])
    cfg["group_rules"] = json.dumps({"groups": groups, "rules": rules})
    return cfg
