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
    match, or None when GBIF couldn't be asked (not cached, so it's retried).
    One retry: a burst of lookups for a whole batch can be throttled."""
    import time

    url = GBIF_URL + "?" + urllib.parse.urlencode({"name": name, "verbose": "false"})
    for attempt in range(2):
        try:
            with urllib.request.urlopen(url, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001 - a lookup must never block a count
            if attempt:
                logger.warning("GBIF lookup failed for %r: %s", name, exc)
                return None
            time.sleep(0.5)
    if (data.get("matchType") or "NONE") == "NONE":
        return {}
    return {k: data.get(k) or "" for k in ("class", "order", "family", "genus")}


def _bee_genera() -> frozenset:
    """Every bee genus BeeMachine knows (354 taxa), so a bee is a bee even
    when GBIF can't be asked or doesn't know the name."""
    global _BEE_GENERA
    if _BEE_GENERA is None:
        try:
            from beemonitor.identification.species import NON_BEE_TAXA, taxa
            _BEE_GENERA = frozenset(t.split("_")[0] for t in taxa() if t not in NON_BEE_TAXA)
        except Exception:  # noqa: BLE001 - the GBIF answer still works without it
            logger.warning("BeeMachine taxa unavailable; bee genera come from GBIF only")
            _BEE_GENERA = frozenset()
    return _BEE_GENERA


_BEE_GENERA = None


def _known_bee(name: str) -> dict | None:
    genus = _clean(name).split(" ")[0]
    if genus in _bee_genera():
        return {"class": "Insecta", "order": "Hymenoptera", "family": "", "genus": genus,
                "bee": True}
    return None


def _cache_key(name) -> str:
    import hashlib

    return "lineage:v1:" + hashlib.sha1(_clean(name).lower().encode()).hexdigest()


def lineages(names, budget=None):
    """``{name: {class, order, family, genus}}`` for every name, ``{}`` for one
    nobody could place. Cached for 90 days; misses are looked up in parallel.

    With ``budget``, at most that many misses are looked up and the call
    returns ``(lineages, pending)``: names left for later are absent from
    ``lineages`` and counted in ``pending``."""
    from django.core.cache import cache

    out, todo = {}, []
    for name in {n for n in names if n}:
        if name in BEEMACHINE:
            out[name] = dict(BEEMACHINE[name])
            continue
        hit = cache.get(_cache_key(name))
        if hit is not None:
            # A "no match" cached before bee genera were known still says bee.
            out[name] = hit or _known_bee(name) or {}
        else:
            todo.append(name)
    left = []
    if budget is not None and len(todo) > budget:
        todo, left = todo[:budget], todo[budget:]
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            found = list(pool.map(lambda n: _gbif(_clean(n)), todo))
        for name, lin in zip(todo, found):
            if not lin:
                # GBIF couldn't be asked or doesn't know it: a known bee genus
                # still says bee. Not cached when GBIF failed, so it's retried.
                out[name] = _known_bee(name) or {}
                if lin is None:
                    continue
                lin = out[name]
            if not lin.get("genus"):
                lin["genus"] = _clean(name).split(" ")[0] if " " in _clean(name) else ""
            cache.set(_cache_key(name), lin, _CACHE_SECONDS)
            out[name] = lin
    if budget is not None:
        return out, len(left)
    return out


def default_group(lin: dict) -> str:
    """The group a lineage falls in under the default grouping."""
    return name_group("", lin, _index(default_grouping()))[0]


# ── The grouping (memory/66) ────────────────────────────────────────────────
# A grouping is data the user edits: each group is a list of taxa at any rank,
# and the most specific match wins, so "Ants = family Formicidae" beats
# "Wasps = order Hymenoptera" for an ant. Track rules (set while reviewing)
# decide one track outright.

RANKS = ("species", "genus", "family", "order", "class")   # most specific first
GROUP_COLORS = {BEES: "#d97706", WASPS: "#dc2626", ANTS: "#2563eb", FLIES: "#0d9488",
                BEETLES: "#65a30d", MOTHS: "#db2777", CRAWLERS: "#6b7280"}
_PALETTE = ["#7c3aed", "#0891b2", "#ca8a04", "#be185d", "#4d7c0f", "#9333ea", "#0f766e"]


def default_grouping() -> dict:
    """The built-in groups, written out so they can be seen and changed."""
    def members(rank, names):
        return [{"rank": rank, "name": n} for n in sorted(names)]

    crawler_orders = [o for o, g in ORDER_GROUPS.items() if g == CRAWLERS]
    groups = [
        (BEES, members("family", BEE_FAMILIES)),
        (WASPS, members("order", ["Hymenoptera"])),
        (ANTS, members("family", ["Formicidae"])),
        (FLIES, members("order", ["Diptera"])),
        (BEETLES, members("order", ["Coleoptera"])),
        (MOTHS, members("order", ["Lepidoptera"])),
        (CRAWLERS, members("order", crawler_orders) + members("class", CRAWLER_CLASSES)),
    ]
    return {"groups": [{"name": n, "color": GROUP_COLORS[n], "members": m} for n, m in groups],
            "tracks": []}


def _key(name) -> str:
    return _clean(name).lower()


def _ensure_group(grouping, name):
    for g in grouping["groups"]:
        if g["name"] == name:
            return g
    g = {"name": name, "color": _PALETTE[len(grouping["groups"]) % len(_PALETTE)], "members": []}
    grouping["groups"].append(g)
    return g


def put_member(grouping, rank, name, group):
    """Put a taxon in ``group`` (taking it out of any other); an empty group
    just takes it out."""
    for g in grouping["groups"]:
        g["members"] = [m for m in g["members"]
                        if not (m["rank"] == rank and _key(m["name"]) == _key(name))]
    if group and group != UNCLEAR:
        _ensure_group(grouping, group)["members"].append({"rank": rank, "name": _clean(name)})
    return grouping


def _valid(grouping) -> dict:
    groups, seen = [], set()
    for g in grouping.get("groups") or []:
        name = " ".join(str((g or {}).get("name") or "").split())[:60]
        if not name or name == UNCLEAR or name in seen:
            continue
        seen.add(name)
        members = [{"rank": m["rank"], "name": _clean(m.get("name"))}
                   for m in g.get("members") or []
                   if isinstance(m, dict) and m.get("rank") in RANKS and _clean(m.get("name"))]
        groups.append({"name": name, "color": str(g.get("color") or "#6b7280")[:9],
                       "members": members})
    tracks = [{"job": t.get("job"), "track": t.get("track"), "group": t.get("group")}
              for t in grouping.get("tracks") or []
              if isinstance(t, dict) and t.get("group") and t.get("track") is not None]
    return {"groups": groups, "tracks": tracks}


def grouping_of(config) -> dict:
    """The node's grouping. A config saved before groupings were explicit is
    converted: the defaults, plus its own groups and rules (memory/65), so it
    counts exactly as it did."""
    cfg = config or {}
    explicit = _json(cfg.get("grouping"), None)
    if isinstance(explicit, dict) and isinstance(explicit.get("groups"), list):
        return _valid(explicit)
    grouping = default_grouping()
    old = _json(cfg.get("group_rules"), {})
    if not isinstance(old, dict):
        old = {}
    for name in old.get("groups") or []:
        if str(name).strip() and str(name).strip() != UNCLEAR:
            _ensure_group(grouping, str(name).strip())
    for r in old.get("rules") or []:
        if not isinstance(r, dict) or not r.get("group"):
            continue
        if r.get("kind") == "track":
            grouping["tracks"].append({"job": r.get("job"), "track": r.get("track"),
                                       "group": r["group"]})
        elif r.get("kind") in ("name", "genus", "family", "order") and r.get("value"):
            put_member(grouping, "species" if r["kind"] == "name" else r["kind"],
                       r["value"], r["group"])
    return grouping


def _json(value, default):
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def settings(config) -> dict:
    """The Identify Species node's group settings, parsed:
    ``{on, leave_out, share, grouping, groups}``. The editor stores the JSON
    fields as strings; a config saved before groups existed gets the defaults."""
    cfg = config or {}
    grouping = grouping_of(cfg)
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
        "grouping": grouping,
        "groups": [g["name"] for g in grouping["groups"]],
    }


def _index(grouping) -> dict:
    index = {}
    for g in grouping["groups"]:
        for m in g["members"]:
            index[(m["rank"], _key(m["name"]))] = g["name"]
    for t in grouping.get("tracks") or []:
        index[("track", str(t.get("job") or ""), str(t.get("track")))] = t["group"]
    # Where a bee genus nobody could place goes: the group holding Apidae.
    index["__bees__"] = next((g["name"] for g in grouping["groups"]
                              if any(m["rank"] == "family" and m["name"] == "Apidae"
                                     for m in g["members"])), None)
    return index


def name_group(name, lin, index) -> tuple[str, str]:
    """``(group, why)`` for one species name: the most specific taxon of it
    that a group holds (species, genus, family, order, class)."""
    lin = lin or {}
    for rank, value in (("species", name), ("genus", lin.get("genus")),
                        ("family", lin.get("family")), ("order", lin.get("order")),
                        ("class", lin.get("class"))):
        if rank == "order" and lin.get("bee") and index.get("__bees__"):
            # A bee genus GBIF couldn't place has no family: it's a bee, not
            # whatever holds the whole of Hymenoptera.
            return index["__bees__"], f"bee genus {lin.get('genus')}"
        if value and (rank, _key(value)) in index:
            return index[(rank, _key(value))], f"{rank} {_clean(value)}"
    return UNCLEAR, "not in a group"


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
    index = _index(cfg["grouping"])
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
        # The share is of the crops that were placed in a group: a crop named
        # something no group covers (a bat, a name GBIF doesn't know) says
        # nothing about bee vs ant, so it doesn't dilute the vote.
        placed = sum(named.values())
        share = round(votes / placed, 3) if placed else 0.0
        group = best if placed >= MIN_CROPS and share >= cfg["share"] else UNCLEAR
        top = max(species.items(), key=lambda kv: kv[1])[0] if species else ""
        why = "crops"
        if placed < MIN_CROPS:
            why = (f"only {placed} of {read} crop{'s' if read != 1 else ''} "
                   f"named something in a group")
        elif group == UNCLEAR:
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
            # What its crops were named, so a grouping can be tried on the
            # batch without reading the crop files again (memory/66).
            "names": species,
        }
    return out


def names_seen(readings: dict) -> list:
    return sorted({n for crops in readings.values() for n in crops if n})


# ── Changing a rule ─────────────────────────────────────────────────────────

def with_rule(config: dict, rule: dict) -> dict:
    """A copy of the node's config with ``rule`` written into its grouping:
    a track's group, or a species, genus or family put in a group (taken out
    of any other). An empty group takes it out. A new group name is added."""
    cfg = dict(config or {})
    grouping = grouping_of(cfg)
    if rule["kind"] == "track":
        grouping["tracks"] = [t for t in grouping["tracks"]
                              if (str(t.get("job")), str(t.get("track")))
                              != (str(rule.get("job")), str(rule.get("track")))]
        if rule.get("group"):
            if rule["group"] != UNCLEAR:
                _ensure_group(grouping, rule["group"])
            grouping["tracks"].append({"job": rule.get("job"), "track": rule.get("track"),
                                       "group": rule["group"]})
    else:
        rank = "species" if rule["kind"] == "name" else rule["kind"]
        group = rule.get("group")
        if not group:
            # Undo: back where the default grouping has it, if anywhere.
            group = next((g["name"] for g in default_grouping()["groups"]
                          for m in g["members"]
                          if m["rank"] == rank and _key(m["name"]) == _key(rule["value"])), "")
        put_member(grouping, rank, rule["value"], group)
    cfg["grouping"] = json.dumps(grouping)
    cfg.pop("group_rules", None)
    return cfg
