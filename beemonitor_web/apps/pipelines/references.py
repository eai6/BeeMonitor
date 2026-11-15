"""Which layout each clip of a batch was counted with, and which it has now
(memory/81).

Fixing a batch's tube references went clip by clip: filter to a day, open a
clip, save for its day, recount, and nothing said which days were done or
which saved days were never recounted. A run's reference step keeps the
geometry it counted with; comparing that with the clip's layout now says
which clips are out of date, with no history to keep: a layout saved again
unchanged, or the same drawing saved on every clip of a day, reads the same.
"""

from __future__ import annotations

import hashlib
import json

from .executors import _REFERENCE_BLOCKS


def geometry_key(roi, polygon, nests) -> str:
    """A short fingerprint of a layout's shapes: equal shapes, equal key."""
    text = json.dumps([roi or None, polygon or None, nests or []], sort_keys=True, default=str)
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def counted(run):
    """``{key, label}`` of the layout the run's analyzers counted against, or
    None when it has no reference step from a layout. With detected tubes
    (memory/82) also ``tubes``: how the detection was numbered, and the
    template's key it was numbered from."""
    for step in run.steps or []:
        if step.get("block_type") not in _REFERENCE_BLOCKS:
            continue
        out = (run.context or {}).get(step.get("id")) or {}
        if out.get("artifact") != "roi":
            continue
        tubes = out.get("tubes")
        if out.get("source") == "detected" and not tubes:
            continue                      # a Detect node's boxes, not a layout
        entry = {"key": geometry_key(out.get("hotel_roi"), out.get("hotel_polygon"),
                                     out.get("nest_layout")),
                 "label": out.get("source_label") or ""}
        if tubes:
            entry["tubes"] = {k: tubes.get(k) for k in (
                "state", "flag", "matched", "kept", "new", "found", "shift", "scale",
                "template_key", "detection", "error", "marks") if tubes.get(k) is not None}
        return entry
    return None


def current_for(videos) -> dict:
    """``{video pk: {key, source, label, number, layout}}``: the layout each
    clip is analysed with now (devices.layouts.layout_for_video), in a few
    queries for clips with their own or their site's, one lookup each for
    the rest."""
    from apps.devices.layouts import layout_for_video
    from apps.videos.models import Layout

    videos = list(videos)
    own, site = {}, {}
    for lay in Layout.objects.filter(video__in=[v.pk for v in videos]).order_by("video_id", "-number"):
        own.setdefault(lay.video_id, lay)
    for lay in (Layout.objects.filter(site__in={v.site_id for v in videos if v.site_id})
                .order_by("site_id", "-number")):
        site.setdefault(lay.site_id, lay)
    out = {}
    for v in videos:
        # The layout it shares with its site's other clips: what detected
        # tubes are numbered from (memory/82).
        if v.site_id in site:
            shared = site[v.site_id].nest_layout
        elif v.pk in own:
            from apps.devices.layouts import shared_layout_for_video
            shared = shared_layout_for_video(v)["nest_layout"]
        else:
            shared = None
        if v.pk in own:
            lay = own[v.pk]
            out[v.pk] = _entry(lay.roi_override, lay.roi_polygon, lay.nest_layout, "video",
                               f"this clip (v{lay.number})", lay)
        elif v.site_id in site:
            lay = site[v.site_id]
            out[v.pk] = _entry(lay.roi_override, lay.roi_polygon, lay.nest_layout, "site",
                               f"site {v.site.name} (v{lay.number})", lay)
        else:
            d = layout_for_video(v)
            out[v.pk] = _entry(d["roi_override"], d["roi_polygon"], d["nest_layout"],
                               d["source"], d["source_label"], None)
            shared = d["nest_layout"]
        out[v.pk]["shared_key"] = geometry_key(None, None, shared if shared is not None
                                               else out[v.pk]["nests"])
    return out


def _entry(roi, polygon, nests, source, label, layout):
    return {"key": geometry_key(roi, polygon, nests), "source": source, "label": label,
            "number": getattr(layout, "number", None), "roi": roi, "polygon": polygon,
            "nests": nests or []}


def days(rows) -> list[dict]:
    """The batch's clips by place and local day (the layout editor's "this
    day"), oldest first, each with what layout its clips have and how many
    finished runs were counted with it.

    ``rows``: listing.rows. A row's ``layout`` (from its listing) is what it
    was counted with."""
    from .period import local_start

    videos = {r["video"].pk: r["video"] for r in rows if r["video"]}
    now = current_for(videos.values())
    out = {}
    for r in rows:
        video = r["video"]
        if video is None:
            continue
        start = local_start(video)
        place = r["site"] or getattr(getattr(video, "device", None), "name", "") or ""
        key = (place, start.date() if start else None)
        day = out.setdefault(key, {
            "place": place, "date": key[1], "clips": 0, "videos": [], "layouts": {},
            "analysed": 0, "up_to_date": 0, "unknown": 0, "stale_runs": [], "own": 0,
            "detected": 0, "waiting": 0, "errors": 0, "flagged": [], "worst": None})
        day["clips"] += 1
        day["videos"].append(video)
        cur = now[video.pk]
        day["own"] += cur["source"] == "video"
        lay = (r.get("layout") or {})
        tubes = lay.get("tubes")
        if not tubes:
            day["layouts"].setdefault(cur["key"], cur)
        if r["status"] != "completed" or not lay:
            continue
        day["analysed"] += 1
        if tubes:
            _detected_row(day, r, lay, tubes, cur)
        elif not lay.get("key"):
            day["unknown"] += 1
        elif lay["key"] == cur["key"]:
            day["up_to_date"] += 1
        else:
            day["stale_runs"].append(r["run"].pk)
    for day in out.values():
        day["videos"].sort(key=lambda v: (v.recorded_at is None, v.recorded_at, v.pk))
        layouts = list(day["layouts"].values())
        day["mixed"] = len(layouts) > 1
        day["layout"] = layouts[0] if len(layouts) == 1 else None
        day["stale"] = len(day["stale_runs"])
        day["kind"] = _kind(day)
        parts = sorted(day.pop("flag_parts", []) + [f"e{day['errors']}"])
        day["flag_key"] = hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]
    return sorted(out.values(), key=lambda d: (d["place"], d["date"] is None, d["date"]))


# How bad a detected clip's numbering is, worst last (memory/82).
_FLAGS = ("", "moved", "kept", "new", "no_template", "none_found", "error")


def _detected_row(day, row, lay, tubes, cur):
    """One run counted with detected tubes, into its day."""
    state = tubes.get("state")
    if state == "pending":
        day["waiting"] += 1
        return
    if state == "error":
        day["errors"] += 1
    if state == "drawn":
        # Counted with the clip's own drawing: current while it is the one.
        if cur["source"] == "video" and lay["key"] == cur["key"]:
            day["up_to_date"] += 1
        else:
            day["stale_runs"].append(row["run"].pk)
        return
    if cur["source"] == "video":
        day["stale_runs"].append(row["run"].pk)    # drawn since: the drawing wins
        return
    if tubes.get("template_key") and tubes["template_key"] != cur.get("shared_key"):
        day["stale_runs"].append(row["run"].pk)    # the site's numbering changed
        return
    day["detected"] += 1
    day["up_to_date"] += state != "error"
    flag = tubes.get("flag") or ""
    if flag:
        day["flagged"].append(row["run"].pk)
        day.setdefault("flag_parts", []).append(
            f'{row["run"].pk}:{tubes.get("detection")}:{tubes.get("template_key")}:{flag}')
    worst = day["worst"]
    if worst is None or _FLAGS.index(flag if flag in _FLAGS else "error") > \
            _FLAGS.index(worst["flag"] if worst["flag"] in _FLAGS else "error"):
        day["worst"] = {"flag": flag, "run": row["run"].pk, "video": row["video"], **{
            k: tubes.get(k) for k in ("matched", "kept", "new", "found", "shift", "error")}}


def _kind(day) -> str:
    """The day's line on the References page."""
    if day["flagged"] or day["errors"]:
        return "look"
    if day["waiting"]:
        return "waiting"
    if day["detected"]:
        return "hand" if day["own"] else "detected"
    return "hand" if day["own"] else "saved"
