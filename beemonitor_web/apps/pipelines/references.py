"""Which layout each run of a batch was counted with (memory/81, 83): kept in
its listing, for the batch page's line on references a model found."""

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
    None when it has no reference step from a layout. With references a model
    found (memory/83) also ``tubes``: how they were numbered."""
    steps = run.steps or []
    # The reference an analyzer is wired to first: a run Update gave a new
    # Reference node keeps its old one too, earlier in its steps (memory/82).
    wired = {(s.get("inputs") or {}).get("rois") for s in steps}
    for step in sorted(steps, key=lambda s: s.get("id") not in wired):
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
                "state", "found", "filled", "dropped", "arrangement", "from_video",
                "reason", "detection", "error") if tubes.get(k) is not None}
        return entry
    return None
