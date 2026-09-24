"""Which sampled frames a batch auto-label run spends GPU on.

"Auto-label 1,000 of 10,000 frames": the eligible frames are the sampled ones
nobody has labelled (no boxes, not reviewed), and the 1,000 are spread as
evenly as the footage allows — across hotel x hour cells first, then across the
clips in each cell, then across each clip's frames — so a run labels the whole
dataset thinly rather than a few busy clips densely.
"""

from __future__ import annotations

from .models import Annotation


def eligible_frames(project, videos):
    """The unlabelled sampled frames in ``videos``."""
    return Annotation.objects.filter(project=project, video__in=videos,
                                     boxes=[], reviewed=False)


def _water_fill(sizes: dict, target: int) -> dict:
    """Split ``target`` over keys as evenly as their ``sizes`` allow."""
    alloc = {k: 0 for k in sizes}
    left, open_keys = target, [k for k in sizes if sizes[k]]
    while left > 0 and open_keys:
        share = max(1, left // len(open_keys))
        still = []
        for k in open_keys:
            if left <= 0:
                break
            give = min(share, sizes[k] - alloc[k], left)
            alloc[k] += give
            left -= give
            if alloc[k] < sizes[k]:
                still.append(k)
        open_keys = still
    return alloc


def _spaced(items, k):
    n = len(items)
    if k >= n:
        return list(items)
    return [items[int((j + 0.5) * n / k)] for j in range(k)]


def pick(project, videos, target: int) -> dict:
    """``{video_id: [frame_number, ...]}`` — up to ``target`` frames, spread."""
    rows = (eligible_frames(project, videos)
            .order_by("video_id", "frame_number")
            .values_list("video_id", "frame_number", "video__device_id", "video__hour"))
    cells = {}   # (device, hour) -> {video_id: [frames]}
    for vid, frame, dev, hour in rows:
        cells.setdefault((dev, hour), {}).setdefault(vid, []).append(frame)

    per_cell = _water_fill({c: sum(len(f) for f in v.values()) for c, v in cells.items()},
                           max(0, target))
    out = {}
    for cell, clips in cells.items():
        per_clip = _water_fill({vid: len(f) for vid, f in clips.items()}, per_cell[cell])
        for vid, n in per_clip.items():
            if n:
                out[vid] = _spaced(clips[vid], n)
    return out
