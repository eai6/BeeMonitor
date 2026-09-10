"""Rejoin tracks that a chunk boundary cut in half.

A long clip is split into contiguous frame ranges so no single GPU invocation
exceeds the async platform's 1 h cap. Each chunk runs its own tracker, starting
its numbering from scratch, so a bee that is mid-flight when the boundary
falls comes back as two tracks. ``_remap_chunk_track_id`` then namespaces them
into two *distinct* ids — correct, in that ids no longer collide, but it also
means the same bee is counted twice in ``unique_tracks`` and appears as two
short visits rather than one long one.

The chunks are contiguous frame ranges of one video, so the join is decidable:
a track ending on the last frame before a boundary and a track starting on the
first frame after it are the same animal if their boxes overlap. That is a very
different problem from matching a bee across two *clips* recorded minutes
apart, which needs re-identification and is deliberately not attempted.
"""

import logging

logger = logging.getLogger(__name__)

# How far either side of a boundary a track may end or start and still be a
# candidate. Chunks are frame-contiguous, so a clean spanning track ends at
# B-1 and resumes at B; the slack covers a detector that dropped a frame or
# two right at the seam.
BOUNDARY_SLACK_FRAMES = 5

# Minimum box overlap to call two halves the same animal. A bee moves a little
# in the ~0.2 s the slack allows but not far, so this is deliberately generous
# on position and strict about "some other bee entirely".
MIN_IOU = 0.3

_ID_KEYS = ("track_id", "track", "tid")
_FRAME_KEYS = ("frame", "frame_num", "frame_number", "frame_idx")
_BOX_KEYS = (("x1", "xmin", "left"), ("y1", "ymin", "top"),
             ("x2", "xmax", "right"), ("y2", "ymax", "bottom"))


def _field(row, candidates):
    lower = {k.lower(): k for k in row}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _box(row, cols):
    try:
        return tuple(float(row[c]) for c in cols)
    except (TypeError, ValueError, KeyError):
        return None


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def stitch(rows, boundaries):
    """Merge boundary-split tracks in ``rows`` in place. Returns the join count.

    ``rows`` are the merged tracking CSV dicts — absolute frame numbers, track
    ids already namespaced per chunk. ``boundaries`` are the absolute start
    frames of chunks 1..N-1 (chunk 0's start is not a seam).
    """
    if not rows or not boundaries:
        return 0

    id_key = _field(rows[0], _ID_KEYS)
    frame_key = _field(rows[0], _FRAME_KEYS)
    box_cols = [_field(rows[0], names) for names in _BOX_KEYS]
    if not id_key or not frame_key or not all(box_cols):
        logger.info("chunk stitch: tracking CSV lacks id/frame/box columns — skipped")
        return 0

    # First and last observation of every track.
    ends, starts = {}, {}
    for row in rows:
        tid = row.get(id_key)
        try:
            frame = int(float(row.get(frame_key)))
        except (TypeError, ValueError):
            continue
        box = _box(row, box_cols)
        if box is None:
            continue
        if tid not in starts or frame < starts[tid][0]:
            starts[tid] = (frame, box)
        if tid not in ends or frame > ends[tid][0]:
            ends[tid] = (frame, box)

    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    joins = 0
    for boundary in sorted(boundaries):
        closing = [(tid, box) for tid, (frame, box) in ends.items()
                   if boundary - 1 - BOUNDARY_SLACK_FRAMES <= frame <= boundary - 1]
        opening = [(tid, box) for tid, (frame, box) in starts.items()
                   if boundary <= frame <= boundary + BOUNDARY_SLACK_FRAMES]
        if not closing or not opening:
            continue

        # Greedy best-first: the strongest overlap claims its partner, so one
        # bee near another can't steal a match that fits something better.
        pairs = sorted(
            ((_iou(cb, ob), ctid, otid)
             for ctid, cb in closing for otid, ob in opening),
            key=lambda p: -p[0])
        used_closing, used_opening = set(), set()
        for score, ctid, otid in pairs:
            if score < MIN_IOU:
                break
            if ctid in used_closing or otid in used_opening:
                continue
            if find(ctid) == find(otid):
                continue
            union(ctid, otid)
            used_closing.add(ctid)
            used_opening.add(otid)
            joins += 1

    if joins:
        for row in rows:
            tid = row.get(id_key)
            if tid in parent:
                row[id_key] = find(tid)
        logger.info("chunk stitch: rejoined %d boundary-split track(s)", joins)
    return joins


def distinct_track_count(rows):
    """How many tracks the rows actually contain, after any stitching.

    Summing each chunk's own ``unique_tracks`` double-counts every bee that
    crossed a seam; counting the merged rows cannot.
    """
    if not rows:
        return 0
    id_key = _field(rows[0], _ID_KEYS)
    if not id_key:
        return 0
    return len({row.get(id_key) for row in rows if row.get(id_key) not in (None, "")})
