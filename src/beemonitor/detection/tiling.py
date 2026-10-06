"""Detect small objects in large photos by tiling (memory/45).

A detector resizes its input (YOLO to 640 px, SAM 3 to ~1K), so a 60 px bee in
a 64 MP photo (9152 x 6944) would shrink to a few pixels. The photo is cut
into overlapping tiles near the detector's own scale, each tile is detected,
boxes are shifted back to full-resolution coordinates and merged.

An object cut by a tile edge is whole in the neighbouring tile (the overlap is
larger than the objects of interest), so merging keeps the larger box when one
box of the same class lies mostly inside another, as well as ordinary IoU NMS.
"""

from __future__ import annotations

TILE = 1280
OVERLAP = 0.2


def tile_grid(width, height, tile=TILE, overlap=OVERLAP):
    """``[(x, y, w, h), ...]`` covering the image. One tile when it fits."""
    if width <= tile * 1.25 and height <= tile * 1.25:
        return [(0, 0, int(width), int(height))]
    step = max(1, int(tile * (1 - overlap)))

    def starts(size):
        if size <= tile:
            return [0]
        out = list(range(0, size - tile, step))
        out.append(size - tile)                      # last tile flush with the edge
        return sorted(set(out))

    return [(x, y, min(tile, width - x), min(tile, height - y))
            for y in starts(int(height)) for x in starts(int(width))]


def _area(b):
    return max(0.0, b["w"]) * max(0.0, b["h"])


def _inter(a, b):
    x1, y1 = max(a["x"], b["x"]), max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def merge_boxes(boxes, iou=0.5, contain=0.6):
    """Class-aware merge of tile detections: biggest/most confident first; a box
    is dropped when it overlaps a kept one by ``iou`` or lies ``contain`` of its
    own area inside it."""
    kept = []
    for b in sorted(boxes, key=lambda b: (_area(b), b.get("confidence") or 0), reverse=True):
        drop = False
        for k in kept:
            if k.get("class") != b.get("class"):
                continue
            inter = _inter(b, k)
            if not inter:
                continue
            union = _area(b) + _area(k) - inter
            if (union and inter / union >= iou) or (_area(b) and inter / _area(b) >= contain):
                drop = True
                break
        if not drop:
            kept.append(b)
    return kept


def detect_tiled(image, detect, tile=TILE, overlap=OVERLAP):
    """Run ``detect(frame) -> [{x, y, w, h, class, confidence, ...}]`` over
    tiles of ``image`` (an HxWx3 array); boxes come back in full-image pixels.
    Returns ``(boxes, tile_count)``."""
    h, w = image.shape[:2]
    grid = tile_grid(w, h, tile, overlap)
    found = []
    for x, y, tw, th in grid:
        for box in detect(image[y:y + th, x:x + tw]):
            found.append({**box, "x": box["x"] + x, "y": box["y"] + y})
    return (merge_boxes(found) if len(grid) > 1 else found), len(grid)
