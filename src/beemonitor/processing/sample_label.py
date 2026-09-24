"""Sample and pre-label a clip in one pass: motion proposes, a detector confirms.

Frame sampling used to keep every Nth frame, then the most-moving frames at
320 px; calibration on real clips (memory/38 §4.5) showed why neither works:

* a bee at 320 px is 3-10 px and a 3x3 open erased it;
* the burned-in clock changes every second and triggered on every clip;
* a hand on the lens, wind in grass and sun/shadow flicker out-score bees;
* a sitting insect barely moves, so motion alone never picks it.

So motion only *proposes* candidates (cheap, computed while decoding) and a
detector (SAM 3 on the GPU) *confirms* them. And because SAM 3 prompted "bee"
also boxes every mud-plugged nest hole on a bee hotel, a detection counts —
and is kept as a pre-label — only if it overlaps something that moved in that
frame. Plugs never move; a sitting insect shifts a little and stays.

Decoding is ``cv2.VideoCapture`` in sequential order, the same decoder and
order the web app's editor uses, so a frame number here is the same frame
there. Nothing in this module touches S3 or the GPU; callers pass a clip path
and a ``detect_fn`` (see ``sagemaker_backend/inference.py``).
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MotionConfig:
    """Motion-proposal settings, calibrated on 13 clips from 7 cameras."""
    width: int = 640               # ROI crop is scored at this width
    blob_min: int = 12             # bee-sized blob area, px at `width`
    blob_max: int = 2000
    handling: float = 0.20         # > this share of the ROI moving = hand/lighting
    clock_box: tuple = (0.0, 0.0, 0.35, 0.05)   # burned-in clock (x1, y1, x2, y2), full frame
    warmup_s: float = 1.0          # background model settles
    history: int = 300             # MOG2
    var_threshold: int = 16


@dataclass
class Candidate:
    n: int                          # frame number (sequential decode index)
    score: int                      # bee-sized moving blobs in the ROI
    frame: np.ndarray               # full-resolution BGR
    blobs: list = field(default_factory=list)   # [(x1, y1, x2, y2)] full-frame px


@dataclass
class ClipScan:
    fps: float
    width: int
    height: int
    frames: int
    scores: List[int]               # per frame
    candidates: List[Candidate]


def _roi_px(roi, w, h):
    x1, y1, x2, y2 = roi or (0.0, 0.0, 1.0, 1.0)
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))
    px1, py1 = int(x1 * w), int(y1 * h)
    return px1, py1, max(px1 + 2, int(x2 * w)), max(py1 + 2, int(y2 * h))


def _open(path):
    import cv2
    params = []
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        # One decode thread per clip: the caller runs several clips at once and
        # the box must keep a core free to answer health checks (memory/38 §3).
        params = [cv2.CAP_PROP_N_THREADS, 1]
    try:
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params) if params else cv2.VideoCapture(path)
    except Exception:
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("could not open video")
    return cap


def scan_clip(path: str, *, candidates: int = 15, spread: Optional[int] = None,
              min_gap_s: float = 0.5, roi=None, polygon=None,
              cfg: MotionConfig = MotionConfig()) -> ClipScan:
    """Decode once; score motion per frame; keep candidate frames in memory.

    Candidates are the ``candidates`` highest-scoring frames at least
    ``min_gap_s`` apart, plus ``spread`` evenly spaced frames (default
    ``ceil(candidates / 4)``) so a sitting insect is still checked.
    Memory is bounded: at most ``3 * candidates + spread`` frames are held.
    """
    import cv2

    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    warmup = int(round(fps * cfg.warmup_s))
    spread = (candidates + 3) // 4 if spread is None else spread
    spread_at = set()
    if spread and total_hint > warmup:
        usable = total_hint - warmup
        spread_at = {warmup + int((j + 0.5) * usable / spread) for j in range(spread)}

    mog = cv2.createBackgroundSubtractorMOG2(cfg.history, cfg.var_threshold, False)
    keep = 3 * candidates
    heap: list = []                 # min-heap of (score, -n, n)
    held: dict = {}                 # n -> Candidate (heap members + spread frames)
    scores: List[int] = []
    width = height = 0
    geom = None

    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if geom is None:
                height, width = frame.shape[:2]
                rx1, ry1, rx2, ry2 = _roi_px(roi, width, height)
                scale = cfg.width / float(rx2 - rx1)
                out_h = max(2, int(round((ry2 - ry1) * scale)))
                cx1, cy1, cx2, cy2 = _roi_px(cfg.clock_box, width, height)
                mask = None
                if polygon:
                    mask = np.zeros((out_h, cfg.width), np.uint8)
                    pts = np.array([[(px * width - rx1) * scale, (py * height - ry1) * scale]
                                    for px, py in polygon], np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                geom = (rx1, ry1, rx2, ry2, scale, out_h, cx1, cy1, cx2, cy2, mask)
            rx1, ry1, rx2, ry2, scale, out_h, cx1, cy1, cx2, cy2, mask = geom

            # Grey first, then shrink: one channel to resize instead of three
            # (~25% faster per frame). cvtColor returns a new image, so masking
            # the clock never touches the frame that may be saved.
            crop = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2GRAY)
            kx1, ky1 = max(cx1, rx1) - rx1, max(cy1, ry1) - ry1        # clock, in crop coords
            kx2, ky2 = min(cx2, rx2) - rx1, min(cy2, ry2) - ry1
            if kx2 > kx1 and ky2 > ky1:
                crop[ky1:ky2, kx1:kx2] = 0
            gray = cv2.resize(crop, (cfg.width, out_h), interpolation=cv2.INTER_AREA)
            fg = mog.apply(gray)

            score, blobs = 0, []
            if n >= warmup:
                if mask is not None:
                    fg = cv2.bitwise_and(fg, mask)
                if np.count_nonzero(fg) <= cfg.handling * fg.size:      # not a hand / light change
                    count, _lab, stats, _c = cv2.connectedComponentsWithStats(
                        (fg > 0).astype(np.uint8), 8)
                    for k in range(1, count):
                        x, y, w, h, area = stats[k]
                        if cfg.blob_min <= area <= cfg.blob_max:
                            blobs.append((rx1 + x / scale, ry1 + y / scale,
                                          rx1 + (x + w) / scale, ry1 + (y + h) / scale))
                    score = len(blobs)
            scores.append(score)

            if n in spread_at:
                held[n] = Candidate(n, score, frame, blobs)
            if score > 0:
                item = (score, -n, n)
                if len(heap) < keep:
                    heapq.heappush(heap, item)
                    held.setdefault(n, Candidate(n, score, frame, blobs))
                elif item > heap[0]:
                    _s, _neg, dropped = heapq.heapreplace(heap, item)
                    if dropped not in spread_at:
                        held.pop(dropped, None)
                    held.setdefault(n, Candidate(n, score, frame, blobs))
            n += 1
    finally:
        cap.release()

    gap = max(1, int(round(min_gap_s * fps)))
    chosen: list = []
    for _score, _neg, idx in sorted(heap, reverse=True):
        if len(chosen) >= candidates:
            break
        if all(abs(idx - c) >= gap for c in chosen):
            chosen.append(idx)
    for idx in sorted(spread_at):
        if idx in held and idx not in chosen:
            chosen.append(idx)
    return ClipScan(fps=fps, width=width, height=height, frames=n, scores=scores,
                    candidates=[held[i] for i in sorted(chosen) if i in held])


def _overlaps(box, blobs, pad: float) -> bool:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    for bx1, by1, bx2, by2 in blobs:
        p = pad * max(bx2 - bx1, by2 - by1)
        if x < bx2 + p and x + w > bx1 - p and y < by2 + p and y + h > by1 - p:
            return True
    return False


def moving_detections(boxes: Iterable[dict], blobs, pad: float = 1.0) -> list:
    """Detections that overlap something that moved in the same frame."""
    return [b for b in boxes if _overlaps(b, blobs, pad)]


def pick_frames(ranked: Sequence[tuple], count: int, min_gap_frames: int) -> list:
    """Top ``count`` of ``[(value, n), ...]`` (value > 0), ``min_gap`` apart."""
    out: list = []
    for value, n in sorted(ranked, key=lambda r: (-r[0], r[1])):
        if len(out) >= count or value <= 0:
            break
        if all(abs(n - m) >= min_gap_frames for m in out):
            out.append(n)
    return sorted(out)


def motion_profile(scores: Sequence[int], picks: Sequence[int], buckets: int = 60) -> dict:
    """The clip row's strip, same shape the web app stores: peak per bucket 0-100."""
    total = len(scores)
    if not total:
        return {"profile": [], "picked": [], "frames": 0}
    size = max(1, -(-total // buckets))
    peaks = [max(scores[i:i + size]) for i in range(0, total, size)]
    top = max(peaks) or 1
    return {"profile": [round(v / top * 100) for v in peaks],
            "picked": sorted({p // size for p in picks}), "frames": total}


def sample_label_clip(path: str, detect_fn: Callable[[List[np.ndarray]], List[List[dict]]], *,
                      max_frames: int = 20, candidates: int = 15, min_gap_s: float = 0.5,
                      roi=None, polygon=None, cfg: MotionConfig = MotionConfig()) -> dict:
    """Scan, detect on candidates, keep the top ``max_frames`` with moving detections.

    Returns ``{fps, width, height, frames, candidates, picks: [{n, frame, boxes}],
    motion}``. ``frame`` is the full-resolution image for the caller to encode.
    A clip where nothing detected moves returns no picks (an empty trigger).
    """
    scan = scan_clip(path, candidates=candidates, min_gap_s=min_gap_s, roi=roi,
                     polygon=polygon, cfg=cfg)
    frames = [c.frame for c in scan.candidates]
    detections = detect_fn(frames) if frames else []
    kept, ranked = {}, []
    for cand, boxes in zip(scan.candidates, detections):
        moving = moving_detections(boxes or [], cand.blobs)
        if moving:
            kept[cand.n] = (cand, moving)
            ranked.append((sum(float(b.get("confidence") or 0.5) for b in moving), cand.n))
    gap = max(1, int(round(min_gap_s * scan.fps)))
    picked = pick_frames(ranked, max_frames, gap)
    return {
        "fps": scan.fps, "width": scan.width, "height": scan.height,
        "frames": scan.frames, "candidates": len(scan.candidates),
        "picks": [{"n": n, "frame": kept[n][0].frame, "boxes": kept[n][1]} for n in picked],
        "motion": motion_profile(scan.scores, picked),
    }
