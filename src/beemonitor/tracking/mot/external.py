"""Standard multi-object trackers behind BeeTracker's surface (memory/43).

``BeeTracking`` and everything after it — crops, species / marker votes, the
tracking CSV, events — talk to the tracker through BeeTracker's interface:
``update(rows, frame_num) -> list``, ``get_active_tracks()``, and
``tracks[*]`` objects with ``id``, ``is_confirmed``, ``time_since_update``,
``last_bbox``, ``history``, ``taxon``… ``ExternalTracker`` gives the standard
trackers that interface, so choosing one changes association and nothing else.

Every tracker here is fed the pipeline's own detections — YOLO, SAM 3 or a
custom model — rather than running a detector of its own.

Trackers:
  * ``bytetrack`` / ``botsort`` — Ultralytics' BYTETracker / BOTSORT
    (already a dependency). BoT-SORT runs without ReID; its camera-motion
    compensation gets the frame.
  * ``ocsort`` — OC-SORT, reference implementation (vendor/ocsort, MIT).
  * ``sfsort`` — SFSORT, reference implementation (vendor/sfsort, MIT).

Rows in: ``[x1, y1, x2, y2, conf, source, taxon]`` (``to_tracker_rows``).
"""

from __future__ import annotations

import logging
from collections import deque
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

TRACKERS = ("bytetrack", "botsort", "ocsort", "sfsort")

# The defaults each tracker ships with (Ultralytics' YAMLs; the papers' code).
# Pipeline settings override these key by key.
DEFAULTS = {
    "bytetrack": {"track_high_thresh": 0.25, "track_low_thresh": 0.1,
                  "new_track_thresh": 0.25, "track_buffer": 30,
                  "match_thresh": 0.8, "fuse_score": True},
    "botsort": {"track_high_thresh": 0.25, "track_low_thresh": 0.1,
                "new_track_thresh": 0.25, "track_buffer": 30,
                "match_thresh": 0.8, "fuse_score": True,
                "gmc_method": "sparseOptFlow", "proximity_thresh": 0.5,
                "appearance_thresh": 0.8, "with_reid": False, "model": "auto"},
    "ocsort": {"det_thresh": 0.25, "max_age": 30, "min_hits": 3,
               "iou_threshold": 0.3, "delta_t": 3, "asso_func": "iou",
               "inertia": 0.2, "use_byte": False},
    # Timeouts / margins None = the middle of the authors' recommended range,
    # from the clip's fps and size (_build). The code's own 0 drops a track the
    # first frame it is missed.
    "sfsort": {"high_th": 0.6, "low_th": 0.1, "new_track_th": 0.7,
               "match_th_first": 0.67, "match_th_second": 0.3,
               "marginal_timeout": None, "central_timeout": None,
               "horizontal_margin": None, "vertical_margin": None},
}

# Frames a track is kept (for the CSV / crops surface) after it was last seen.
_VIEW_TTL = 1


class _Boxes:
    """The slice of Ultralytics' ``Boxes`` its trackers read: ``conf``,
    ``cls``, ``xyxy``, ``xywh``, boolean / index selection and ``len``."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    @property
    def xywh(self):
        xy = self.xyxy
        return np.stack([(xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2,
                         xy[:, 2] - xy[:, 0], xy[:, 3] - xy[:, 1]], axis=1) if len(xy) \
            else np.zeros((0, 4), np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _Boxes(self.xyxy[idx], self.conf[idx], self.cls[idx])


class TrackView:
    """One external track, shaped like ``mot.bee_tracker.Track``."""

    def __init__(self, track_id, bbox, confidence, source, taxon, frame_num):
        self.id = int(track_id)
        self.history = deque(maxlen=30)
        self.time_since_update = 0
        self.start_frame = frame_num
        self.bee_id = None
        self.bee_id_method = None
        self.bee_id_confidence = 0.0
        self.hits = 0
        self.observe(bbox, confidence, source, taxon, frame_num)

    def observe(self, bbox, confidence, source, taxon, frame_num):
        self.last_bbox = tuple(float(v) for v in bbox)
        self.last_confidence = float(confidence)
        self.last_source = source
        if taxon:
            self.taxon = taxon
        elif not hasattr(self, "taxon"):
            self.taxon = "bee"
        self.last_detection_frame = frame_num
        self.time_since_update = 0
        self.hits += 1
        self.history.append(self.centroid)

    # The external tracker only reports tracks it has confirmed.
    is_confirmed = True

    @property
    def centroid(self):
        x1, y1, x2, y2 = self.last_bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def set_bee_id(self, bee_id, method, confidence=1.0):
        if self.bee_id is None or confidence > self.bee_id_confidence:
            self.bee_id, self.bee_id_method, self.bee_id_confidence = bee_id, method, confidence

    def as_dict(self):
        x1, y1, x2, y2 = self.last_bbox
        cx, cy = self.centroid
        return {"track_id": self.id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "cx": float(cx), "cy": float(cy), "confidence": self.last_confidence,
                "source": self.last_source, "taxon": self.taxon, "bee_id": self.bee_id,
                "bee_id_method": self.bee_id_method,
                "bee_id_confidence": self.bee_id_confidence, "history": list(self.history)}


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ultralytics_tracker(kind, params, fps):
    """BYTETracker / BOTSORT across Ultralytics versions.

    The tracker reads whatever keys its version's YAML defines, and the set
    changes between releases — so start from the installed package's own
    ``cfg/trackers/<kind>.yaml`` and lay our settings over it. Before 8.4.1xx
    the constructor also took ``frame_rate`` (track_buffer was scaled by
    fps/30); later releases take ``args`` alone and count ``track_buffer`` in
    plain frames.
    """
    import os

    import yaml
    import ultralytics
    from ultralytics.trackers.basetrack import BaseTrack

    cfg_path = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", f"{kind}.yaml")
    try:
        with open(cfg_path) as fh:
            base = yaml.safe_load(fh) or {}
    except OSError:
        base = {}
    args = SimpleNamespace(**{**base, **params, "tracker_type": kind})
    if kind == "bytetrack":
        from ultralytics.trackers.byte_tracker import BYTETracker as cls
    else:
        from ultralytics.trackers.bot_sort import BOTSORT as cls
    getattr(BaseTrack, "reset_id", lambda: None)()   # ids from 1, per clip
    try:
        return cls(args, frame_rate=int(round(fps)))
    except TypeError:
        return cls(args)


class ExternalTracker:
    """A standard tracker that BeeTracking can use in place of BeeTracker."""

    def __init__(self, kind: str, params: dict | None = None, fps: float = 30.0,
                 frame_size: tuple | None = None):
        kind = str(kind).lower()
        if kind not in TRACKERS:
            raise ValueError(f"unknown tracker {kind!r}; choose one of {TRACKERS}")
        self.kind = kind
        self.params = {**DEFAULTS[kind], **{k: v for k, v in (params or {}).items()
                                            if k in DEFAULTS[kind] and v is not None}}
        self.fps = float(fps or 30.0)
        self.frame = None              # set per frame; BoT-SORT's GMC reads it
        self._views: dict[int, TrackView] = {}
        self._classes: dict[str, int] = {}
        w, h = frame_size or (0, 0)
        self._size = (int(h), int(w))
        self._impl = self._build(kind, self.params, self.fps, w, h)
        logger.info("Tracker: %s %s", kind, self.params)

    @staticmethod
    def _build(kind, p, fps, w, h):
        if kind in ("bytetrack", "botsort"):
            return _ultralytics_tracker(kind, p, fps)
        if kind == "ocsort":
            from beemonitor.tracking.mot.vendor.ocsort import OCSort
            return OCSort(**p)
        from beemonitor.tracking.mot.vendor.sfsort import SFSORT
        args = dict(p)
        if args.get("central_timeout") is None:
            args["central_timeout"] = int(round(1.0 * fps))      # README: 0.5-1.5 x fps
        if args.get("marginal_timeout") is None:
            args["marginal_timeout"] = int(round(0.5 * fps))     # README: 0.1-0.9 x fps
        if w and args.get("horizontal_margin") is None:
            args["horizontal_margin"] = int(0.075 * w)           # README: 0.05-0.1 x width
        if h and args.get("vertical_margin") is None:
            args["vertical_margin"] = int(0.075 * h)
        return SFSORT({**args, "frame_width": w or None, "frame_height": h or None})

    # --- BeeTracker surface ----------------------------------------------------
    @property
    def tracks(self):
        return list(self._views.values())

    def get_active_tracks(self):
        return [v.as_dict() for v in self._views.values() if v.time_since_update == 0]

    def update(self, detections, frame_num=None):
        rows = [r for r in (detections or []) if len(r) >= 5]
        boxes = np.array([r[:4] for r in rows], dtype=np.float32).reshape(-1, 4)
        conf = np.array([float(r[4]) for r in rows], dtype=np.float32)
        taxa = [r[6] if len(r) > 6 else "bee" for r in rows]
        sources = [r[5] if len(r) > 5 else "yolo" for r in rows]

        for view in self._views.values():
            view.time_since_update += 1

        for track_id, bbox, idx in self._associate(boxes, conf, taxa):
            if idx is None:   # tracker gave no index: the detection it overlaps most
                idx = max(range(len(rows)), key=lambda i: _iou(bbox, boxes[i]), default=None)
            src = sources[idx] if idx is not None else "yolo"
            taxon = taxa[idx] if idx is not None else None
            score = float(conf[idx]) if idx is not None else 0.0
            box = boxes[idx] if idx is not None else bbox
            view = self._views.get(track_id)
            if view is None:
                self._views[track_id] = TrackView(track_id, box, score, src, taxon, frame_num)
            else:
                view.observe(box, score, src, taxon, frame_num)

        for tid in [t for t, v in self._views.items() if v.time_since_update > _VIEW_TTL]:
            del self._views[tid]
        return self.get_active_tracks()

    # --- per tracker -----------------------------------------------------------
    def _associate(self, boxes, conf, taxa):
        """``[(track_id, bbox, detection_index | None), ...]`` for this frame."""
        if self.kind in ("bytetrack", "botsort"):
            cls = [self._classes.setdefault(t, len(self._classes)) for t in taxa]
            out = self._impl.update(_Boxes(boxes, conf, cls), self.frame)
            # [x1, y1, x2, y2, id, score, cls, idx]
            return [(int(r[4]), tuple(r[:4]), int(r[7])) for r in np.asarray(out).reshape(-1, 8)]
        if self.kind == "ocsort":
            dets = np.hstack([boxes, conf[:, None]]) if len(boxes) else np.empty((0, 5))
            out = self._impl.update(dets, self._size, self._size)
            return [(int(r[4]), tuple(r[:4]), None) for r in np.asarray(out).reshape(-1, 5)]
        out = self._impl.update(boxes, conf) if len(boxes) else self._impl.update(
            np.empty((0, 4)), np.empty((0,)))
        return [(int(tid) + 1, tuple(np.asarray(box, dtype=float)), None) for box, tid in out]
