"""Tubes found by a model in each clip, on CPU (memory/82), numbered as the
Reference node says (memory/83).

Fixing a 6,306-clip batch's tube boxes by hand went a day at a time. A
trained tube model finds them instead: a few frames of each clip (ffmpeg
seeks a presigned URL, no download), run through the model exported to ONNX
with OpenCV's DNN module, which the web image already has, so the recount
worker on Lambda can do it: no GPU, no new image. Checked on the built-in
nest model (YOLOv12m): the same 58-59 tubes as ultralytics on three clips,
every box within IoU 0.95, ~0.8 s a frame on a laptop CPU.

A detector's boxes have no identities, and tube numbers must mean the same
tube on every day: per-tube totals, trips (an exit paired with the next entry
at that tube), hops and review votes are all keyed on them. The Reference node
says how they are numbered (memory/83): as a grid of its rows and columns,
row by row from the top left (beemonitor.references.arrange.grid: extra boxes
dropped, missed ones placed from their row and column), or as found, in
reading order. A clip whose grid can't be told borrows the nearest clip's at
its site; else it is counted with no tubes, never wrong ones. No drawing is
compared with, and no person checks days.

The ONNX file sits beside the model's weights: ``x/best.pt`` ->
``x/best.onnx`` plus ``x/best.onnx.json`` (``{"names": [...], "imgsz": 640}``),
written by training (sagemaker_backend/train.py) or, for models trained
before, by ``scripts/export_onnx.py``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

DEFAULT_FRAMES = 8
DEFAULT_CONFIDENCE = 0.4
MAX_FRAMES = 30
# A tube is on at least this share of the frames read: a bee sitting on one
# shouldn't hide it, and a bee passing by isn't one.
MIN_SEEN = 0.5
# Frames are read between these fractions of the clip (its first and last
# moments are often a dark start or a moved camera).
FROM, TO = 0.05, 0.95
# Boxes of one tube on two frames overlap at least this much.
SAME_TUBE_IOU = 0.5
DEFAULT_ROWS, DEFAULT_COLUMNS = 6, 10
MAX_CELLS = 2000
ARRANGEMENTS = ("grid", "found")
# A clip whose grid fails borrows one from a clip this close at its site.
NEIGHBOUR_DAYS = 3
NEIGHBOURS_TRIED = 12

CACHE_DIR = os.environ.get("BEEMONITOR_TUBE_MODELS", "/tmp/beemonitor_tube_models")
_nets: dict = {}
_lock = threading.Lock()


class ModelNotReady(Exception):
    """The model has no ONNX copy (or no class names) to run on CPU."""


def onnx_key(model_key: str) -> str:
    base = model_key[:-3] if model_key.endswith(".pt") else model_key
    return base + ".onnx"


def meta_key(model_key: str) -> str:
    return onnx_key(model_key) + ".json"


def settings_key(frames, confidence) -> str:
    return f"f{int(frames)}-c{float(confidence):.2f}"


# ── The model on CPU ────────────────────────────────────────────────────────

def _load(model_key):
    """``(net, meta)``, downloaded once a process. The net isn't thread-safe:
    callers hold ``_lock`` while it runs."""
    if model_key in _nets:
        return _nets[model_key]
    import cv2

    from config.storage import get_s3_client

    folder = os.path.join(CACHE_DIR, hashlib.sha1(model_key.encode()).hexdigest()[:16])
    os.makedirs(folder, exist_ok=True)
    local, local_meta = os.path.join(folder, "model.onnx"), os.path.join(folder, "meta.json")
    client = get_s3_client()
    if not os.path.exists(local_meta):
        try:
            client.download_file("models", meta_key(model_key), local_meta)
        except Exception as exc:
            raise ModelNotReady(f"{model_key} has no CPU copy yet (scripts/export_onnx.py)") from exc
    if not os.path.exists(local):
        try:
            client.download_file("models", onnx_key(model_key), local + ".part")
            os.replace(local + ".part", local)
        except Exception as exc:
            raise ModelNotReady(f"{model_key} has no CPU copy yet (scripts/export_onnx.py)") from exc
    with open(local_meta) as fh:
        meta = json.load(fh)
    net = cv2.dnn.readNetFromONNX(local)
    _nets[model_key] = (net, meta)
    return _nets[model_key]


def model_names(model_key):
    """The class names the model's CPU copy carries, or None when it has none yet."""
    try:
        return list(_load(model_key)[1].get("names") or [])
    except ModelNotReady:
        return None


def boxes_in(net, meta, img, class_ids, confidence):
    """``[(x1, y1, x2, y2, score)]`` in the image's pixels, for these classes."""
    import cv2
    import numpy as np

    size = int(meta.get("imgsz") or 640)
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    pad = np.full((size, size, 3), 114, np.uint8)
    pad[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    net.setInput(cv2.dnn.blobFromImage(pad, 1 / 255.0, (size, size), swapRB=True))
    out = np.squeeze(net.forward())
    if out.ndim != 2:
        return []
    if out.shape[1] == 6 and out.shape[0] > out.shape[1]:
        # End-to-end head (YOLO26 and v10): [x1, y1, x2, y2, score, class], NMS done.
        rows = [(x1, y1, x2, y2, s) for x1, y1, x2, y2, s, c in out
                if int(c) in class_ids and s >= confidence]
    else:
        # v8 / v11 / v12 head: (4 + classes) x candidates, centre boxes, NMS here.
        preds = out.T if out.shape[0] < out.shape[1] else out
        scores = preds[:, 4:]
        cls, best = scores.argmax(1), scores.max(1)
        keep = np.isin(cls, list(class_ids)) & (best >= confidence)
        b, s = preds[keep, :4], best[keep]
        xywh = [[float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)] for cx, cy, bw, bh in b]
        idx = np.array(cv2.dnn.NMSBoxes(xywh, s.tolist(), confidence, 0.7)).flatten() if xywh else []
        rows = [(xywh[i][0], xywh[i][1], xywh[i][0] + xywh[i][2], xywh[i][1] + xywh[i][3], s[i])
                for i in idx]
    return [(float((x1 - left) / r), float((y1 - top) / r), float((x2 - left) / r),
             float((y2 - top) / r), float(sc)) for x1, y1, x2, y2, sc in rows]


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def objects(per_frame, frames_read):
    """The static objects across frames: boxes that overlap from frame to
    frame are one object, its box their median, kept when seen on at least
    MIN_SEEN of the frames read. ``[{box, confidence, seen}]``, 0..1 boxes."""
    import statistics

    clusters = []
    for boxes in per_frame:
        for box in sorted(boxes, key=lambda b: -b[4]):
            for c in clusters:
                if _iou(c["anchor"], box) >= SAME_TUBE_IOU:
                    c["boxes"].append(box)
                    break
            else:
                clusters.append({"anchor": box, "boxes": [box]})
    out = []
    for c in clusters:
        seen = len(c["boxes"])
        if not frames_read or seen / frames_read < MIN_SEEN:
            continue
        box = [round(float(statistics.median(b[k] for b in c["boxes"])), 5) for k in range(4)]
        out.append({"box": box, "confidence": round(float(sum(b[4] for b in c["boxes"])) / seen, 3),
                    "seen": seen})
    return out


def frame_times(duration, frames):
    """``frames`` times spread over the clip (FROM..TO of it)."""
    frames = max(1, min(MAX_FRAMES, int(frames)))
    if not duration:
        return [float(i) for i in range(frames)]
    if frames == 1:
        return [duration / 2]
    return [duration * (FROM + (TO - FROM) * i / (frames - 1)) for i in range(frames)]


def detect(video, model_key, label, frames=DEFAULT_FRAMES, confidence=DEFAULT_CONFIDENCE):
    """The clip's tubes by this model and class: the stored detection, else
    found now and stored. Returns the ``TubeDetection``; its ``error`` says
    why none were found (no frame read, a model with no CPU copy...)."""
    import cv2
    import numpy as np

    from apps.videos.models import TubeDetection
    from apps.videos.thumbnails import ensure_dimensions

    key = settings_key(frames, confidence)
    found = TubeDetection.objects.filter(video=video, model_key=model_key, label=label,
                                         settings_key=key).first()
    if found is not None and not found.retry():
        return found
    row = found or TubeDetection(video=video, model_key=model_key, label=label, settings_key=key)
    try:
        net, meta = _load(model_key)
        names = [str(n).lower() for n in meta.get("names") or []]
        class_ids = {i for i, n in enumerate(names) if n == str(label).strip().lower()}
        if not class_ids:
            raise ModelNotReady(f'"{label}" is not one of this model\'s classes ({", ".join(names)})')
        try:
            ensure_dimensions(video)
        except Exception:  # noqa: BLE001 - only the clip's length is wanted here
            pass
        from apps.videos.frames import frame_jpeg

        per_frame, read, size = [], 0, None
        for t in frame_times(float(video.duration_seconds or 0), frames):
            data = frame_jpeg(video, t)
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
            if img is None:
                continue
            read += 1
            size = img.shape[1], img.shape[0]
            with _lock:
                boxes = boxes_in(net, meta, img, class_ids, float(confidence))
            per_frame.append([(x1 / size[0], y1 / size[1], x2 / size[0], y2 / size[1], s)
                              for x1, y1, x2, y2, s in boxes])
        row.frames, row.boxes = read, objects(per_frame, read)
        row.frame_width, row.frame_height = size or (None, None)
        row.error = "" if read else "No frame of the clip could be read."
    except ModelNotReady as exc:
        row.frames, row.boxes, row.error = 0, [], str(exc)[:300]
    except Exception as exc:  # noqa: BLE001 - stored, so the day says why
        logger.exception("tube detection of video %s failed", video.pk)
        row.frames, row.boxes, row.error = 0, [], f"Detection failed: {exc}"[:300]
    row.save()
    return row


# ── Numbering: the Reference node's arrangement (memory/83) ─────────────────

def _centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def tube_settings(cfg) -> dict:
    """The model settings of a reference node's config."""
    def num(name, default, lo, hi, kind=float):
        try:
            v = kind(float(cfg.get(name)))
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    arrangement = str(cfg.get("arrangement") or "grid").strip().lower()
    return {"model": str(cfg.get("tube_model") or "").strip(),
           "label": str(cfg.get("tube_class") or "").strip(),
           "frames": num("tube_frames", DEFAULT_FRAMES, 1, MAX_FRAMES, int),
           "confidence": num("tube_confidence", DEFAULT_CONFIDENCE, 0.05, 0.95),
           "arrangement": arrangement if arrangement in ARRANGEMENTS else "grid",
           "rows": num("rows", DEFAULT_ROWS, 1, MAX_CELLS, int),
           "columns": num("columns", DEFAULT_COLUMNS, 1, MAX_CELLS, int)}
    return out


def arrangement_label(s) -> str:
    if s["arrangement"] == "found":
        return "as found"
    return f'{s["rows"]} × {s["columns"]} grid'


def arrange(boxes, s) -> dict:
    """``boxes`` (``[{box}]``) numbered as the node says: ``{ok, reason,
    nests, filled, dropped}``, ``nests`` a layout (``[{id, box}]``)."""
    from beemonitor.references.arrange import grid, reading_order

    raw = [list(map(float, b["box"])) for b in boxes or []]
    if s["arrangement"] == "found":
        ordered = reading_order(raw)
        return {"ok": bool(ordered), "reason": "" if ordered else "None found.",
                "nests": [{"id": i, "box": [round(v, 5) for v in b]} for i, b in enumerate(ordered, 1)],
                "filled": 0, "dropped": 0}
    g = grid(raw, s["rows"], s["columns"])
    return {"ok": g["ok"], "reason": g["reason"],
            "nests": [{"id": i, "box": [round(v, 5) for v in b]} for i, b in enumerate(g["boxes"], 1)],
            "filled": g["filled"], "dropped": g["dropped"]}


def _in_roi(boxes, roi):
    """Only boxes inside the ROI: what tracking runs over."""
    if not (roi and len(roi) == 4):
        return list(boxes or [])
    return [b for b in boxes or [] if roi[0] <= _centre(b["box"])[0] <= roi[2]
            and roi[1] <= _centre(b["box"])[1] <= roi[3]]


def _neighbour(video, found, s, roi):
    """``(video pk, arranged)`` of the nearest clip in time at the clip's site
    (else unit) whose same detection arranges, within NEIGHBOUR_DAYS; or None."""
    from datetime import timedelta

    from apps.videos.models import TubeDetection

    t = video.recorded_at
    place = ({"video__site_id": video.site_id} if video.site_id else
             {"video__device_id": video.device_id} if video.device_id else None)
    if t is None or place is None:
        return None
    qs = (TubeDetection.objects.filter(model_key=found.model_key, label=found.label,
                                       settings_key=found.settings_key, error="", **place)
          .exclude(video_id=video.pk).values("video_id", "video__recorded_at", "boxes"))
    span = timedelta(days=NEIGHBOUR_DAYS)
    near = [*qs.filter(video__recorded_at__lte=t, video__recorded_at__gte=t - span)
            .order_by("-video__recorded_at")[:NEIGHBOURS_TRIED],
            *qs.filter(video__recorded_at__gt=t, video__recorded_at__lte=t + span)
            .order_by("video__recorded_at")[:NEIGHBOURS_TRIED]]
    near.sort(key=lambda r: abs((r["video__recorded_at"] - t).total_seconds()))
    for row in near[:NEIGHBOURS_TRIED]:
        got = arrange(_in_roi(row["boxes"], roi), s)
        if got["ok"]:
            return row["video_id"], got
    return None


# ── For a clip: what a reference node with a model counts with ──────────────

def layout_for(video, user, cfg, detect_now=False) -> dict:
    """What a reference node set to a model gives this clip:
    ``{hotel_roi, hotel_polygon, nest_layout, source, source_label, tubes}``.

    The ROI is the saved one; the tubes are the model's, arranged as the node
    says, whatever was drawn. With no detection yet: found now when
    ``detect_now`` (the recount worker), else ``tubes.pending`` and no tubes,
    and the run is queued for the worker. ``tubes.state``: placed, borrowed
    (a nearby clip's grid), unplaced (none: counted without tubes), pending
    or error."""
    from apps.devices.layouts import layout_for_video
    from apps.training import model_classes

    s = tube_settings(cfg)
    saved = layout_for_video(video)
    base = {"artifact": "roi", "hotel_roi": saved["roi_override"],
            "hotel_polygon": saved["roi_polygon"], "label": s["label"], "source": "detected"}
    picked = model_classes.resolve(s["model"], user)
    if picked is None or not s["label"]:
        return {**base, "nest_layout": None, "source_label": "",
                "tubes": {"state": "error",
                          "error": "Pick the model and its class on the Reference node."}}
    from apps.videos.models import TubeDetection

    found = TubeDetection.objects.filter(video=video, model_key=picked["key"], label=s["label"],
                                         settings_key=settings_key(s["frames"], s["confidence"])).first()
    if (found is None or found.retry()) and detect_now:
        found = detect(video, picked["key"], s["label"], s["frames"], s["confidence"])
    label = f'{picked["name"]}: {s["label"]}, {arrangement_label(s)}'
    if found is None:
        return {**base, "nest_layout": None, "source_label": label, "tubes": {"state": "pending"}}
    if found.error:
        return {**base, "nest_layout": None, "source_label": label,
                "tubes": {"state": "error", "error": found.error, "detection": found.pk}}
    tubes = {"state": "placed", "detection": found.pk, "frames": found.frames,
             "found": len(found.boxes), "arrangement": arrangement_label(s)}
    roi = saved["roi_override"]
    got = arrange(_in_roi(found.boxes, roi), s)
    if not got["ok"]:
        near = _neighbour(video, found, s, roi)
        if near is None:
            return {**base, "nest_layout": None, "source_label": label,
                    "tubes": {**tubes, "state": "unplaced", "reason": got["reason"]}}
        tubes.update(state="borrowed", from_video=near[0], reason=got["reason"])
        got = near[1]
    tubes.update(filled=got["filled"], dropped=got["dropped"])
    return {**base, "nest_layout": got["nests"] or None, "source_label": label, "tubes": tubes}
