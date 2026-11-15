"""Tubes found by a model in each clip, on CPU, numbered from the site's
layout (memory/82).

Fixing a 6,306-clip batch's tube boxes by hand went a day at a time. A
trained tube model finds them instead: a few frames of each clip (ffmpeg
seeks a presigned URL, no download), run through the model exported to ONNX
with OpenCV's DNN module, which the web image already has, so the recount
worker on Lambda can do it: no GPU, no new image. Checked on the built-in
nest model (YOLOv12m): the same 58-59 tubes as ultralytics on three clips,
every box within IoU 0.95, ~0.8 s a frame on a laptop CPU.

A detector's boxes have no identities, and tube numbers must mean the same
tube on every day: per-tube totals, trips (an exit paired with the next entry
at that tube), hops and review votes are all keyed on them. So the boxes are
matched to the site's saved layout: the camera's shift (and zoom) worked out
first, then each detected tube paired with the nearest template tube. A tube
the model missed keeps the template's box, moved by the shift; one it found
that the template lacks gets the next free number. Either, or a camera that
moved more than a tube's width, flags the day to look at.

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
# Matching: a detected tube pairs with a template tube whose centre is within
# this many tube widths once the camera's shift is taken out.
PAIR_WITHIN = 0.5
# A day to look at: any tube new, more than this share kept from the
# template, or the camera moved more than a tube's width (or zoomed > 10 %).
KEPT_SHARE = 0.10
MOVED_TUBES = 1.0
ZOOMED = 0.10

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


# ── Numbering from the site's layout ────────────────────────────────────────

def _centre(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _by_position(boxes):
    """Boxes in reading order: rows top to bottom (a row is boxes whose
    centres are within half a tube's height), then left to right."""
    import statistics

    if not boxes:
        return []
    h = statistics.median(b[3] - b[1] for b in boxes) or 0.01
    rows = []
    for b in sorted(boxes, key=lambda b: _centre(b)[1]):
        if rows and abs(_centre(b)[1] - _centre(rows[-1][-1])[1]) <= h / 2:
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: _centre(b)[0])]


def _fit(pairs):
    """Least-squares ``(scale, dx, dy)`` taking template centres onto the
    detected ones: x' = s·x + dx, y' = s·y + dy."""
    import numpy as np

    t = np.array([p[0] for p in pairs], float)
    f = np.array([p[1] for p in pairs], float)
    if len(pairs) < 3:
        d = (f - t).mean(0)
        return 1.0, float(d[0]), float(d[1])
    tc, fc = t.mean(0), f.mean(0)
    var = ((t - tc) ** 2).sum()
    s = float(((t - tc) * (f - fc)).sum() / var) if var > 0 else 1.0
    if not 0.5 < s < 2.0:
        s = 1.0
    d = fc - s * tc
    return s, float(d[0]), float(d[1])


def match(template, found):
    """The detected tubes numbered from the template layout.

    ``template``: ``[{id, box, name?, points?}]`` (0..1), ``found``: ``[{box,
    ...}]``. Returns ``{nests, matched, kept, new, shift, scale, flag}``:
    ``nests`` is a layout (``[{id, box, name?, kept?, new?}]``), ``shift`` the
    camera's move in tube widths/heights, ``flag`` why the day is worth a look
    ("" when it isn't)."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    tmpl = [t for t in template or [] if isinstance(t, dict) and len(t.get("box") or []) == 4]
    boxes = [list(map(float, f["box"])) for f in found or []]
    if not tmpl:
        nests = [{"id": i, "box": b} for i, b in enumerate(_by_position(boxes), 1)]
        return {"nests": nests, "matched": 0, "kept": 0, "new": len(nests),
                "shift": [0.0, 0.0], "scale": 1.0, "flag": "no_template" if nests else "none_found"}
    tb = [list(map(float, t["box"])) for t in tmpl]
    tw = float(np.median([b[2] - b[0] for b in tb])) or 0.01
    th = float(np.median([b[3] - b[1] for b in tb])) or 0.01
    if not boxes:
        return {"nests": [{**_ref(t), "kept": True} for t in tmpl], "matched": 0,
                "kept": len(tmpl), "new": 0, "shift": [0.0, 0.0], "scale": 1.0,
                "flag": "none_found"}
    T = np.array([_centre(b) for b in tb])
    F = np.array([_centre(b) for b in boxes])
    unit = np.array([tw, th])

    def paired_at(s, dx, dy, within):
        moved = T * s + np.array([dx, dy])
        cost = np.linalg.norm((moved[:, None, :] - F[None, :, :]) / unit, axis=2)
        rows, cols = linear_sum_assignment(cost)
        keep = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= within]
        return keep, sum(cost[r, c] for r, c in keep)

    def refined(s, dx, dy):
        for _ in range(4):
            keep, _cost = paired_at(s, dx, dy, PAIR_WITHIN * 2)
            if not keep:
                break
            s, dx, dy = _fit([(T[r], F[c]) for r, c in keep])
        keep, cost = paired_at(s, dx, dy, PAIR_WITHIN)
        return len(keep), -cost, (s, dx, dy)

    # The camera's shift: the commonest template -> detected offsets, on a
    # grid of half a tube (robust to a few missed or extra tubes), each
    # refined (with the zoom) and the one that pairs the most tubes kept. A
    # regular block of tubes offers a lattice of near-ties, so not just one.
    diffs = (F[None, :, :] - T[:, None, :]).reshape(-1, 2) / unit
    cells = np.round(diffs * 2).astype(int)
    keys, counts = np.unique(cells, axis=0, return_counts=True)
    tries = [keys[i] / 2.0 for i in np.argsort(-counts)[:12]] + [np.zeros(2)]
    best = max((refined(1.0, float(k[0] * tw), float(k[1] * th)) for k in tries),
               key=lambda r: (r[0], r[1]))
    s, dx, dy = best[2]
    moved = T * s + np.array([dx, dy])
    cost = np.linalg.norm((moved[:, None, :] - F[None, :, :]) / unit, axis=2)
    rows, cols = linear_sum_assignment(cost)
    paired = {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] <= PAIR_WITHIN}

    nests, used = [], set(paired.values())
    for i, t in enumerate(tmpl):
        if i in paired:
            nests.append({**_ref(t), "box": [round(v, 5) for v in boxes[paired[i]]]})
        else:
            b = tb[i]
            cx, cy = _centre(b)
            hw, hh = (b[2] - b[0]) * s / 2, (b[3] - b[1]) * s / 2
            ncx, ncy = cx * s + dx, cy * s + dy
            nests.append({**_ref(t), "box": [round(v, 5) for v in (ncx - hw, ncy - hh, ncx + hw, ncy + hh)],
                          "kept": True})
    next_id = max([_int(t.get("id")) for t in tmpl] + [0]) + 1
    extra = _by_position([boxes[j] for j in range(len(boxes)) if j not in used])
    for k, b in enumerate(extra):
        nests.append({"id": next_id + k, "box": [round(v, 5) for v in b], "new": True})
    kept = len(tmpl) - len(paired)
    # The shift in tube sizes: where the template's middle went.
    mid = T.mean(0)
    shift = ((mid * s + np.array([dx, dy])) - mid) / unit
    flag = ""
    if extra:
        flag = "new"
    elif kept > KEPT_SHARE * len(tmpl):
        flag = "kept"
    elif float(np.abs(shift).max()) > MOVED_TUBES or abs(s - 1) > ZOOMED:
        flag = "moved"
    return {"nests": nests, "matched": len(paired), "kept": kept, "new": len(extra),
            "shift": [round(float(shift[0]), 3), round(float(shift[1]), 3)],
            "scale": round(s, 4), "flag": flag}


def _ref(t):
    out = {"id": t.get("id"), "box": list(t["box"])}
    if t.get("name"):
        out["name"] = t["name"]
    return out


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ── For a clip: what a reference node with detected tubes counts with ───────

def tube_settings(cfg) -> dict:
    """The detected-tubes settings of a reference node's config."""
    def num(name, default, lo, hi, kind=float):
        try:
            v = kind(float(cfg.get(name)))
        except (TypeError, ValueError):
            return default
        return max(lo, min(hi, v))

    return {"model": str(cfg.get("tube_model") or "").strip(),
            "label": str(cfg.get("tube_class") or "").strip(),
            "frames": num("tube_frames", DEFAULT_FRAMES, 1, MAX_FRAMES, int),
            "confidence": num("tube_confidence", DEFAULT_CONFIDENCE, 0.05, 0.95),
            "drawn_wins": str(cfg.get("drawn_wins", "on")).lower() not in ("", "0", "false", "off", "no")}


def layout_for(video, user, cfg, detect_now=False) -> dict:
    """What a reference node set to detected tubes gives this clip:
    ``{hotel_roi, hotel_polygon, nest_layout, source, source_label, tubes}``.

    The ROI is always the saved one. A clip with its own drawn layout keeps it
    when the node says a drawing wins. Otherwise the clip's detection, matched
    to its site's (or unit's) layout. With no detection yet: found now when
    ``detect_now`` (the recount worker), else ``tubes.pending`` and no tubes,
    and the run is queued for the worker."""
    from apps.devices.layouts import layout_for_video, shared_layout_for_video
    from apps.training import model_classes

    from .references import geometry_key

    s = tube_settings(cfg)
    saved = layout_for_video(video)
    base = {"artifact": "roi", "hotel_roi": saved["roi_override"],
            "hotel_polygon": saved["roi_polygon"], "label": s["label"]}
    if saved["source"] == "video" and s["drawn_wins"]:
        return {**base, "nest_layout": saved["nest_layout"] or None, "source": "video",
                "source_label": saved["source_label"],
                "tubes": {"state": "drawn", "flag": ""}}
    picked = model_classes.resolve(s["model"], user)
    if picked is None or not s["label"]:
        return {**base, "nest_layout": None, "source": "detected", "source_label": "",
                "tubes": {"state": "error", "flag": "error",
                          "error": "Pick the tube model and its class on the Reference node."}}
    from apps.videos.models import TubeDetection

    found = TubeDetection.objects.filter(video=video, model_key=picked["key"], label=s["label"],
                                         settings_key=settings_key(s["frames"], s["confidence"])).first()
    if (found is None or found.retry()) and detect_now:
        found = detect(video, picked["key"], s["label"], s["frames"], s["confidence"])
    label = f'{picked["name"]}: {s["label"]}'
    if found is None:
        return {**base, "nest_layout": None, "source": "detected", "source_label": label,
                "tubes": {"state": "pending", "flag": ""}}
    if found.error:
        return {**base, "nest_layout": None, "source": "detected", "source_label": label,
                "tubes": {"state": "error", "flag": "error", "error": found.error,
                          "detection": found.pk}}
    template = shared_layout_for_video(video)
    boxes = found.boxes
    roi = saved["roi_override"]
    if roi and len(roi) == 4:
        # Only tubes inside the ROI: what tracking runs over.
        boxes = [b for b in boxes if roi[0] <= _centre(b["box"])[0] <= roi[2]
                 and roi[1] <= _centre(b["box"])[1] <= roi[3]]
    m = match(template["nest_layout"] or [], boxes)
    return {**base, "nest_layout": [{k: v for k, v in n.items() if k not in ("kept", "new")}
                                    for n in m["nests"]] or None,
            "source": "detected", "source_label": label,
            "tubes": {"state": "detected", "detection": found.pk, "frames": found.frames,
                      "found": len(found.boxes), "matched": m["matched"], "kept": m["kept"],
                      "new": m["new"], "shift": m["shift"], "scale": m["scale"],
                      "flag": m["flag"], "template": template["source_label"],
                      "template_key": geometry_key(None, None, template["nest_layout"]),
                      "marks": {str(n["id"]): ("kept" if n.get("kept") else "new")
                                for n in m["nests"] if n.get("kept") or n.get("new")}}}
