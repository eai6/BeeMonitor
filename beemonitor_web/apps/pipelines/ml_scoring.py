"""Optional ML scoring of events (memory/60).

The Events node's "Score events with the ML classifier" switch. Each exit and
enter gets the trained event classifier's probability that it is a real event,
in an ``ml_confidence`` column. Nothing is dropped: the reader picks their own
cut-off. Stays get no score — the classifier was only ever trained on exits
and entries.

The classifier (``models/event_classifier_model.pkl``) is a scikit-learn
random forest over 20 trajectory features, in pixels and frames. The feature
code is the GPU worker's ``EventProcessor._extract_trajectory_features``,
copied rather than imported because importing the worker's package pulls in
torch, which the web image does not have. ``test_ml_scoring`` holds the two to
the same numbers.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)

MODEL_PATHS = (
    os.environ.get("EVENT_CLASSIFIER_PATH", ""),
    "/app/models/event_classifier_model.pkl",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "models",
                 "event_classifier_model.pkl"),
)


@lru_cache(maxsize=1)
def _model():
    """``(model, feature_names)``, or None when the classifier can't be loaded."""
    try:
        import joblib
    except ImportError:
        logger.error("scikit-learn/joblib are not installed — ML scoring unavailable")
        return None
    for path in MODEL_PATHS:
        if path and os.path.exists(path):
            try:
                data = joblib.load(path)
                return data["model"], list(data["feature_names"])
            except Exception:  # noqa: BLE001 - reported as "unavailable"
                logger.exception("could not load the event classifier at %s", path)
                return None
    logger.error("event classifier not found (looked in %s)", [p for p in MODEL_PATHS if p])
    return None


def trajectory_features(centroids, frames, nest_box, is_entry):
    """The worker's 20 features for one trajectory and one reference.

    ``centroids`` are box centres in pixels, one per detected frame, in frame
    order; ``nest_box`` is the reference's box in pixels. Same formulas, same
    order, as ``EventProcessor._extract_trajectory_features``.
    """
    import numpy as np

    nest_x = (nest_box[0] + nest_box[2]) / 2
    nest_y = (nest_box[1] + nest_box[3]) / 2

    trajectory_length = len(centroids)
    steps = [float(np.sqrt((centroids[i + 1][0] - centroids[i][0]) ** 2
                           + (centroids[i + 1][1] - centroids[i][1]) ** 2))
             for i in range(len(centroids) - 1)]
    path_length = sum(steps)
    displacement = np.sqrt((centroids[-1][0] - centroids[0][0]) ** 2
                           + (centroids[-1][1] - centroids[0][1]) ** 2)
    tortuosity = path_length / displacement if displacement > 0 else 0

    speeds = steps
    avg_speed = np.mean(speeds) if speeds else 0
    max_speed = np.max(speeds) if speeds else 0
    speed_std = np.std(speeds) if speeds else 0
    speed_cv = speed_std / avg_speed if avg_speed > 0 else 0
    third = len(speeds) // 3 if len(speeds) >= 3 else 1
    start_speed = np.mean(speeds[:third]) if speeds else 0
    middle_speed = np.mean(speeds[third:2 * third]) if len(speeds) >= 3 else avg_speed
    end_speed = np.mean(speeds[-third:]) if speeds else 0
    decel_ratio = end_speed / start_speed if start_speed > 0 else 1.0

    start_to_nest = np.sqrt((centroids[0][0] - nest_x) ** 2 + (centroids[0][1] - nest_y) ** 2)
    end_to_nest = np.sqrt((centroids[-1][0] - nest_x) ** 2 + (centroids[-1][1] - nest_y) ** 2)
    approach_ratio = end_to_nest / start_to_nest if start_to_nest > 0 else 1.0

    xs = [c[0] for c in centroids]
    ys = [c[1] for c in centroids]
    return {
        "trajectory_length": trajectory_length,
        "path_length": path_length,
        "displacement": displacement,
        "tortuosity": tortuosity,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "speed_std": speed_std,
        "speed_cv": speed_cv,
        "start_speed": start_speed,
        "middle_speed": middle_speed,
        "end_speed": end_speed,
        "decel_ratio": decel_ratio,
        "start_to_nest": start_to_nest,
        "end_to_nest": end_to_nest,
        "approach_ratio": approach_ratio,
        "x_var": np.var(xs),
        "y_var": np.var(ys),
        "vertical_movement": centroids[-1][1] - centroids[0][1],
        "horizontal_movement": abs(centroids[-1][0] - centroids[0][0]),
        "is_entry": 1 if is_entry else 0,
    }


# A stay's own features (memory/70): the exit/enter features describe coming
# from or going to a reference; a stay is an insect that stayed in one.
STAY_FEATURES = [
    "trajectory_length", "duration_frames", "path_length", "displacement", "avg_speed",
    "max_speed", "start_speed", "end_speed", "spread_x", "spread_y", "mean_to_centre",
    "max_to_centre", "start_to_centre", "end_to_centre", "box_w", "box_h", "body_length",
]


def stay_features(centroids, frames, box, body_lengths=()):
    """Features of a stay: how long, how much it moved, how far from the
    reference's centre, in pixels and frames, relative to the reference's
    size where a distance is compared with it."""
    import numpy as np

    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    w, h = max(box[2] - box[0], 1e-6), max(box[3] - box[1], 1e-6)
    size = (w * w + h * h) ** 0.5
    steps = [float(np.hypot(centroids[i + 1][0] - centroids[i][0],
                            centroids[i + 1][1] - centroids[i][1]))
             for i in range(len(centroids) - 1)]
    third = max(len(steps) // 3, 1)
    to_centre = [float(np.hypot(x - cx, y - cy)) / size for x, y in centroids]
    bls = [b for b in body_lengths if b and b == b]
    return {
        "trajectory_length": len(centroids),
        "duration_frames": (frames[-1] - frames[0]) if frames else 0,
        "path_length": sum(steps),
        "displacement": float(np.hypot(centroids[-1][0] - centroids[0][0],
                                       centroids[-1][1] - centroids[0][1])),
        "avg_speed": float(np.mean(steps)) if steps else 0.0,
        "max_speed": float(np.max(steps)) if steps else 0.0,
        "start_speed": float(np.mean(steps[:third])) if steps else 0.0,
        "end_speed": float(np.mean(steps[-third:])) if steps else 0.0,
        "spread_x": float(np.std([c[0] for c in centroids])) / w,
        "spread_y": float(np.std([c[1] for c in centroids])) / h,
        "mean_to_centre": float(np.mean(to_centre)),
        "max_to_centre": float(np.max(to_centre)),
        "start_to_centre": to_centre[0],
        "end_to_centre": to_centre[-1],
        "box_w": w, "box_h": h,
        "body_length": float(np.mean(bls)) if bls else 0.0,
    }


def track_paths(result, video):
    """``{track id: (centroids, frames, body lengths)}`` in pixels from the
    run's tracking table (detected rows), or a note saying why not."""
    from . import ops

    width, height = getattr(video, "width", None), getattr(video, "height", None)
    df = ops.load_tracking_df(result)
    if df is None or not (width and height):
        return None, "needs the clip's tracking table and frame size"
    det_col = ops._pick(df, ["detected"])
    if det_col is not None:
        df = df[df[det_col].fillna(0).astype(int) == 1]
    id_col, frame_col = ops._pick(df, ops._ID_COLS), ops._pick(df, ops._FRAME_COLS)
    boxes = [ops._pick(df, ops._BBOX[k]) for k in ("x1", "y1", "x2", "y2")]
    centres = [ops._pick(df, ops._CX_COLS), ops._pick(df, ops._CY_COLS)]
    if id_col is None or frame_col is None or (None in boxes and None in centres):
        return None, "needs box columns in the tracking table"
    out = {}
    for tid, grp in df.sort_values(frame_col).groupby(id_col):
        if None not in boxes:
            x1, y1, x2, y2 = (grp[c].astype(float) for c in boxes)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            sizes = [max(a, b) for a, b in zip((x2 - x1).tolist(), (y2 - y1).tolist())]
        else:
            # Centres only (no boxes): the features need nothing else; a
            # stay's body length is then unknown.
            cx, cy = (grp[c].astype(float) for c in centres)
            sizes = []
        out[str(ops._as_native(tid))] = (list(zip(cx.tolist(), cy.tolist())),
                                         [int(f) for f in grp[frame_col]], sizes)
    return out, None


def ref_boxes(refs, video):
    """``{reference id: box in pixels}``."""
    w, h = getattr(video, "width", None) or 0, getattr(video, "height", None) or 0
    return {r["id"]: (r["box"][0] * w, r["box"][1] * h, r["box"][2] * w, r["box"][3] * h)
            for r in refs}


def features_for(row, paths, boxes):
    """``(kind, features)`` for one event row: "exit_enter" or "stay"; None
    when its track or reference can't be found."""
    from . import ops

    traj = paths.get(str(row.get("track_id")))
    box = boxes.get(row.get("reference"))
    if not traj or box is None or not traj[0]:
        return None
    if row.get("action") == ops.STAY:
        return "stay", stay_features(traj[0], traj[1], box, traj[2])
    return "exit_enter", trajectory_features(traj[0], traj[1], box, row.get("action") == ops.ENTER)


# ── Models ──────────────────────────────────────────────────────────────────

def _model_id(spec):
    spec = str(spec or "")
    return int(spec.split(":", 1)[1]) if spec.startswith("model:") and spec[6:].isdigit() else None


@lru_cache(maxsize=16)
def _trained(model_pk, storage_key):
    """A trained event classifier (memory/70): ``(model, feature names, kind)``."""
    import io

    import joblib

    from . import eventml

    data = joblib.load(io.BytesIO(eventml.read_model(storage_key)))
    return data["model"], list(data["feature_names"]), data["kind"]


def load(spec):
    """``(model, feature names, kind)`` for an Events node's choice: the
    built-in exit/enter model ("on"), a trained one ("model:<id>"), or None."""
    if spec in (True, "true", "on", "1", 1, "builtin"):
        loaded = _model()
        return (loaded[0], loaded[1], "exit_enter") if loaded else None
    pk = _model_id(spec)
    if pk is None:
        return None
    from apps.training.models import CustomModel

    m = CustomModel.objects.filter(pk=pk, model_type="event_classifier").first()
    if m is None or not m.storage_key:
        return None
    try:
        return _trained(m.pk, m.storage_key)
    except Exception:  # noqa: BLE001 - reported as "could not be loaded"
        logger.exception("could not load event classifier %s", pk)
        return None


def score(rows, result, refs, extra, spec="on", stay_spec=None):
    """Fill ``ml_confidence`` in place: exits and entries with ``spec`` (the
    built-in model, or a trained exit/enter one), stays with ``stay_spec`` (a
    trained stay model) when given. Returns a note when scoring could not run,
    else None."""
    import pandas as pd

    from . import ops

    plans = [(spec, (ops.EXIT, ops.ENTER), "exit_enter"), (stay_spec, (ops.STAY,), "stay")]
    plans = [(s, acts, kind) for s, acts, kind in plans
             if s and any(r["action"] in acts for r in rows)]
    if not plans:
        return None
    notes, ready = [], []
    for s, acts, kind in plans:
        loaded = load(s)
        if loaded is None or loaded[2] != kind:
            if kind == "exit_enter" and s in (True, "true", "on", "1", 1, "builtin"):
                notes.append("is on but the classifier could not be loaded")
            else:
                notes.append(f"could not load the {kind.replace('_', '/')} classifier")
            continue
        ready.append((acts, loaded))
    if not ready:
        return "ML scoring " + "; ".join(notes) + "; ml_confidence is blank."
    paths, why = track_paths(result, extra.get("video"))
    if paths is None:
        return f"ML scoring {why}; ml_confidence is blank."
    boxes = ref_boxes(refs, extra.get("video"))
    for acts, loaded in ready:
        model, names, _kind = loaded
        feats, scored = [], []
        for row in rows:
            if row["action"] not in acts:
                continue
            got = features_for(row, paths, boxes)
            if got is None:
                continue
            feats.append(got[1])
            scored.append(row)
        if scored:
            probs = model.predict_proba(pd.DataFrame(feats)[names])[:, 1]
            for row, p in zip(scored, probs):
                row["ml_confidence"] = round(float(p), 4)
    if notes:
        return "ML scoring " + "; ".join(notes) + "; those events' ml_confidence is blank."
    return None
