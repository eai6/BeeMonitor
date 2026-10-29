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


def score(rows, result, refs, extra):
    """Fill ``ml_confidence`` on the exit/enter rows in place. Returns a note
    when scoring could not run, else None."""
    from . import ops

    targets = [r for r in rows if r["action"] in (ops.EXIT, ops.ENTER)]
    if not targets:
        return None
    loaded = _model()
    if loaded is None:
        return "ML scoring is on but the classifier could not be loaded; ml_confidence is blank."
    model, names = loaded

    video = extra.get("video")
    width, height = getattr(video, "width", None), getattr(video, "height", None)
    df = ops.load_tracking_df(result)
    if df is None or not (width and height):
        return "ML scoring needs the clip's tracking table and frame size; ml_confidence is blank."

    import pandas as pd

    det_col = ops._pick(df, ["detected"])
    if det_col is not None:
        df = df[df[det_col].fillna(0).astype(int) == 1]
    id_col, frame_col = ops._pick(df, ops._ID_COLS), ops._pick(df, ops._FRAME_COLS)
    boxes = [ops._pick(df, ops._BBOX[k]) for k in ("x1", "y1", "x2", "y2")]
    if id_col is None or frame_col is None or None in boxes:
        return "ML scoring needs box columns in the tracking table; ml_confidence is blank."

    by_track = {}
    for tid, grp in df.sort_values(frame_col).groupby(id_col):
        x1, y1, x2, y2 = (grp[c].astype(float) for c in boxes)
        by_track[str(ops._as_native(tid))] = (
            list(zip(((x1 + x2) / 2).tolist(), ((y1 + y2) / 2).tolist())),
            [int(f) for f in grp[frame_col]])
    ref_px = {r["id"]: (r["box"][0] * width, r["box"][1] * height,
                        r["box"][2] * width, r["box"][3] * height) for r in refs}

    feats, scored = [], []
    for row in targets:
        traj = by_track.get(str(row["track_id"]))
        box = ref_px.get(row["reference"])
        if not traj or box is None or len(traj[0]) < 1:
            continue
        feats.append(trajectory_features(traj[0], traj[1], box, row["action"] == ops.ENTER))
        scored.append(row)
    if not scored:
        return None
    probs = model.predict_proba(pd.DataFrame(feats)[names])[:, 1]
    for row, p in zip(scored, probs):
        row["ml_confidence"] = round(float(p), 4)
    return None
