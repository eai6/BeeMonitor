"""
Shared post-processing ops for pipeline analyze steps (Phase 1.5).

These turn the tracking CSV produced by the ``detect_and_track`` Job into ecological
aggregates — visitation counts and colony-activity time series — entirely on the
web side (no extra GPU). They are deliberately **schema-tolerant**: the exact
tracking-CSV header has varied across tracker versions, so we detect the
track-id / frame / centroid columns by trying common names and normalise
coordinates to 0..1 whether the CSV stores pixels or fractions.

If the CSV can't be read (e.g. not present in a dev DB), callers fall back to the
Job summary. See ``memory/23_pipeline_builder_port_design.md`` §Phasing (Phase 1.5).
"""

import logging

logger = logging.getLogger(__name__)

# Candidate column names, most-specific first.
_ID_COLS = ["track_id", "track", "tid", "id", "object_id", "particle"]
_FRAME_COLS = ["frame", "frame_num", "frame_number", "frame_idx", "frame_id", "t"]
_CX_COLS = ["cx", "centroid_x", "x_center", "xc", "x", "cent_x"]
_CY_COLS = ["cy", "centroid_y", "y_center", "yc", "y", "cent_y"]
_BBOX = {
    "x1": ["x1", "xmin", "bbox_x1", "left"],
    "y1": ["y1", "ymin", "bbox_y1", "top"],
    "x2": ["x2", "xmax", "bbox_x2", "right"],
    "y2": ["y2", "ymax", "bbox_y2", "bottom"],
}


def _pick(df, candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_tracking_df(job_result):
    """Read ``tracking_csv_path`` (s3:// or local) into a pandas DataFrame or None."""
    if not job_result:
        return None
    path = job_result.get("tracking_csv_path") or ""
    if not path:
        return None
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas unavailable — cannot post-process tracking CSV")
        return None
    try:
        if path.startswith("s3://"):
            import boto3
            from urllib.parse import urlparse
            from io import BytesIO
            from django.conf import settings

            parsed = urlparse(path)
            s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
            body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
            return pd.read_csv(BytesIO(body))
        return pd.read_csv(path)
    except Exception as exc:
        logger.info("Could not read tracking CSV %s: %s", path, exc)
        return None


def normalized_tracks(df, summary=None):
    """Return a tidy DataFrame with columns [tid, frame, x, y] normalised to 0..1.

    Returns None if the essential columns can't be located.
    """
    if df is None or len(df) == 0:
        return None
    id_col = _pick(df, _ID_COLS)
    frame_col = _pick(df, _FRAME_COLS)
    cx_col = _pick(df, _CX_COLS)
    cy_col = _pick(df, _CY_COLS)

    out = df.copy()
    # Derive centroid from bbox if no explicit centroid columns.
    if cx_col is None or cy_col is None:
        bx1, by1 = _pick(df, _BBOX["x1"]), _pick(df, _BBOX["y1"])
        bx2, by2 = _pick(df, _BBOX["x2"]), _pick(df, _BBOX["y2"])
        if None not in (bx1, by1, bx2, by2):
            out["_cx"] = (out[bx1] + out[bx2]) / 2.0
            out["_cy"] = (out[by1] + out[by2]) / 2.0
            cx_col, cy_col = "_cx", "_cy"
    if cx_col is None or cy_col is None or id_col is None:
        return None
    if frame_col is None:
        out["_frame"] = range(len(out))
        frame_col = "_frame"

    tidy = out[[id_col, frame_col, cx_col, cy_col]].copy()
    tidy.columns = ["tid", "frame", "x", "y"]
    tidy = tidy.dropna(subset=["tid", "x", "y"])

    # Normalise coords: if values look like pixels (max > 1.5), divide by frame
    # dims (from summary) or by the observed max as a last resort.
    def _norm(series, dim_keys):
        m = float(series.abs().max() or 0)
        if m <= 1.5:
            return series  # already fractional
        dim = None
        for k in dim_keys:
            if summary and summary.get(k):
                dim = float(summary[k]); break
        if not dim:
            dim = m
        return series / dim

    tidy["x"] = _norm(tidy["x"], ["frame_width", "width", "res_width", "video_width"])
    tidy["y"] = _norm(tidy["y"], ["frame_height", "height", "res_height", "video_height"])
    return tidy


def roi_boxes(roi_output):
    """Normalise a roi-step output into a list of 0..1 boxes [(x1,y1,x2,y2), ...]."""
    if not roi_output:
        return []
    boxes = []

    def _add(box):
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        except (TypeError, ValueError):
            pass

    hotel = roi_output.get("hotel_roi")
    if hotel:
        _add(hotel)
    layout = roi_output.get("nest_layout") or []
    for tube in layout:
        if isinstance(tube, dict) and tube.get("box"):
            _add(tube["box"])
    for region in roi_output.get("regions") or []:
        if isinstance(region, dict) and region.get("box"):
            _add(region["box"])
        elif isinstance(region, (list, tuple)):
            _add(region)
    return boxes


def in_any_box(x, y, boxes):
    for (x1, y1, x2, y2) in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def fps_of(summary, default=30.0):
    for k in ("fps", "video_fps", "frame_rate"):
        if summary and summary.get(k):
            try:
                return float(summary[k])
            except (TypeError, ValueError):
                pass
    return default


def compute_visitation(tidy, boxes, fps, gap_frames=15):
    """Count ROI visits per track.

    A *visit* is a contiguous run of in-ROI frames for a track (runs separated by
    more than ``gap_frames`` out-of-ROI frames count as separate visits). Returns a
    summary dict + per-track rows.
    """
    rows = []
    total_visits = 0
    dwell_frames_total = 0
    for tid, grp in tidy.sort_values("frame").groupby("tid"):
        inside = [(int(f), in_any_box(x, y, boxes)) for f, x, y in zip(grp["frame"], grp["x"], grp["y"])]
        visits, dwell = 0, 0
        run_open, last_in = False, None
        for frame, is_in in inside:
            if is_in:
                dwell += 1
                if not run_open or (last_in is not None and frame - last_in > gap_frames):
                    visits += 1
                run_open, last_in = True, frame
        if visits:
            rows.append({
                "track": _as_native(tid),
                "visits": visits,
                "dwell_sec": round(dwell / fps, 2) if fps else None,
            })
            total_visits += visits
            dwell_frames_total += dwell
    return {
        "unique_visitors": len(rows),
        "total_visits": total_visits,
        "total_dwell_sec": round(dwell_frames_total / fps, 2) if fps else None,
        "rows": rows,
    }


def compute_colony_activity(tidy, boxes, fps, metric="occupancy", bin_sec=5.0):
    """Time-binned colony-activity series.

    occupancy = distinct tracks present per time bin; motion = detections per bin.
    If ``boxes`` is non-empty, only in-ROI detections count. Returns rows
    [{t_sec, value}] plus peak/mean.
    """
    if tidy is None or len(tidy) == 0:
        return {"metric": metric, "rows": [], "peak": 0, "mean": 0}
    df = tidy
    if boxes:
        mask = [in_any_box(x, y, boxes) for x, y in zip(df["x"], df["y"])]
        df = df[mask]
    bin_frames = max(1, int(round(bin_sec * fps)))
    rows = []
    if len(df) == 0:
        return {"metric": metric, "rows": [], "peak": 0, "mean": 0}
    df = df.assign(_bin=(df["frame"] // bin_frames).astype(int))
    for b, grp in df.groupby("_bin"):
        value = grp["tid"].nunique() if metric == "occupancy" else int(len(grp))
        rows.append({"t_sec": round(b * bin_frames / fps, 1) if fps else int(b), "value": _as_native(value)})
    values = [r["value"] for r in rows] or [0]
    return {
        "metric": metric,
        "rows": rows,
        "peak": max(values),
        "mean": round(sum(values) / len(values), 2),
    }


def marker_identities(df):
    """Aggregate per-track individual IDs from the tracking CSV's marker columns.

    The tracker emits ``bee_id`` / ``bee_id_method`` (color|number|qrcode) /
    ``bee_id_confidence`` when individual identification is enabled. This picks each
    track's dominant (most-frequent) marker. Returns None if the CSV carries no
    marker data (identification wasn't enabled upstream).
    """
    if df is None or len(df) == 0:
        return None
    id_col = _pick(df, _ID_COLS)
    bee_col = _pick(df, ["bee_id", "individual_id", "beeid"])
    if id_col is None or bee_col is None:
        return None
    method_col = _pick(df, ["bee_id_method", "id_method"])
    conf_col = _pick(df, ["bee_id_confidence", "id_confidence"])

    rows, markers = [], set()
    for tid, grp in df.groupby(id_col):
        vals = grp[bee_col].dropna()
        vals = vals[vals.astype(str).str.strip().str.len() > 0]
        if len(vals) == 0:
            continue
        mode = vals.astype(str).mode()
        marker = mode.iloc[0] if len(mode) else str(vals.iloc[0])
        method = ""
        if method_col is not None and grp[method_col].notna().any():
            method = str(grp[method_col].dropna().iloc[0])
        conf = None
        if conf_col is not None and grp[conf_col].notna().any():
            conf = round(float(grp[conf_col].dropna().mean()), 3)
        rows.append({
            "track": _as_native(tid), "marker": marker,
            "method": method, "confidence": conf, "frames": int(len(vals)),
        })
        markers.add(marker)
    return {"identified_tracks": len(rows), "unique_markers": len(markers), "rows": rows}


def _as_native(v):
    """Coerce numpy scalars to JSON-serialisable Python types."""
    try:
        return v.item()
    except AttributeError:
        return v
