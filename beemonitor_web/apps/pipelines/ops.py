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


def _read_csv(path):
    """Read a CSV at ``path`` (s3:// or local) into a DataFrame, or None."""
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
        logger.info("Could not read CSV %s: %s", path, exc)
        return None


def load_tracking_df(job_result):
    """Read the job's ``tracking_csv_path`` into a pandas DataFrame or None."""
    return _read_csv((job_result or {}).get("tracking_csv_path") or "")


def load_interactions_df(job_result):
    """Read the job's ``interactions_csv_path`` into a pandas DataFrame or None."""
    return _read_csv((job_result or {}).get("interactions_csv_path") or "")


def load_detections_df(job_result):
    """Read the job's raw (pre-association) detections CSV, or None.

    Only jobs run since the worker started emitting it have one; older jobs
    return None and callers fall back to the tracked table.
    """
    return _read_csv((job_result or {}).get("detections_csv_path") or "")


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


def _points(raw):
    """A polygon outline as [(x, y), ...] in 0..1, or None if it isn't one."""
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return None
    out = []
    for p in raw:
        try:
            x, y = (float(v) for v in p)
        except (TypeError, ValueError):
            return None
        out.append((x, y))
    return out


def roi_shapes(roi_output):
    """Normalise a roi-step output into a list of 0..1 shapes.

    Each shape is ``(box, points)``: the box is always present, and ``points`` is
    the traced outline when the user drew a polygon rather than a rectangle (then
    the box is merely its bounding box). Containment tests use the points, so a
    bee over the grass beside a round trap is not counted as a visit.
    """
    if not roi_output:
        return []
    shapes = []

    def _add(box, points=None):
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return
        shapes.append(((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)),
                       _points(points)))

    def _add_shape(obj):
        """One {box, points?} dict, or a bare box."""
        if isinstance(obj, dict):
            if obj.get("box"):
                _add(obj["box"], obj.get("points"))
        elif isinstance(obj, (list, tuple)):
            _add(obj)

    hotel = roi_output.get("hotel_roi")
    if hotel:
        _add(hotel, roi_output.get("hotel_polygon"))
    for tube in roi_output.get("nest_layout") or []:
        _add_shape(tube)
    for region in roi_output.get("regions") or []:
        _add_shape(region)
    return shapes


def roi_boxes(roi_output):
    """Just the bounding boxes of ``roi_shapes`` — for callers that can't do
    polygons (e.g. anything handing geometry to a box-only API)."""
    return [box for box, _points in roi_shapes(roi_output)]


def _in_polygon(x, y, points):
    """Ray casting: is (x, y) inside the polygon? Handles concave outlines."""
    inside = False
    n = len(points)
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[i - 1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
    return inside


def in_any_box(x, y, shapes):
    """Is (x, y) inside any shape? Accepts ``roi_shapes`` output or bare boxes."""
    for shape in shapes:
        if len(shape) == 2 and not isinstance(shape[0], (int, float)):
            (x1, y1, x2, y2), points = shape
        else:
            (x1, y1, x2, y2), points = shape, None
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue          # outside the bounding box — cheap reject
        if points is None or _in_polygon(x, y, points):
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


def filter_by_label(df, label):
    """Keep only rows whose taxon matches ``label`` (case-insensitive).

    This is what makes one GPU pass serve several Detect nodes: every node reads
    the same table and takes its own class. An empty label means "no filter", and
    a table with no taxon column is passed through unchanged rather than emptied —
    older results predate the column, and silently returning nothing would look
    like "no detections" instead of "can't tell".
    """
    if df is None or len(df) == 0 or not label:
        return df
    col = _pick(df, ["taxon", "label", "class", "class_name"])
    if col is None:
        return df
    wanted = {p.strip().lower() for p in str(label).split(",") if p.strip()}
    if not wanted:
        return df
    return df[df[col].astype(str).str.strip().str.lower().isin(wanted)]


def boxes_for_label(df, label, max_boxes=200):
    """Distinct 0..1 boxes for a label — a detected reference object.

    One box per detected instance: rows are per-frame, so the same nest tube
    appears in every frame. Dedupes on rounded coordinates to collapse those back
    into the handful of real objects.
    """
    df = filter_by_label(df, label)
    if df is None or len(df) == 0:
        return []
    cols = {k: _pick(df, v) for k, v in _BBOX.items()}
    if any(c is None for c in cols.values()):
        return []
    seen, boxes = set(), []
    for _, r in df.iterrows():
        try:
            box = tuple(round(float(r[cols[k]]), 3) for k in ("x1", "y1", "x2", "y2"))
        except (TypeError, ValueError):
            continue
        if box in seen:
            continue
        seen.add(box)
        boxes.append(list(box))
        if len(boxes) >= max_boxes:
            break
    return boxes


def compute_detection_counts(tidy, boxes, fps, per_frame=False, count_tracks=True):
    """Detection totals from a tidy [tid, frame, x, y] table.

    One row is one detection. If ``boxes`` is non-empty only detections inside
    the reference count. With ``per_frame`` the rows are per-frame counts;
    otherwise just the totals.

    ``count_tracks=False`` for the raw detections table, whose rows carry no real
    track id — reporting a track count there would just restate the detection
    count.
    """
    empty = {"detections": 0, "frames_with_detections": 0, "mean_per_frame": 0,
             "rows": []}
    if count_tracks:
        empty["unique_tracks"] = 0
    if tidy is None or len(tidy) == 0:
        return empty
    df = tidy
    if boxes:
        mask = [in_any_box(x, y, boxes) for x, y in zip(df["x"], df["y"])]
        df = df[mask]
    if len(df) == 0:
        return empty

    frames = df["frame"].nunique()
    summary = {
        "detections": int(len(df)),
        "frames_with_detections": int(frames),
        "mean_per_frame": round(len(df) / frames, 2) if frames else 0,
        "rows": [],
    }
    if count_tracks:
        summary["unique_tracks"] = int(df["tid"].nunique())
    if per_frame:
        counts = df.groupby("frame").size()
        summary["rows"] = [
            {"frame": int(f), "t_sec": round(int(f) / fps, 2) if fps else None,
             "detections": _as_native(n)}
            for f, n in counts.items()
        ]
    return summary


def summarize_interactions(df, kind=None):
    """Aggregate the interactions CSV into a summary + per-interaction rows.

    ``kind`` filters on the worker's ``interaction_type`` literals
    (``organism-to-organism`` / ``organism-to-reference``); None keeps both.
    Schema-tolerant like the tracking reader — column names have drifted.
    """
    if df is None or len(df) == 0:
        return {"interaction_count": 0, "organism_organism": 0,
                "organism_reference": 0, "rows": []}

    type_col = _pick(df, ["interaction_type", "type", "kind"])
    if type_col is not None and kind:
        df = df[df[type_col].astype(str) == kind]
    if len(df) == 0:
        return {"interaction_count": 0, "organism_organism": 0,
                "organism_reference": 0, "rows": []}

    def _count(literal):
        if type_col is None:
            return 0
        return int((df[type_col].astype(str) == literal).sum())

    cols = {
        "type": type_col,
        "a": _pick(df, ["organism_track_id", "entity1_id", "track_id"]),
        "b": _pick(df, ["partner_track_id", "entity2_id"]),
        "reference": _pick(df, ["reference_id", "nest", "nest_id"]),
        "duration": _pick(df, ["duration_seconds", "duration_sec", "duration"]),
        "start": _pick(df, ["start_frame", "frame_start", "frame"]),
    }
    rows = []
    for _, r in df.iterrows():
        row = {}
        for key, col in cols.items():
            if col is not None:
                row[key] = _as_native(r[col])
        rows.append(row)
    durations = (
        [float(r["duration"]) for r in rows if r.get("duration") is not None]
        if cols["duration"] else []
    )
    return {
        "interaction_count": int(len(df)),
        "organism_organism": _count("organism-to-organism"),
        "organism_reference": _count("organism-to-reference"),
        "total_duration_sec": round(sum(durations), 2) if durations else None,
        "rows": rows,
    }


def _iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def sampled_boxes(frames, label=""):
    """Flatten a sampled-detection result into [(frame, box, confidence), ...].

    ``frames`` is the worker's sampled-frame payload:
    ``[{frame_number, boxes: [{x, y, w, h, class, confidence}]}, ...]`` — boxes in
    native pixels, x/y/w/h rather than corners.
    """
    wanted = {p.strip().lower() for p in str(label).split(",") if p.strip()}
    out = []
    for frame in frames or []:
        n = frame.get("frame_number")
        for b in frame.get("boxes") or []:
            if wanted and str(b.get("class", "")).strip().lower() not in wanted:
                continue
            try:
                x, y = float(b["x"]), float(b["y"])
                w, h = float(b["w"]), float(b["h"])
            except (KeyError, TypeError, ValueError):
                continue
            out.append((n, (x, y, x + w, y + h), float(b.get("confidence") or 0.0)))
    return out


def count_distinct_objects(frames, label="", iou_threshold=0.5):
    """Count physically distinct objects across sampled frames.

    A static object — a nest tube, a flower — appears in *every* sampled frame,
    so summing detections multiplies it by the frame count. This clusters boxes
    that overlap across frames so one real object counts once, however many
    frames saw it. Greedy agglomeration against cluster representatives: cheap,
    order-stable, and sufficient because the objects don't move.

    Boxes are matched by IoU rather than exact coordinates because detector
    output jitters by a few pixels between frames — exact matching would report
    one object per frame.
    """
    detections = sampled_boxes(frames, label)
    if not detections:
        return {"distinct_objects": 0, "rows": [], "frames_sampled": len(frames or [])}

    # Strongest detections first, so each cluster is anchored on its best box.
    detections.sort(key=lambda d: -d[2])
    clusters = []
    for _frame, box, conf in detections:
        for c in clusters:
            if _iou(c["box"], box) >= iou_threshold:
                c["hits"] += 1
                c["confidence"] += conf
                break
        else:
            clusters.append({"box": box, "hits": 1, "confidence": conf})

    rows = []
    for i, c in enumerate(sorted(clusters, key=lambda c: (c["box"][1], c["box"][0])), 1):
        x1, y1, x2, y2 = c["box"]
        rows.append({
            "object": i,
            "x1": round(x1, 1), "y1": round(y1, 1),
            "x2": round(x2, 1), "y2": round(y2, 1),
            "seen_in_frames": c["hits"],
            "confidence": round(c["confidence"] / c["hits"], 3),
        })
    return {"distinct_objects": len(rows), "rows": rows,
            "frames_sampled": len(frames or [])}


def modal_frame_count(frames, label=""):
    """Most common per-frame detection count across sampled frames.

    For a static scene every frame should see every object, so the modal count is
    a robust estimate that ignores the odd frame where one was missed or
    double-detected. Cheaper and steadier than clustering, but it yields only a
    number — no per-object boxes — and undercounts objects occluded in most
    frames.
    """
    from collections import Counter

    detections = sampled_boxes(frames, label)
    per_frame = Counter()
    for n, _box, _conf in detections:
        per_frame[n] += 1
    # Sampled frames with no detections are real zeros and must count.
    counts = [per_frame.get(f.get("frame_number"), 0) for f in frames or []]
    if not counts:
        return {"modal_count": 0, "rows": [], "frames_sampled": 0}
    tally = Counter(counts)
    modal = max(tally, key=lambda c: (tally[c], c))
    return {
        "modal_count": modal,
        "frames_sampled": len(counts),
        "frames_agreeing": tally[modal],
        "rows": [{"count": c, "frames": n} for c, n in sorted(tally.items())],
    }


def species_identities(df):
    """Per-track species from the tracking CSV's taxon columns.

    The voting already happened on the GPU during tracking (every frame of a
    trajectory got a say), so ``taxon`` is constant within a track and this is a
    read, not a re-aggregation. Returns None when the CSV carries no
    ``taxon_votes`` column — that means the run predates species classification,
    and the plain ``taxon`` there is just the detector's class label, which would
    be misleading to report as an identification.
    """
    if df is None or len(df) == 0:
        return None
    id_col = _pick(df, _ID_COLS)
    taxon_col = _pick(df, ["taxon"])
    votes_col = _pick(df, ["taxon_votes"])
    if id_col is None or taxon_col is None or votes_col is None:
        return None
    conf_col = _pick(df, ["taxon_confidence"])

    rows, seen = [], set()
    for tid, grp in df.groupby(id_col):
        votes = int(grp[votes_col].fillna(0).max() or 0)
        if votes <= 0:
            continue  # detector label only — nothing was actually classified
        taxon = str(grp[taxon_col].dropna().iloc[0]) if grp[taxon_col].notna().any() else ""
        if not taxon:
            continue
        confidence = None
        if conf_col is not None and grp[conf_col].notna().any():
            confidence = round(float(grp[conf_col].dropna().mean()), 3)
        rows.append({
            "track": _as_native(tid), "taxon": taxon,
            "confidence": confidence, "votes": votes,
            "frames": int(len(grp)),
        })
        seen.add(taxon)
    if not rows:
        return None
    return {"identified_tracks": len(rows), "unique_taxa": len(seen), "rows": rows}


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
