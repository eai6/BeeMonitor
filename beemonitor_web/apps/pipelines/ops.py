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


def load_events_df(job_result):
    """Read the job's ``events_csv_path`` (worker Entry/Exit) into a DataFrame."""
    return _read_csv((job_result or {}).get("events_csv_path") or "")


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


def roi_references(roi_output):
    """The ROI's shapes WITH their identities.

    ``roi_shapes`` returns geometry only, which is why every analyzer built on it
    can say a track was inside *something* and never inside *which* — and a
    treatment comparison is exactly the "which" question. This keeps the id the
    layout already carries.

    Each entry is ``{"id", "label", "box", "points"}``:

    * a nest tube keeps the ``id`` from the device layout ("nest 3");
    * a drawn region keeps its own ``id``/``name`` when the editor stored one,
      otherwise its 1-based index ("region 2");
    * the hotel ROI is its own reference, since a pipeline may count visits to
      the hotel as a whole.

    Ids are stable within one layout, which is what makes them comparable across
    clips in a batch. They are not names — nothing in the editor lets a user call
    one "full UV" yet — so ``label`` is a readable fallback, not a title.
    """
    if not roi_output:
        return []

    refs = []

    def _shape(box, points=None):
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return None
        return ((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), _points(points))

    def _add(ref_id, label, box, points=None):
        shape = _shape(box, points)
        if shape:
            refs.append({"id": str(ref_id), "label": label,
                         "box": shape[0], "points": shape[1]})

    hotel = roi_output.get("hotel_roi")
    if hotel:
        _add("hotel", "Hotel", hotel, roi_output.get("hotel_polygon"))

    for n, tube in enumerate(roi_output.get("nest_layout") or [], start=1):
        if isinstance(tube, dict):
            tube_id = tube.get("id", n)
            _add(f"nest_{tube_id}", f"Nest {tube_id}", tube.get("box"), tube.get("points"))
        else:
            _add(f"nest_{n}", f"Nest {n}", tube)

    for n, region in enumerate(roi_output.get("regions") or [], start=1):
        if isinstance(region, dict):
            # An editor that learns to name regions should put it in `name`;
            # until then the index is the identity.
            region_id = region.get("id", n)
            label = region.get("name") or f"Region {region_id}"
            _add(f"region_{region_id}", label, region.get("box"), region.get("points"))
        else:
            _add(f"region_{n}", f"Region {n}", region)

    return refs


def detected_references(job_result, video=None):
    """References the DETECTOR found, when the graph defines none.

    The worker writes the nest/reference boxes it detected into
    ``summary_stats["nest_bboxes"]``, and until now nothing local ever read
    them. A pipeline whose reference class is *detected* rather than drawn —
    flowers on a board, say — therefore ran its analyzers against an empty
    reference list and reported "0 references, 0 visits" while the job page
    cheerfully said it had found four nests. The geometry was there; nobody
    handed it over.

    These are PIXEL coordinates (the annotator draws them straight onto the
    frame), whereas references must be normalised 0..1 to match the tracks. The
    frame size comes from the video row — measured at ingest — falling back to
    the summary, and when neither knows we return nothing rather than emit
    references at the wrong scale.
    """
    stats = (job_result or {}).get("summary_stats") or {}
    boxes = stats.get("nest_bboxes") or {}
    hotel = stats.get("hotel_bbox")
    if not boxes and not hotel:
        return []

    width = height = None
    for source, w_key, h_key in ((video, "width", "height"),
                                 (stats, "frame_width", "frame_height"),
                                 (stats, "width", "height")):
        w = getattr(source, w_key, None) if video is source else (source or {}).get(w_key)
        h = getattr(source, h_key, None) if video is source else (source or {}).get(h_key)
        try:
            if w and h and float(w) > 0 and float(h) > 0:
                width, height = float(w), float(h)
                break
        except (TypeError, ValueError):
            continue

    def _norm(box):
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return None
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
            if not width:
                return None              # pixels with no frame size: refuse to guess
            x1, y1, x2, y2 = x1 / width, y1 / height, x2 / width, y2 / height
        # Ordered like roi_references does: containment tests read x1 <= x <= x2,
        # so a box given corner-reversed would match nothing at all.
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    refs = []
    for box_id, box in boxes.items():
        shape = _norm(box)
        if shape:
            refs.append({"id": f"nest_{box_id}", "label": _reference_label(box_id),
                         "box": shape, "points": None})
    if hotel and not refs:
        # Only when nothing finer was found — the hotel contains every tube, so
        # counting both would double every episode.
        shape = _norm(hotel)
        if shape:
            refs.append({"id": "hotel", "label": "Hotel", "box": shape, "points": None})
    return refs


def which_reference(x, y, refs):
    """The FIRST reference containing (x, y), or None.

    The counterpart to ``in_any_box``, which answers only yes/no. First-match
    rather than all-matches because references can nest — a tube sits inside the
    hotel ROI — and the tube is the more specific, more useful answer. Ordering
    from ``roi_references`` puts the hotel first, so callers that want tube-level
    detail should exclude it rather than rely on order.
    """
    for ref in refs:
        x1, y1, x2, y2 = ref["box"]
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            continue
        if ref["points"] is None or _in_polygon(x, y, ref["points"]):
            return ref
    return None


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


# The rate assumed when a clip records none. It is deliberately a module
# constant and not an inline literal: a guessed frame rate silently rescales
# every duration in the system, so there is exactly one place it can come from.
DEFAULT_FPS = 30.0

# Keys the GPU backend has used for the frame rate over the life of the
# project. ``video_fps`` is what ``cloud/wrapper/pipeline.py`` writes today;
# the others are older runs still in the database.
_FPS_KEYS = ("video_fps", "fps", "frame_rate")


def fps_with_source(summary=None, video=None, default=DEFAULT_FPS):
    """Resolve a clip's frame rate and say where the number came from.

    Returns ``(fps, source)`` where source is ``"video"`` (measured from the
    file at ingest), ``"analysis"`` (reported by the GPU run) or ``"assumed"``
    (nothing recorded one — the caller should disclose this rather than
    present the derived seconds as measured).

    Precedence puts the video row first: it is measured from the container by
    ``videos.thumbnails``, whereas the analysis value is whatever OpenCV
    reported on the GPU host for a copy of the same file.
    """
    measured = None
    try:
        measured = float(getattr(video, "fps", None) or 0) or None
    except (TypeError, ValueError):
        measured = None
    if measured and measured > 0:
        return measured, "video"

    # Callers pass either the GPU result dict or the summary_stats inside it.
    # PipelineResult.to_dict() is a plain asdict(), so the frame rate lives one
    # level down under "summary_stats" — a resolver that only checked the top
    # level found nothing and quietly assumed 30 for every analyzer.
    for scope in (summary, (summary or {}).get("summary_stats")):
        for key in _FPS_KEYS:
            if scope and scope.get(key):
                try:
                    value = float(scope[key])
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    return value, "analysis"

    return float(default), "assumed"


def fps_of(summary=None, video=None, default=DEFAULT_FPS):
    """The frame rate alone, for callers with nothing to disclose it to."""
    return fps_with_source(summary, video, default)[0]


def compute_episodes(tidy, refs, gap_frames=15):
    """Contiguous spells each track spends inside each reference.

    An *episode* is a run of frames one track stays inside one reference; a gap
    of more than ``gap_frames`` starts a new one. Moving from one reference to
    another also ends the current episode — otherwise a bee crossing from tube 3
    to tube 7 would read as a single long stay in neither.

    This is the single geometric pass the whole analyze layer stands on. A visit
    is an episode; an interaction with a reference is an episode; the enter and
    exit events are an episode's two ends. Computing it once is what lets those
    three stop being three code paths that can disagree about the same clip.

    Episodes are returned in frame order and carry frames, not seconds: the
    frame rate is applied by the projections, so there is exactly one place a
    rate can be wrong (see ``fps_with_source``).

    ``refs`` comes from ``roi_references``. Bare shapes from ``roi_shapes`` are
    still accepted, and then references are identified by index — old pipelines
    keep working, they just get numbers for names.
    """
    refs = _as_references(refs)
    labels = {r["id"]: r["label"] for r in refs}
    episodes = []

    for tid, grp in tidy.sort_values("frame").groupby("tid"):
        open_ep = None
        for frame, x, y in zip(grp["frame"], grp["x"], grp["y"]):
            frame = int(frame)
            ref = which_reference(x, y, refs)
            if ref is None:
                continue
            broke = (
                open_ep is None
                or open_ep["reference"] != ref["id"]
                or frame - open_ep["end_frame"] > gap_frames
            )
            if broke:
                open_ep = {
                    "track": _as_native(tid),
                    "reference": ref["id"],
                    "reference_label": labels.get(ref["id"], ref["id"]),
                    "start_frame": frame,
                    "end_frame": frame,
                    "frames": 0,
                }
                episodes.append(open_ep)
            open_ep["end_frame"] = frame
            open_ep["frames"] += 1

    episodes.sort(key=lambda e: (e["start_frame"], str(e["reference"])))
    return episodes


# Proximity radius as a fraction of FRAME WIDTH, not pixels.
#
# The worker's InteractionAnalyzer uses a flat 50 px, which means the same
# setting describes a different real distance at every resolution — and after
# the tracks are normalised to 0..1 there are no pixels left to compare against
# anyway. A fraction of the frame is the only threshold that means the same
# thing on a 1080p clip and a 4K one.
#
# 5% of frame width is roughly two body lengths for a bee filling ~2% of the
# frame: close enough to be an encounter, not so close that only overlapping
# boxes qualify.
DEFAULT_PROXIMITY = 0.05

# Frames with more tracks than this are skipped for pairwise proximity: the
# work is quadratic, and a frame with hundreds of detections is a detector
# failure rather than a swarm worth measuring.
_MAX_PAIRWISE_TRACKS = 200


def frame_aspect(summary=None, default=16 / 9):
    """Frame width ÷ height, for un-squashing normalised coordinates.

    ``normalized_tracks`` divides x by width and y by height *separately*, so a
    circle in pixel space becomes an ellipse in normalised space. Scaling y back
    by the aspect ratio restores a true Euclidean distance, expressed in
    fractions of frame width.
    """
    for w_key, h_key in (("frame_width", "frame_height"), ("width", "height"),
                         ("res_width", "res_height"), ("video_width", "video_height")):
        w = (summary or {}).get(w_key)
        h = (summary or {}).get(h_key)
        try:
            if w and h and float(h) > 0:
                return float(w) / float(h)
        except (TypeError, ValueError):
            continue
    return default


def compute_proximity_episodes(tidy, radius=DEFAULT_PROXIMITY, gap_frames=15,
                               aspect=16 / 9):
    """Contiguous spells two tracks spend within ``radius`` of each other.

    The organism-to-organism half of the interaction table. Unlike containment
    in a reference, a distance threshold genuinely is the right model here —
    two bees have no boundary to be inside of — but the threshold has to be
    resolution-independent to mean anything, hence a fraction of frame width
    rather than a pixel count.

    Returns episodes shaped like ``compute_episodes``' output, so both halves of
    the table are built the same way and honour the same gap tolerance.
    """
    import numpy as np

    if tidy is None or len(tidy) == 0 or radius <= 0:
        return []

    open_eps = {}
    episodes = []
    for frame, grp in tidy.sort_values("frame").groupby("frame"):
        frame = int(frame)
        if len(grp) < 2 or len(grp) > _MAX_PAIRWISE_TRACKS:
            continue
        ids = list(grp["tid"])
        # y is scaled back up by the aspect ratio so both axes are in units of
        # frame width and the distance below is a real circle.
        pts = np.column_stack([
            np.asarray(grp["x"], dtype=float),
            np.asarray(grp["y"], dtype=float) / float(aspect or 1.0),
        ])
        deltas = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((deltas ** 2).sum(axis=-1))

        close_i, close_j = np.where(dists <= radius)
        for i, j in zip(close_i, close_j):
            if i >= j:
                continue  # each unordered pair once
            a, b = _as_native(ids[i]), _as_native(ids[j])
            key = (str(a), str(b)) if str(a) <= str(b) else (str(b), str(a))
            open_ep = open_eps.get(key)
            if open_ep is None or frame - open_ep["end_frame"] > gap_frames:
                open_ep = {
                    "track": a if key[0] == str(a) else b,
                    "partner": b if key[0] == str(a) else a,
                    "start_frame": frame,
                    "end_frame": frame,
                    "frames": 0,
                    "min_distance": float(dists[i, j]),
                }
                open_eps[key] = open_ep
                episodes.append(open_ep)
            open_ep["end_frame"] = frame
            open_ep["frames"] += 1
            open_ep["min_distance"] = min(open_ep["min_distance"], float(dists[i, j]))

    episodes.sort(key=lambda e: (e["start_frame"], str(e["track"]), str(e["partner"])))
    return episodes


def compute_visitation(tidy, refs, fps, gap_frames=15):
    """Visit counts per track and per reference, rolled up from the episodes.

    Kept as-is in shape so pipelines and batch pages built on it keep rendering
    identically; the counting now happens over ``compute_episodes`` rather than
    in a second traversal of its own.

    A reference with no visits stays in the breakdown — "nothing visited the
    control" is a result, and dropping the row would leave the reader to notice
    an absence.
    """
    refs = _as_references(refs)
    episodes = compute_episodes(tidy, refs, gap_frames=gap_frames)

    per_ref = {r["id"]: {"id": r["id"], "label": r["label"], "visits": 0,
                         "visitors": set(), "dwell_frames": 0} for r in refs}
    per_track = {}
    for ep in episodes:
        bucket = per_ref.setdefault(ep["reference"], {
            "id": ep["reference"], "label": ep["reference_label"],
            "visits": 0, "visitors": set(), "dwell_frames": 0})
        bucket["visits"] += 1
        bucket["visitors"].add(ep["track"])
        bucket["dwell_frames"] += ep["frames"]

        track = per_track.setdefault(ep["track"], {"visits": 0, "dwell_frames": 0})
        track["visits"] += 1
        track["dwell_frames"] += ep["frames"]

    rows = sorted(
        ({"track": tid, "visits": t["visits"],
          "dwell_sec": round(t["dwell_frames"] / fps, 2) if fps else None}
         for tid, t in per_track.items()),
        key=lambda r: str(r["track"]),
    )

    per_reference = sorted(
        ({"id": b["id"], "label": b["label"], "visits": b["visits"],
          "visitors": len(b["visitors"]),
          "dwell_sec": round(b["dwell_frames"] / fps, 2) if fps else None}
         for b in per_ref.values()),
        key=lambda r: (-r["visits"], str(r["id"])),
    )
    dwell_frames_total = sum(ep["frames"] for ep in episodes)

    return {
        "unique_visitors": len(rows),
        "total_visits": len(episodes),
        "total_dwell_sec": round(dwell_frames_total / fps, 2) if fps else None,
        "rows": rows,
        "per_reference": per_reference,
    }


def _as_references(refs):
    """Accept ``roi_references`` dicts, or bare ``roi_shapes`` tuples.

    Callers that predate references pass geometry only; give those an index for
    an id so the breakdown still works, rather than refusing to compute.
    """
    out = []
    for n, ref in enumerate(refs or [], start=1):
        if isinstance(ref, dict):
            out.append(ref)
            continue
        box, points = (ref if len(ref) == 2 and not isinstance(ref[0], (int, float))
                       else (ref, None))
        out.append({"id": f"region_{n}", "label": f"Region {n}",
                    "box": tuple(box), "points": points})
    return out


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
    # Per reference. Interactions already carry a reference_id, so unlike
    # visitation this needed exposing rather than computing — the breakdown was
    # sitting in the rows and no caller ever grouped it.
    per_ref = {}
    for row in rows:
        ref_id = row.get("reference")
        if ref_id in (None, ""):
            continue          # an insect-to-insect interaction has no reference
        bucket = per_ref.setdefault(str(ref_id), {
            "id": str(ref_id), "label": _reference_label(ref_id),
            "interactions": 0, "partners": set(), "duration_sec": 0.0,
        })
        bucket["interactions"] += 1
        if row.get("a") is not None:
            bucket["partners"].add(row["a"])
        try:
            bucket["duration_sec"] += float(row.get("duration") or 0)
        except (TypeError, ValueError):
            pass

    per_reference = sorted(
        ({"id": b["id"], "label": b["label"], "interactions": b["interactions"],
          "partners": len(b["partners"]), "duration_sec": round(b["duration_sec"], 2)}
         for b in per_ref.values()),
        key=lambda r: (-r["interactions"], r["id"]),
    )

    return {
        "interaction_count": int(len(df)),
        "organism_organism": _count("organism-to-organism"),
        "organism_reference": _count("organism-to-reference"),
        "total_duration_sec": round(sum(durations), 2) if durations else None,
        "rows": rows,
        "per_reference": per_reference,
    }


def _reference_label(ref_id):
    """A readable name for a reference id written by the worker.

    The worker writes literals like ``nest_3``; the layout would call that
    "Nest 3". Keeps the two vocabularies from diverging on screen until the
    editor can carry a real name.
    """
    text = str(ref_id)
    if text.startswith("nest_"):
        return f"Nest {text[5:]}"
    if text.startswith("region_"):
        return f"Region {text[7:]}"
    return text


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
