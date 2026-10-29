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
# The detector's class for a row. ``class`` first: the tracking table's
# ``taxon`` becomes the voted species once Identify species has run, so it is
# the detector's class only in tables written before the worker kept ``class``
# apart (memory/60).
_CLASS_COLS = ["class", "label", "class_name", "taxon"]
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
    """Read a CSV into a DataFrame, or None.

    Accepts the three spellings a path arrives in: an ``s3://bucket/key`` URL, a
    local file (tests, dev), and — the one that matters in production — a bare
    key into the *processed* bucket, which is what every JobResult stores.

    That last case used to fall through to ``pd.read_csv("1/abc/tracking.csv")``,
    which is a relative filename that does not exist, so every local analyzer
    quietly took its "tracking CSV not available" branch and reported the job
    summary instead of analysing anything. It looked like a data condition; it
    was a plumbing bug.
    """
    if not path:
        return None
    try:
        import pandas as pd
    except ImportError:
        # A deployment defect, not a property of the clip: without pandas none
        # of the analyzers can run at all. Loud, so it cannot hide as "no data".
        logger.error("pandas is not installed in this image — no analyzer can "
                     "post-process a tracking table")
        return None

    from io import BytesIO

    try:
        if path.startswith("s3://"):
            from urllib.parse import urlparse

            import boto3
            from django.conf import settings

            parsed = urlparse(path)
            s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
            body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
            return pd.read_csv(BytesIO(body))

        import os

        if os.path.exists(path):
            return pd.read_csv(path)

        # A key in the processed bucket — the same place aggregate reads from.
        from config.storage import get_s3_client

        buf = BytesIO()
        get_s3_client().download_to_stream("processed", path, buf)
        buf.seek(0)
        return pd.read_csv(buf)
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

    # Whether the tracker matched a detection on this row (1) or only predicted
    # where a lost insect might be (0) — memory/60. Tracking CSVs written before
    # the worker recorded it have no such column, and it is NOT guessed: those
    # rows all count, and ``has_detected`` lets callers say so.
    det_col = _pick(df, ["detected"])
    out["_detected"] = (out[det_col].fillna(0).astype(int) if det_col is not None else 1)

    # Body length in pixels: the long side of the box. Distances, proximity and
    # padding are all expressed in body lengths, so a threshold means the same
    # thing for a bumblebee close to the lens and a sweat bee far from it.
    bx1, by1 = _pick(df, _BBOX["x1"]), _pick(df, _BBOX["y1"])
    bx2, by2 = _pick(df, _BBOX["x2"]), _pick(df, _BBOX["y2"])
    if None not in (bx1, by1, bx2, by2):
        out["_bl"] = ((out[bx2] - out[bx1]).abs().combine((out[by2] - out[by1]).abs(), max))
    else:
        out["_bl"] = float("nan")

    tidy = out[[id_col, frame_col, cx_col, cy_col, "_detected", "_bl"]].copy()
    tidy.columns = ["tid", "frame", "x", "y", "detected", "bl"]
    tidy = tidy.dropna(subset=["tid", "x", "y"])

    # Normalise pixel coordinates against the FRAME, never against the tracks.
    #
    # This used to fall back to the observed max when no frame size was known,
    # which silently rescales every clip by its own activity: a bee at (700,620)
    # in a 1920x1080 frame is at (0.365, 0.574) — the top-left flower — but
    # divided by the extent of its own track it lands at (0.974, 0.970), the
    # bottom-right one. References are in true frame fractions, so every episode
    # was attributed to whichever box the distortion happened to land in. That
    # is why a clip whose activity was plainly in flower 1 reported nest_4.
    #
    # Returning None instead is the honest answer: without the frame size these
    # coordinates cannot be placed against anything, and a wrong placement is
    # far worse than a missing one.
    def _dim(keys):
        for k in keys:
            try:
                value = float((summary or {}).get(k) or 0)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    for axis, keys in (("x", ["frame_width", "width", "res_width", "video_width"]),
                       ("y", ["frame_height", "height", "res_height", "video_height"])):
        extent = float(tidy[axis].abs().max() or 0)
        if extent <= 1.5:
            continue                       # already fractional
        dim = _dim(keys)
        if not dim:
            logger.warning(
                "tracking table is in pixels but the frame size is unknown — "
                "refusing to place tracks rather than scaling them by their own "
                "extent")
            return None
        tidy[axis] = tidy[axis] / dim
        if axis == "x":
            # Body length in units of frame WIDTH, the unit every distance is in.
            tidy["bl"] = tidy["bl"] / dim

    # One position per track per frame — which is what "tidy" claims and what
    # every consumer assumes. The worker's tracking CSV can carry more than one
    # row for the same (frame, track): two overlapping detections handed the
    # same id. Left in, those rows are two points a few ten-thousandths apart,
    # and the proximity pass paired them with each other and reported track 1
    # interacting with track 1. They also inflate containment: compute_episodes
    # counts frames, so a duplicated frame counts twice and the episode reads
    # twice as long as it was.
    tidy = tidy.drop_duplicates(subset=["frame", "tid"], keep="first")
    tidy.attrs["has_detected"] = det_col is not None
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

    The hotel ROI is NOT a reference (memory/60): it is the area tracking runs
    over. Counting it as one turned "no tubes drawn" into "every insect visited
    the hotel"; a pipeline with no references now has no references.

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

    # Tubes and drawn regions are all references: one numbering, reference_<n>.
    # A tube keeps its layout id (stable across the batch's clips); a region
    # keeps its own id unless a tube already holds that number.
    taken = set()

    def _numbered(wanted, fallback):
        n = wanted if wanted not in (None, "") else fallback
        while str(n) in taken:
            n = (int(n) + 1) if str(n).isdigit() else f"{n}b"
        taken.add(str(n))
        return f"reference_{n}", f"Reference {n}"

    for n, tube in enumerate(roi_output.get("nest_layout") or [], start=1):
        if isinstance(tube, dict):
            ref_id, label = _numbered(tube.get("id"), n)
            # A name given in the editor ("Flag C", "Hole 3") is what results show.
            _add(ref_id, tube.get("name") or label, tube.get("box"), tube.get("points"))
        else:
            _add(*_numbered(None, n), tube)

    for n, region in enumerate(roi_output.get("regions") or [], start=1):
        if isinstance(region, dict):
            ref_id, label = _numbered(region.get("id"), n)
            # An editor that learns to name regions should put it in `name`.
            _add(ref_id, region.get("name") or label, region.get("box"), region.get("points"))
        else:
            _add(*_numbered(None, n), region)

    return refs


def _shape_points(ref):
    """A reference's outline as points: its polygon, or its box's corners."""
    if ref.get("points"):
        return ref["points"]
    x1, y1, x2, y2 = ref["box"]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def _inside(x, y, ref):
    x1, y1, x2, y2 = ref["box"]
    if not (x1 <= x <= x2 and y1 <= y <= y2):
        return False
    return ref.get("points") is None or _in_polygon(x, y, ref["points"])


def _distance_outside(x, y, ref, aspect):
    """How far (x, y) is outside the reference's outline, in frame WIDTHS; 0
    inside. y is divided by the aspect ratio so the distance is a true one."""
    if _inside(x, y, ref):
        return 0.0
    pts = [(px, py / aspect) for px, py in _shape_points(ref)]
    qx, qy = x, y / aspect
    best = float("inf")
    for i in range(len(pts)):
        (ax, ay), (bx, by) = pts[i - 1], pts[i]
        dx, dy = bx - ax, by - ay
        seg = dx * dx + dy * dy
        t = 0.0 if seg == 0 else max(0.0, min(1.0, ((qx - ax) * dx + (qy - ay) * dy) / seg))
        cx, cy = ax + t * dx, ay + t * dy
        best = min(best, ((qx - cx) ** 2 + (qy - cy) ** 2) ** 0.5)
    return best


def _centre(ref):
    x1, y1, x2, y2 = ref["box"]
    return (x1 + x2) / 2, (y1 + y2) / 2


def reference_at(x, y, refs, pad=0.0, aspect=16 / 9):
    """The reference (x, y) is in, or None.

    ``pad`` grows every reference by that distance, in frame widths (the
    Events/Interactions node sets it in body lengths). References should not
    overlap; when a point is in more than one anyway, it belongs to the one
    whose centre is nearest — the single rule ``overlapping_references`` warns
    about, so nothing is counted twice and nothing is decided silently.
    """
    if pad and pad > 0:
        hits = [r for r in refs if _distance_outside(x, y, r, aspect) <= pad]
    else:
        hits = [r for r in refs if _inside(x, y, r)]
    if len(hits) <= 1:
        return hits[0] if hits else None

    def _dist(r):
        cx, cy = _centre(r)
        return ((x - cx) ** 2 + ((y - cy) / aspect) ** 2) ** 0.5
    return min(hits, key=_dist)


def which_reference(x, y, refs):
    """The reference containing (x, y), no padding, or None."""
    return reference_at(x, y, refs)


def _segments_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(p3, p4, p1), orient(p3, p4, p2)
    d3, d4 = orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 > 0) != (d2 > 0) and (d3 > 0) != (d4 > 0)


def overlapping_references(refs):
    """Pairs of references whose shapes overlap, as ``[(id_a, id_b), ...]``.

    References are meant not to overlap (memory/60); a layout where they do
    still runs, and this is what its warning names.
    """
    refs = list(refs or [])
    pairs = []
    for i, a in enumerate(refs):
        for b in refs[i + 1:]:
            ax1, ay1, ax2, ay2 = a["box"]
            bx1, by1, bx2, by2 = b["box"]
            if ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1:
                continue                      # boxes apart (touching is not overlap)
            if not a.get("points") and not b.get("points"):
                pairs.append((a["id"], b["id"]))   # two boxes sharing some area
                continue
            pa, pb = _shape_points(a), _shape_points(b)
            if (any(_inside(x, y, b) for x, y in pa) or any(_inside(x, y, a) for x, y in pb)
                    or any(_segments_cross(pa[i - 1], pa[i], pb[j - 1], pb[j])
                           for i in range(len(pa)) for j in range(len(pb)))):
                pairs.append((a["id"], b["id"]))
    return pairs


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


def has_detected(tidy):
    """Whether the tracking table said which rows were detections (memory/60).

    Older tracking CSVs did not. Their rows are used as they are — predicted
    positions included — and every output built from them says so.
    """
    return bool(getattr(tidy, "attrs", {}).get("has_detected", False))


def _detected_rows(tidy):
    """Rows where the tracker matched a detection. A lost insect's predicted
    positions decide nothing: not where a track starts or ends, not how long
    it was somewhere. A table with no body length (no boxes) gets NaN, which
    means no padding and no proximity — never a made-up size."""
    if "bl" not in tidy.columns:
        tidy = tidy.assign(bl=float("nan"))
    if "detected" not in tidy.columns:
        return tidy
    return tidy[tidy["detected"] == 1]


def _pad(bl, padding_bl):
    """Padding in frame widths for one row: body lengths x that row's body
    length. No box, no body length, no padding."""
    try:
        value = float(bl) * float(padding_bl or 0)
    except (TypeError, ValueError):
        return 0.0
    return value if value == value and value > 0 else 0.0     # NaN -> 0


def compute_episodes(tidy, refs, gap_frames=15, padding_bl=0.0, aspect=16 / 9):
    """Contiguous spells each track spends in each reference.

    An episode is a run of DETECTED frames one track spends in one reference
    (grown by ``padding_bl`` body lengths). A gap of more than ``gap_frames``,
    or moving to another reference, starts a new one. ``frames`` counts the
    detected frames held, which is what a duration is made of.
    """
    refs = _as_references(refs)
    labels = {r["id"]: r["label"] for r in refs}
    episodes = []
    if tidy is None or len(tidy) == 0 or not refs:
        return episodes

    rows = _detected_rows(tidy).sort_values("frame")
    for tid, grp in rows.groupby("tid"):
        open_ep = None
        for frame, x, y, bl in zip(grp["frame"], grp["x"], grp["y"], grp["bl"]):
            frame = int(frame)
            ref = reference_at(x, y, refs, _pad(bl, padding_bl), aspect)
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


# The event rule (memory/60). An insect that comes out of a reference is first
# seen there; one that goes in is last seen there. How long it was there does
# not matter — exits are often a fraction of a second.
EXIT, ENTER, STAY = "exit", "enter", "stay"


def compute_events(tidy, refs, fps, padding_bl=0.0, aspect=16 / 9,
                   first_frame=0, last_frame=None):
    """Exits, entries and stays, one pass over each track's detected rows.

    For each track, its start (first detected row) and end (last detected row):

    * starts in R and is somewhere else afterwards  -> **exit** R at the start
    * ends in R and was somewhere else before       -> **enter** R at the end
    * starts and ends in R and never left it        -> **stay** R, start to end
    * neither end in a reference                    -> nothing (passing over a
      reference is an interaction)

    "Somewhere else" is any detected row not in R: another reference or none.
    ``first_frame`` / ``last_frame`` are the first and last frame analysed (the
    clip's, or a chosen window's). An exit on the first, or an entry on the
    last, is ``at_edge`` — counted, and flagged so it can be filtered out.
    ``last_frame`` None = the clip's length is unknown, and then an entry's
    flag is None, not a guess.

    Returns ``[{track, action, reference, frame, end_frame, at_edge}]`` (end_frame
    on stays) in frame order.
    """
    refs = _as_references(refs)
    if tidy is None or len(tidy) == 0 or not refs:
        return []

    # An exit can only be cut by the start of what was analysed, an entry only
    # by its end — so an exit's flag is always known, an entry's only when the
    # clip's length is.
    def _at_start(frame):
        return frame <= first_frame

    def _at_end(frame):
        return None if last_frame is None else frame >= last_frame

    events = []
    rows = _detected_rows(tidy).sort_values("frame")
    for tid, grp in rows.groupby("tid"):
        tid = _as_native(tid)
        where = [
            (ref["id"] if ref else None)
            for ref in (reference_at(x, y, refs, _pad(bl, padding_bl), aspect)
                        for x, y, bl in zip(grp["x"], grp["y"], grp["bl"]))
        ]
        frames = [int(f) for f in grp["frame"]]
        start_ref, end_ref = where[0], where[-1]
        if start_ref is not None and all(w == start_ref for w in where):
            end_edge = _at_end(frames[-1])
            events.append({"track": tid, "action": STAY, "reference": start_ref,
                           "frame": frames[0], "end_frame": frames[-1],
                           "at_edge": True if _at_start(frames[0]) or end_edge else end_edge})
            continue
        if start_ref is not None:
            events.append({"track": tid, "action": EXIT, "reference": start_ref,
                           "frame": frames[0], "at_edge": _at_start(frames[0])})
        if end_ref is not None:
            events.append({"track": tid, "action": ENTER, "reference": end_ref,
                           "frame": frames[-1], "at_edge": _at_end(frames[-1])})

    order = {EXIT: 0, STAY: 1, ENTER: 2}
    events.sort(key=lambda e: (e["frame"], order[e["action"]], str(e["track"])))
    return events


# Insect-to-insect radius, in BODY LENGTHS (memory/60): the long side of the
# insects' own boxes, averaged over the pair so A-B reads the same as B-A. A
# fraction of the frame meant a different real distance for an insect near the
# lens and one far from it.
DEFAULT_PROXIMITY_BL = 2.0


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


def compute_proximity_episodes(tidy, radius_bl=DEFAULT_PROXIMITY_BL, gap_frames=15,
                               aspect=16 / 9):
    """Contiguous spells two tracks spend within ``radius_bl`` body lengths.

    Detected rows only, every frame (no cap on how many insects a frame holds).
    Pairs where either insect has no box have no body length and are not
    measured. ``min_distance_bl`` is the closest the pair came, in body lengths.
    """
    import numpy as np

    if tidy is None or len(tidy) == 0 or not radius_bl or radius_bl <= 0:
        return []

    rows = _detected_rows(tidy).dropna(subset=["bl"])
    open_eps = {}
    episodes = []
    for frame, grp in rows.sort_values("frame").groupby("frame"):
        frame = int(frame)
        if len(grp) < 2:
            continue
        ids = list(grp["tid"])
        pts = np.column_stack([
            np.asarray(grp["x"], dtype=float),
            np.asarray(grp["y"], dtype=float) / float(aspect or 1.0),
        ])
        bl = np.asarray(grp["bl"], dtype=float)
        deltas = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt((deltas ** 2).sum(axis=-1))
        pair_bl = (bl[:, None] + bl[None, :]) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            in_bl = np.where(pair_bl > 0, dists / pair_bl, np.inf)

        close_i, close_j = np.where(in_bl <= radius_bl)
        for i, j in zip(close_i, close_j):
            if i >= j:
                continue  # each unordered pair once
            a, b = _as_native(ids[i]), _as_native(ids[j])
            if str(a) == str(b):
                continue
            key = (str(a), str(b)) if str(a) <= str(b) else (str(b), str(a))
            open_ep = open_eps.get(key)
            if open_ep is None or frame - open_ep["end_frame"] > gap_frames:
                open_ep = {
                    "track": a if key[0] == str(a) else b,
                    "partner": b if key[0] == str(a) else a,
                    "start_frame": frame,
                    "end_frame": frame,
                    "frames": 0,
                    "min_distance_bl": float(in_bl[i, j]),
                }
                open_eps[key] = open_ep
                episodes.append(open_ep)
            open_ep["end_frame"] = frame
            open_ep["frames"] += 1
            open_ep["min_distance_bl"] = min(open_ep["min_distance_bl"], float(in_bl[i, j]))

    episodes.sort(key=lambda e: (e["start_frame"], str(e["track"]), str(e["partner"])))
    return episodes


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
        out.append({"id": f"reference_{n}", "label": f"Reference {n}",
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


def resolve_taxa(df, wanted, exclude=()):
    """Which taxa in ``df`` belong to this branch. Returns ``(taxa, how)``.

    A pipeline must not break because the detector calls the animal something
    other than the node was configured with. The configured class is a
    *preference*, resolved against the labels the table actually carries:

    * ``"exact"``  — the configured class is in the table. Nothing to decide.
    * ``"role"``   — it is not, but a sibling branch claims another class, so
      this branch is what remains. A Detect(flower) node on the reference port
      tells us the rest of the table is the subject, whatever it is called.
    * ``"only"``   — it is not, and the table holds a single class. There is
      nothing to separate, so that class is this branch.
    * ``"absent"`` — several classes exist, none is the wanted one, and no
      sibling claim narrows it down. Genuinely nothing here, and saying so
      beats handing back bees when the question was wasps.

    ``taxa`` of None means "take everything"; an empty set means "take nothing".
    """
    if df is None or len(df) == 0:
        return None, "exact"
    col = _pick(df, _CLASS_COLS)
    if col is None:
        # Older results predate the column. Returning nothing would read as
        # "no detections" instead of "can't tell".
        return None, "exact"

    wanted = {p.strip().lower() for p in str(wanted or "").split(",") if p.strip()}
    if not wanted:
        return None, "exact"

    have = set(df[col].astype(str).str.strip().str.lower().unique())
    hit = wanted & have
    if hit:
        return hit, "exact"

    claimed = {p.strip().lower()
               for label in (exclude or []) if label
               for p in str(label).split(",") if p.strip()}
    remainder = have - claimed
    if (claimed & have) and remainder:
        logger.info("label filter: %s not in the table; taking %s by role "
                    "(%s claimed by another branch)", sorted(wanted),
                    sorted(remainder), sorted(claimed & have))
        return remainder, "role"

    if len(have) == 1:
        logger.info("label filter: table is entirely %s but this branch asks "
                    "for %s — taking it", sorted(have), sorted(wanted))
        return have, "only"

    return set(), "absent"


def filter_by_label(df, label, exclude=()):
    """Keep only the rows belonging to this branch.

    One GPU pass serves several Detect nodes, so "which rows are mine" has to be
    decided somehow — but it must not be decided by a literal string the user
    typed, or renaming a class silently empties every analyzer downstream of it.
    ``resolve_taxa`` explains how the answer is reached.
    """
    taxa, _how = resolve_taxa(df, label, exclude)
    if taxa is None:
        return df
    col = _pick(df, _CLASS_COLS)
    return df if col is None else df[
        df[col].astype(str).str.strip().str.lower().isin(taxa)]


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


def _reference_label(ref_id):
    """A readable name for a reference id: ``reference_3`` → "Reference 3".

    The worker writes literals like ``nest_3`` and a bare box id like ``3``;
    every reference is called a reference, whatever it is (a tube, a flower).
    """
    from .primitives import reference_id

    text = str(reference_id(ref_id))
    if text.startswith("reference_"):
        return f"Reference {text[10:]}"
    if text.isdigit():
        return f"Reference {text}"
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
        row = {
            "track": _as_native(tid), "taxon": taxon,
            "confidence": confidence, "votes": votes,
            "frames": int(len(grp)),
        }
        # Crop-vote runs (memory/42) also say how many crops were read and the
        # winner's share of the readings.
        for col, key in (("taxon_crops", "crops"), ("taxon_vote_share", "vote_share")):
            c = _pick(df, [col])
            if c is not None and grp[c].notna().any():
                row[key] = _as_native(grp[c].dropna().iloc[0])
        rows.append(row)
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
