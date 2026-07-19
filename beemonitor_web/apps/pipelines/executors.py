"""
Step executors for the pipeline engine.

Two kinds of step:

* **local** — runs inline during ``advance_run`` (instant): inputs, ROI resolution,
  filters, post-processing reads of an upstream GPU job's result, and outputs.
* **gpu** — spawns an ``analysis.Job`` on SageMaker and parks the run until the
  poller reports it finished (``engine.on_job_finished`` then advances the run).

Everything is deliberately thin: a step's job is to move a small **artifact
reference** (a dict with an ``artifact`` type + a few keys) into the run context,
never a big blob. See ``memory/23_pipeline_builder_port_design.md``.

Paths that are fully wired to the real backend: ``input.video``, ``roi.*``, and the
detect/track GPU steps (they submit the existing ``detect_and_track`` Job). Steps
marked ``# SCAFFOLD`` below return a structural placeholder pending dedicated backend
work (colony-activity metric, marker decode, richer per-block SageMaker payloads).
"""

import logging
import uuid

from .registry import get_block

logger = logging.getLogger(__name__)


# ── Input resolution helpers ──────────────────────────────────────────────────

def upstream_ids(step, steps, index):
    """Return the list of step ids this step depends on.

    Explicit ``inputs: {port: step_id}`` wins; otherwise the linear default is the
    previous step (unless this block takes no input).
    """
    inputs = step.get("inputs") or {}
    if inputs:
        return [sid for sid in inputs.values() if sid]
    block = get_block(step.get("block_type", "")) or {}
    if block.get("input_type", "none") == "none":
        return []
    if index > 0:
        return [steps[index - 1].get("id")]
    return []


def resolve_inputs(step, steps, index, context):
    """Return {port: upstream_output_dict} for a step, using the run context."""
    inputs = step.get("inputs") or {}
    resolved = {}
    if inputs:
        for port, sid in inputs.items():
            resolved[port] = context.get(sid, {})
    else:
        for uid in upstream_ids(step, steps, index):
            resolved["in"] = context.get(uid, {})
    return resolved


def find_artifact(artifact_type, steps, index, context):
    """Walk upstream (transitively, simple linear/inputs walk) for an artifact type.

    Used by GPU builders to locate e.g. the source ``video`` regardless of how many
    filter/roi steps sit between it and the current step.
    """
    seen = set()
    frontier = list(upstream_ids(steps[index], steps, index))
    by_id = {s.get("id"): (i, s) for i, s in enumerate(steps)}
    while frontier:
        sid = frontier.pop()
        if sid in seen:
            continue
        seen.add(sid)
        out = context.get(sid, {})
        if out.get("artifact") == artifact_type:
            return out
        if sid in by_id:
            i, s = by_id[sid]
            frontier.extend(upstream_ids(s, steps, i))
    return None


# ── Local executors ───────────────────────────────────────────────────────────

def _exec_input_video(step, run, context, inputs, index):
    from apps.videos.models import Video

    video_id = (step.get("config") or {}).get("video_id")
    if not video_id:
        return {"error": "No video selected."}
    try:
        video = Video.objects.get(pk=video_id, user=run.user)
    except (Video.DoesNotExist, ValueError):
        return {"error": "Selected video not found."}
    return {
        "artifact": "video",
        "video_id": video.pk,
        "storage_key": getattr(video, "storage_key", ""),
        "title": getattr(video, "title", "") or f"Video {video.pk}",
    }


def _exec_roi_nest_layout(step, run, context, inputs, index):
    video_out = find_artifact("video", run.steps, index, context)
    if not video_out:
        return {"error": "No upstream video for the ROI layout."}
    hotel_roi, nest_layout = None, None
    try:
        from apps.videos.models import Video

        video = Video.objects.select_related("device").get(pk=video_out["video_id"])
        device = getattr(video, "device", None)
        if device is not None:
            hotel_roi = getattr(device, "roi_override", None)
            nest_layout = getattr(device, "nest_layout", None)
    except Exception as exc:  # defensive — device linkage is optional
        logger.info("roi.nest_layout: could not read device layout: %s", exc)
    return {
        "artifact": "roi",
        "hotel_roi": hotel_roi,
        "nest_layout": nest_layout,
        "source": "device",
    }


def _exec_roi_draw(step, run, context, inputs, index):
    import json

    raw = (step.get("config") or {}).get("regions", "[]")
    try:
        regions = json.loads(raw) if isinstance(raw, str) else (raw or [])
    except json.JSONDecodeError:
        regions = []
    return {"artifact": "roi", "regions": regions, "source": "drawn"}


def _first_upstream_result(inputs):
    """Return the (single) upstream output dict for a linear step."""
    if not inputs:
        return {}
    # prefer an explicitly-named tracks/events port, else the linear 'in'
    for key in ("tracks", "events", "in"):
        if key in inputs:
            return inputs[key]
    return next(iter(inputs.values()), {})


def _exec_analyze_foraging_trips(step, run, context, inputs, index):
    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    return {
        "artifact": "events",
        "event_kind": "foraging_trip",
        "foraging_trip_count": result.get("foraging_trip_count", 0),
        "avg_trip_duration_sec": result.get("avg_trip_duration_sec"),
        "csv": result.get("foraging_trips_csv_path", ""),
        "job_id": (up or {}).get("job_id"),
    }


def _exec_analyze_visitation(step, run, context, inputs, index):
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    roi = find_artifact("roi", run.steps, index, context) or {}
    boxes = ops.roi_boxes(roi)

    df = ops.load_tracking_df(result)
    tidy = ops.normalized_tracks(df, result) if df is not None else None
    if tidy is not None:
        if not boxes:
            return {"artifact": "table", "table_kind": "visitation",
                    "note": "No ROI upstream — add an ROI (draw or nest layout) to count visits."}
        summary = ops.compute_visitation(tidy, boxes, ops.fps_of(result))
        return {"artifact": "table", "table_kind": "visitation", **summary}

    # Fallback: no readable tracking CSV (e.g. dev DB) — surface the job summary.
    return {
        "artifact": "table",
        "table_kind": "visitation",
        "unique_visitors": result.get("unique_tracks", 0),
        "total_visits": result.get("entry_count", 0),
        "rows": [],
        "note": "Derived from job summary (tracking CSV not available).",
    }


def _exec_analyze_colony_activity(step, run, context, inputs, index):
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    metric = (step.get("config") or {}).get("metric", "occupancy")
    roi = find_artifact("roi", run.steps, index, context) or {}
    boxes = ops.roi_boxes(roi)

    df = ops.load_tracking_df(result)
    tidy = ops.normalized_tracks(df, result) if df is not None else None
    if tidy is not None:
        series = ops.compute_colony_activity(tidy, boxes, ops.fps_of(result), metric=metric)
        return {"artifact": "table", "table_kind": "colony_activity", **series}
    return {
        "artifact": "table",
        "table_kind": "colony_activity",
        "metric": metric,
        "rows": [],
        "note": "Derived from job summary (tracking CSV not available).",
    }


def _exec_identify_marker(step, run, context, inputs, index):
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    df = ops.load_tracking_df(result)
    ident = ops.marker_identities(df) if df is not None else None
    if ident is not None:
        return {"artifact": "table", "table_kind": "marker_id", **ident}
    return {
        "artifact": "table", "table_kind": "marker_id",
        "identified_tracks": 0, "unique_markers": 0, "rows": [],
        "note": "No marker IDs in the tracking data — enable bee identification "
                "(the marker step sets it on the tracking job).",
    }


def _exec_filter_passthrough(step, run, context, inputs, index):
    up = _first_upstream_result(inputs)
    out = dict(up) if isinstance(up, dict) else {}
    out["filtered_by"] = step.get("block_type")
    out.setdefault("artifact", "any")
    return out


def _exec_output(step, run, context, inputs, index):
    block_type = step.get("block_type", "")
    up = _first_upstream_result(inputs)
    return {
        "artifact": "none",
        "output_kind": block_type.split(".", 1)[-1],
        "from": up,
    }


LOCAL_EXECUTORS = {
    "input.video": _exec_input_video,
    "input.image_set": lambda s, r, c, i, idx: {"artifact": "frames", "source": (s.get("config") or {}).get("source", "device_crops")},
    "roi.nest_layout": _exec_roi_nest_layout,
    "roi.draw": _exec_roi_draw,
    "analyze.foraging_trips": _exec_analyze_foraging_trips,
    "analyze.visitation": _exec_analyze_visitation,
    "analyze.colony_activity": _exec_analyze_colony_activity,
    "identify.marker": _exec_identify_marker,
    "filter.roi": _exec_filter_passthrough,
    "filter.confidence": _exec_filter_passthrough,
    "filter.taxon": _exec_filter_passthrough,
    "filter.time": _exec_filter_passthrough,
    "output.table": _exec_output,
    "output.chart": _exec_output,
    "output.summary": _exec_output,
    "output.dataset": _exec_output,
}


def run_local_step(step, run, context, index):
    """Execute a local step inline and return its output dict (may hold 'error')."""
    block_type = step.get("block_type", "")
    fn = LOCAL_EXECUTORS.get(block_type)
    inputs = resolve_inputs(step, run.steps, index, context)
    if not fn:
        return {"artifact": "any", "note": f"No local executor for {block_type} (scaffold)."}
    try:
        return fn(step, run, context, inputs, index)
    except Exception as exc:  # keep the run alive; surface the error on the step
        logger.exception("Local step %s failed", block_type)
        return {"error": str(exc)}


# ── GPU steps (spawn an analysis.Job) ─────────────────────────────────────────

_DETECT_TRACK_BLOCKS = {"detect.bee", "detect.nest", "track.bee"}


def _pipeline_wants_markers(steps):
    return any(s.get("block_type") == "identify.marker" for s in steps)


def _marker_method(steps):
    for s in steps:
        if s.get("block_type") == "identify.marker":
            return (s.get("config") or {}).get("marker_type", "auto")
    return "auto"


def _pipeline_event_confidence(steps):
    """Entry/Exit event-classifier cutoff, read from a downstream Foraging Trips
    node (events are computed during tracking, so the Track step needs it).
    Defaults to 0.6 when no foraging node is present."""
    for s in steps:
        if s.get("block_type") == "analyze.foraging_trips":
            try:
                return float((s.get("config") or {}).get("event_confidence", 0.6) or 0.6)
            except (TypeError, ValueError):
                return 0.6
    return 0.6


def build_detect_and_track_config(step, run, context, index):
    """Assemble the ``detect_and_track`` Job config for a detect/track GPU step.

    Pulls the source video + any upstream ROI/nest layout. Tracking always
    uploads per-track crops for later species ID.
    """
    video_out = find_artifact("video", run.steps, index, context)
    if not video_out:
        return None, "No upstream video for this GPU step."

    cfg = (step.get("config") or {})
    detector_kind = "sam3" if str(cfg.get("detector", "yolo")).lower() == "sam3" else "yolo"
    config = {
        "detection_mode": "yolo",
        "confidence_threshold": float(cfg.get("confidence", 0.4) or 0.4),
        # Entry/Exit event-classifier cutoff — set on the downstream Foraging
        # Trips node (events are produced during tracking).
        "ml_threshold": _pipeline_event_confidence(run.steps),
        "run_tracking": step.get("block_type") in ("track.bee", "detect.bee"),
        # Detector: SAM 3 text-prompt tracking, else YOLO.
        "detector_kind": detector_kind,
        "text_prompt": (cfg.get("text_prompt", "") or "").strip(),
        # Annotated video is opt-in (off = much faster, needed for long clips).
        "visualize": str(cfg.get("annotated_video", "")).lower() in ("1", "true", "on", "yes"),
    }
    # Custom models picked on the node (empty = the built-in ones). Resolved by
    # pk against the run owner's models — same contract as the New Analysis
    # form's custom_bee_model / custom_nest_model selects.
    for field, config_key, label in (
        ("bee_model", "custom_bee_model_path", "bee"),
        ("nest_model", "custom_nest_model_path", "nest"),
    ):
        if not cfg.get(field):
            continue
        from apps.training.models import CustomModel
        try:
            model_pk = int(cfg[field])
        except (TypeError, ValueError):
            return None, f"Invalid {label} model selection on this node."
        cm = CustomModel.objects.filter(
            pk=model_pk, user=run.user, is_active=True,
        ).exclude(storage_key="").first()
        if cm:
            config[config_key] = cm.storage_key
        else:
            return None, f"The selected {label} model is unavailable (removed or deactivated)."
    # When a marker step is downstream, ask the tracker to run individual bee
    # identification so bee_id columns land in the tracking CSV (worker must honour).
    if _pipeline_wants_markers(run.steps):
        config["identify_bees"] = True
        config["marker_method"] = _marker_method(run.steps)
    roi_out = find_artifact("roi", run.steps, index, context)
    if roi_out:
        if roi_out.get("hotel_roi"):
            config["hotel_roi"] = roi_out["hotel_roi"]
        if roi_out.get("nest_layout"):
            config["nest_layout"] = roi_out["nest_layout"]
    return {"video_id": video_out["video_id"], "config": config}, None


def submit_gpu_step(run, step, context, index):
    """Handle a GPU step. Returns (state, output) where state is one of
    'submitted' (job in flight), 'done' (placeholder), or 'error'.
    """
    block_type = step.get("block_type", "")

    if block_type not in _DETECT_TRACK_BLOCKS:
        # Any future GPU block without a dedicated job builder resolves to a
        # placeholder so the run doesn't hang. (All current GPU blocks are covered.)
        return "done", {
            "artifact": get_block(block_type).get("output_type", "any"),
            "note": f"{block_type} is not yet wired to a dedicated GPU job (scaffold placeholder).",
        }

    built, err = build_detect_and_track_config(step, run, context, index)
    if err:
        return "error", {"error": err}

    from apps.videos.models import Video
    from apps.analysis.models import Job

    try:
        video = Video.objects.get(pk=built["video_id"])
    except Video.DoesNotExist:
        return "error", {"error": "Source video vanished before job spawn."}

    job_config = dict(built["config"])
    # Tags so the poller hook can find the run + step when the job finishes.
    job_config["pipeline_run_id"] = str(run.pk)
    job_config["pipeline_step_id"] = step.get("id")

    # Create QUEUED (no immediate spawn): the reconciler / an inline drain in
    # run_on_videos promotes it to PROCESSING only while under the global
    # SageMaker cap, so a large "run on all filtered" launch drains in waves
    # instead of flooding the endpoint (which mass-failed jobs on throttling).
    job = Job.objects.create(
        user=run.user,
        video=video,
        status=Job.Status.QUEUED,
        modal_job_id=f"pl_{uuid.uuid4().hex[:14]}",
        config=job_config,
    )
    logger.info("Pipeline run %s step %s queued Job %s", run.pk, step.get("id"), job.pk)
    return "submitted", {
        "artifact": get_block(block_type).get("output_type", "tracks"),
        "job_id": job.pk,
        "pending": True,
    }
