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


def _walk_upstream(steps, index):
    """Yield ``(step_id, step_index, step)`` for every transitive upstream step.

    Breadth-ish walk over the ``inputs`` map (with the legacy linear fallback).
    Both artifact lookups below share it, so they traverse identically — which is
    what lets ``find_reference`` see past the local MOT step to the Detector.
    """
    seen = set()
    frontier = list(upstream_ids(steps[index], steps, index))
    by_id = {s.get("id"): (i, s) for i, s in enumerate(steps)}
    while frontier:
        sid = frontier.pop()
        if sid in seen:
            continue
        seen.add(sid)
        i, s = by_id.get(sid, (None, None))
        yield sid, i, s
        if s is not None:
            frontier.extend(upstream_ids(s, steps, i))


def find_artifact(artifact_type, steps, index, context):
    """Walk upstream (transitively, simple linear/inputs walk) for an artifact type.

    Used by GPU builders to locate e.g. the source ``video`` regardless of how many
    filter/roi steps sit between it and the current step.
    """
    for sid, _i, _s in _walk_upstream(steps, index):
        out = context.get(sid, {})
        if out.get("artifact") == artifact_type:
            return out
    return None


def detector_label(step):
    """The class a Detect node is aimed at (its detections' ``taxon``)."""
    cfg = step.get("config") or {}
    # ``text_prompt`` is the legacy detect.bee/track.bee key for the same idea.
    return (cfg.get("label") or cfg.get("text_prompt") or "").strip()


def resolve_reference(step, run, context, index):
    """Resolve a reference node's config into ROI-shaped data.

    Returns the ``{hotel_roi?, hotel_polygon?, nest_layout?, regions?}`` shape
    ``ops.roi_shapes`` already consumes. Deliberately a function of the step's **config** (plus the
    upstream video's device), never of its cached output: ``engine`` caches a GPU
    step's output dict in ``StepResult`` and replays it verbatim on a hit, and
    drawn regions are not part of the hashed job config — so a reference stashed
    in the output would go stale the moment the user edited it.

    Handles ``reference.layout`` (saved geometry) and the legacy detect.objects
    ``reference_source`` config, which predates the split into one node per class.
    """
    import json

    cfg = step.get("config") or {}
    # reference.layout uses "source"; legacy detect.objects used "reference_source".
    source = cfg.get("source") or cfg.get("reference_source") or "device_layout"

    if source == "none":
        return {}

    if source == "drawn":
        raw = cfg.get("regions", "[]")
        try:
            regions = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except json.JSONDecodeError:
            regions = []
        return {"artifact": "roi", "regions": regions, "source": "drawn"}

    if source == "detect":
        # Legacy only. The nest/hotel model ran inside the GPU job; its boxes come
        # back on the result. New graphs express this as a Detect node wired to the
        # analyzer's reference port instead.
        out = context.get(step.get("id")) or {}
        summary = ((out.get("result") or {}).get("summary_stats")) or {}
        boxes = summary.get("nest_bboxes") or []
        return {
            "artifact": "roi",
            "nest_layout": [{"box": b} for b in boxes if b],
            "source": "detected",
        }

    # "device_layout" — the hotel ROI + nest tubes drawn in the ROI editor.
    # hotel_polygon (and a nest's "points") is the traced outline of a shape the
    # user drew as a polygon; the box is its bounding box, so a consumer that
    # only reads boxes still works, and one that reads points excludes the
    # background the box swept in.
    hotel_roi, hotel_polygon, nest_layout = None, None, None
    video_out = find_artifact("video", run.steps, index, context)
    if video_out:
        try:
            from apps.videos.models import Video

            video = Video.objects.select_related("device").get(pk=video_out["video_id"])
            device = getattr(video, "device", None)
            if device is not None:
                # The layout in use when this clip was recorded, not today's.
                from apps.devices.layouts import layout_for_video
                layout = layout_for_video(video)
                hotel_roi = layout["roi_override"]
                hotel_polygon = layout["roi_polygon"]
                nest_layout = layout["nest_layout"] or None
        except Exception as exc:  # defensive — device linkage is optional
            logger.info("detect.objects: could not read device layout: %s", exc)
    return {
        "artifact": "roi",
        "hotel_roi": hotel_roi,
        "hotel_polygon": hotel_polygon,
        "nest_layout": nest_layout,
        "source": "device",
    }


_REFERENCE_BLOCKS = ("reference.layout", "roi.nest_layout", "roi.draw")


def find_reference(steps, index, context, run):
    """Find the reference geometry for a step, from any era of the builder.

    Prefers an explicit ``rois`` edge — with one node per class, a Detect node
    aimed at the reference class is wired straight into the analyzer's reference
    port, and that intent should win over anything found by walking. Falls back to
    an upstream reference node (or a legacy Detector carrying reference config).
    Returns ``{}`` when there is no reference.
    """
    by_id = {s.get("id"): (i, s) for i, s in enumerate(steps)}
    explicit = ((steps[index].get("inputs") or {}).get("rois"))
    if explicit and explicit in by_id:
        i, step = by_id[explicit]
        ref = _reference_from(step, run, context, i)
        if ref:
            return ref

    for sid, i, step in _walk_upstream(steps, index):
        out = context.get(sid, {})
        if out.get("artifact") == "roi":  # a reference node that already ran
            return out
        if step is not None and step.get("block_type") in _REFERENCE_BLOCKS:
            return resolve_reference(step, run, context, i)
        # Legacy: the Detector used to carry reference_source in its own config.
        if (step is not None and step.get("block_type") == "detect.objects"
                and (step.get("config") or {}).get("reference_source")):
            return resolve_reference(step, run, context, i)
    return {}


def _reference_from(step, run, context, index):
    """Reference geometry from one node, whichever kind it is."""
    block_type = step.get("block_type")
    if block_type in _REFERENCE_BLOCKS:
        return resolve_reference(step, run, context, index)
    if block_type == "detect.objects":
        # A Detect node aimed at the reference class: its boxes ARE the reference.
        return detected_reference(step, run, context, index)
    out = context.get(step.get("id")) or {}
    return out if out.get("artifact") == "roi" else {}


def detected_reference(step, run, context, index):
    """Turn a Detect node's detections into reference boxes.

    Reads the shared GPU result and keeps only rows whose taxon matches this
    node's label — the same label-filtering every Detect consumer does. Falls back
    to the job's ``nest_bboxes`` summary, which the worker fills in when it
    located the hotel/nest layout itself.
    """
    from . import ops

    out = context.get(step.get("id")) or {}
    result = out.get("result") or {}
    label = detector_label(step)

    df = ops.load_detections_df(result)
    if df is None:
        df = ops.filter_by_label(ops.load_tracking_df(result), _upstream_label(inputs))
    boxes = ops.boxes_for_label(df, label) if df is not None else []
    if boxes:
        return {"artifact": "roi", "source": "detected", "label": label,
                "regions": [{"box": b} for b in boxes]}

    summary = result.get("summary_stats") or {}
    nest_boxes = summary.get("nest_bboxes") or {}
    if nest_boxes:
        values = nest_boxes.values() if isinstance(nest_boxes, dict) else nest_boxes
        return {"artifact": "roi", "source": "detected", "label": label,
                "nest_layout": [{"box": b} for b in values if b]}
    return {}


# ── Local executors ───────────────────────────────────────────────────────────

def _exec_input_video(step, run, context, inputs, index):
    from apps.videos.models import Video

    video_id = (step.get("config") or {}).get("video_id")
    if not video_id:
        return {"error": "No video selected."}
    # `manageable` — owned, or on a device shared with the runner as manager.
    # MUST match the launch-side check (pipelines.views.run_on_videos and
    # devices.scheduling._videos_for both select with `manageable`); an
    # owner-only lookup here failed every run on a shared device's videos
    # AFTER it had already passed validation.
    try:
        video = Video.manageable(run.user).get(pk=video_id)
    except (Video.DoesNotExist, ValueError, TypeError):
        return {"error": f"Video {video_id} not found, or not yours to analyze."}
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
    hotel_roi, hotel_polygon, nest_layout = None, None, None
    try:
        from apps.videos.models import Video

        video = Video.objects.select_related("device").get(pk=video_out["video_id"])
        device = getattr(video, "device", None)
        if device is not None:
            # The layout in use when this clip was recorded, not today's.
            from apps.devices.layouts import layout_for_video
            layout = layout_for_video(video)
            hotel_roi = layout["roi_override"]
            hotel_polygon = layout["roi_polygon"]
            nest_layout = layout["nest_layout"] or None
    except Exception as exc:  # defensive — device linkage is optional
        logger.info("roi.nest_layout: could not read device layout: %s", exc)
    return {
        "artifact": "roi",
        "hotel_roi": hotel_roi,
        "hotel_polygon": hotel_polygon,
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


def _upstream_label(inputs):
    """The class carried by whatever fed this analyzer.

    Set by the MOT step (inherited from its Detect node). Analyzers use it to take
    only their own rows out of the shared GPU result, so a Detect(bee) → MOT →
    Visitation branch counts bees even when a Detect(nest tube) node put nest rows
    in the same table.
    """
    for key in ("tracks", "detections", "in"):
        value = inputs.get(key)
        if isinstance(value, dict) and value.get("label"):
            return value["label"]
    return ""


def _first_upstream_result(inputs):
    """Return the (single) upstream output dict for a linear step."""
    if not inputs:
        return {}
    # prefer an explicitly-named tracks/detections/events port, else the linear 'in'
    for key in ("tracks", "detections", "events", "in"):
        if key in inputs:
            return inputs[key]
    return next(iter(inputs.values()), {})


def _exec_mot(step, run, context, inputs, index):
    """Module 2 — relabel the Detector's fused GPU result as a ``tracks`` artifact.

    Tracking is executed *inside* the Detector's ``detect_and_track`` job (the
    worker runs detection + BeeTrack + events in one pass), so this step costs
    nothing: it names the algorithm and gives the graph the explicit MOT stage the
    design calls for. The upstream ``result`` is copied through verbatim so every
    analyzer's ``ops.load_tracking_df(result)`` keeps working unchanged.
    """
    up = inputs.get("detections") or _first_upstream_result(inputs)
    up = up or {}
    if up.get("error"):
        return {"error": "The upstream detector step failed."}
    result = up.get("result") or {}
    # Carry the upstream Detect node's class forward. Several Detect nodes share
    # one GPU result, so "which rows are mine" is decided by label, and every
    # downstream analyzer needs to inherit that answer rather than re-deriving it.
    detector = _upstream_detector(run.steps, index) if run is not None else None
    label = detector_label(detector) if detector else ""

    if not result.get("tracking_csv_path"):
        # Only a detector we can positively see was asked for objects earns the
        # empty-clip reading. A reference-only scope, a legacy nest-only block,
        # or a graph we cannot inspect all keep the original error: guessing
        # "empty clip" for a misconfigured pipeline would hide the misconfiguration
        # behind empty tables, which is the harder bug to find.
        if detector is None or not _run_tracking_for(detector):
            return {
                "error": "No detections to track. The upstream Detector produced no "
                         "tracking data — set its Run scope to 'Objects + reference' "
                         "if it is currently 'Reference only'.",
            }
        # The detector ran the full pass and found nothing, and the worker writes
        # no tracking CSV when it has nothing to write (files_uploaded: {}). That
        # is a measurement, not a failure: an empty clip is the commonest thing in
        # field footage, and failing the run for it turned "19 clips had no bees"
        # into "19 failures to triage" — while each clip's own page said its job
        # completed. The run now completes with zero rows, which is the answer.
        return {
            "artifact": "tracks",
            "result": result,
            "job_id": up.get("job_id"),
            "tracker": (step.get("config") or {}).get("tracker", "beetrack"),
            "label": label,
            "unique_tracks": 0,
            "empty": True,
            "note": "The detector found nothing in this clip — no tracks to "
                    "analyse. Downstream tables are empty because the clip is, "
                    "not because the run failed.",
        }
    return {
        "artifact": "tracks",
        "result": result,
        "job_id": up.get("job_id"),
        "tracker": (step.get("config") or {}).get("tracker", "beetrack"),
        "label": label,
        "unique_tracks": result.get("unique_tracks", 0),
    }


def _analysis_inputs(step, run, context, inputs, index):
    """The pieces every primitive analyzer needs: tidy tracks, refs, fps.

    Returns ``(tidy, refs, result, fps, fps_source)``; ``tidy`` is None when
    there is no readable tracking CSV, which callers report rather than treat
    as "nothing happened".
    """
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})

    video = _run_video(run)
    scale = _frame_scale(run, result, video)

    roi = find_reference(run.steps, index, context, run)
    refs = ops.roi_references(roi)
    ref_source = "graph"
    if not refs:
        # Nothing drawn or wired — but the detector may have FOUND the
        # references (a pipeline whose reference class is detected rather than
        # drawn: flowers on a board). Those boxes sit in the job's summary and
        # nothing local ever read them, so such a pipeline reported "0
        # references, 0 visits" while its job page said it had found four
        # nests.
        # The worker reports those boxes in pixels too, placed by the same
        # frame size measured above.
        refs = ops.detected_references(result, video)
        ref_source = "detected" if refs else "none"
    # The hotel ROI contains every tube, so counting it as a reference would
    # double every episode. It is kept only when it is the ONLY thing defined.
    tubes = [r for r in refs if r["id"] != "hotel"]
    refs = tubes or refs

    # What the reference branch claims, so the subject branch can be "the rest"
    # when the detector's own class names differ from the configured ones. That
    # is what makes the pipeline taxon-agnostic: renaming a class moves rows
    # between branches by role, it does not empty one.
    raw = ops.load_tracking_df(result)
    taxa, taxa_how = ops.resolve_taxa(raw, _upstream_label(inputs),
                                      exclude=[(roi or {}).get("label")])
    df = ops.filter_by_label(raw, _upstream_label(inputs),
                             exclude=[(roi or {}).get("label")])
    tidy = ops.normalized_tracks(df, scale) if df is not None else None
    fps, fps_source = ops.fps_with_source(result, video)
    return tidy, refs, result, fps, fps_source, ref_source, {
        "taxa": sorted(taxa) if taxa else ([] if taxa == set() else None),
        "how": taxa_how,
    }


def _frame_scale(run, result, video=None):
    """Summary dict for ``normalized_tracks``, carrying the clip's frame size.

    The frame size is what turns a pixel coordinate into a position, and the
    worker never reports it — so it comes from the video row, measured at
    ingest and probed now for clips that predate that. Without it tracks cannot
    be placed against a reference at all, and scaling them by their own extent
    (what this used to fall back to) puts every clip's activity wherever that
    clip happened to be busiest.
    """
    if video is None:
        video = _run_video(run)
    if video is not None:
        from apps.videos.thumbnails import ensure_dimensions

        ensure_dimensions(video)
    scale = dict((result or {}).get("summary_stats") or {})
    if video is not None and video.width and video.height:
        scale.setdefault("frame_width", video.width)
        scale.setdefault("frame_height", video.height)
    return scale


def _run_video(run):
    """The Video row this run was launched on, or None.

    Needed because detected reference boxes are in pixels and the frame size
    that normalises them was measured at ingest onto the video row.
    """
    from apps.videos.models import Video

    from . import aggregate

    vid = aggregate.run_video_id(run)
    return Video.objects.filter(pk=vid).first() if vid else None


def _gap_frames(step, default=15):
    try:
        return max(int(float((step.get("config") or {}).get("gap_frames", default))), 0)
    except (TypeError, ValueError):
        return default


def _proximity_radius(step):
    """Insect-to-insect radius, config'd as a percent of frame width."""
    from . import ops

    raw = (step.get("config") or {}).get("proximity_percent")
    if raw in (None, ""):
        return ops.DEFAULT_PROXIMITY
    try:
        return max(float(raw), 0.0) / 100.0
    except (TypeError, ValueError):
        return ops.DEFAULT_PROXIMITY


def _run_for_job(job):
    """The PipelineRun that spawned this job, if one did.

    There is no back-reference from Job, and the run records the job id inside
    its context JSON. Narrowed to runs contemporaneous with the job — the run
    that submitted it started moments before — so this stays a small scan
    rather than every run the user ever launched.
    """
    from datetime import timedelta

    from . import aggregate
    from .models import PipelineRun

    when = job.created_at
    if not when:
        return None
    nearby = (PipelineRun.objects
              .filter(user_id=job.user_id,
                      started_at__gte=when - timedelta(days=1),
                      started_at__lte=when + timedelta(days=1))
              .order_by("-started_at")[:400])
    for run in nearby:
        if aggregate.run_job_id(run) == job.pk:
            return run
    return None


def primitives_for_job(job, kind):
    """Events or interactions for one job, however it was launched.

    The per-clip results page shows a job, not a run, and used to render the
    worker's own interactions CSV — so it disagreed with the batch export,
    which computes the primitives. Two tables of the same clip saying different
    things is worse than either being wrong on its own.

    Prefers the real run, so a drawn ROI or a device layout is honoured. Falls
    back to a synthetic one carrying just this job's result and video, which
    resolves references from the detector's own boxes.
    """
    from .models import PipelineRun

    result = getattr(job, "result", None)
    if result is None:
        return []

    run = _run_for_job(job)
    if run is None:
        payload = {f: getattr(result, f, None) for f in (
            "events_csv_path", "tracking_csv_path", "detections_csv_path",
            "interactions_csv_path", "summary_stats", "unique_tracks",
            "entry_count", "exit_count", "total_events", "interaction_count")}
        run = PipelineRun(user_id=job.user_id)
        # A reference.layout step so the clip's own hotel geometry is used —
        # that is what a pipeline on this device would have resolved. When the
        # device has no saved layout it resolves to nothing and the detector's
        # own boxes take over, which is the right order of preference either
        # way: the user's geometry is intent, a detection is a guess.
        run.steps = [
            {"id": "v", "block_type": "input.video",
             "config": {"video_id": str(job.video_id)}},
            {"id": "r", "block_type": "reference.layout",
             "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
            {"id": "a", "block_type": "analyze.interactions", "config": {},
             "inputs": {"tracks": "m", "rois": "r"}},
        ]
        run.context = {"v": {"artifact": "video", "video_id": job.video_id},
                       "m": {"artifact": "tracks", "result": payload}}
    return recompute_primitive(run, kind)


def recompute_primitive(run, kind):
    """Events or interactions for a run whose analyzer never produced them.

    A batch analysed before the primitives existed has no ``events`` or
    ``interactions`` rows in its context — only whatever its retired analyzer
    stored — so an export would have to fall back to the worker's own CSV, and
    that is the file with the centroid-distance bug.

    Nothing about the answer requires a re-run, though: the tracking table and
    the reference geometry are both already saved, and the primitives are a
    pure function of them. This recomputes at export time so an old batch
    exports the same rows a fresh one would, without touching the GPU.

    Returns [] when the run has no readable tracking table.
    """
    from . import aggregate

    result = aggregate.run_gpu_result(run)
    if not result.get("tracking_csv_path"):
        return []

    steps = list(run.steps or [])
    # Resolve the reference the way the analyzer would have: from the analyzer's
    # own position in the graph, so an explicit `rois` edge still wins.
    index = next((i for i in range(len(steps) - 1, -1, -1)
                  if str(steps[i].get("block_type", "")).startswith("analyze.")),
                 max(len(steps) - 1, 0))
    step = steps[index] if steps else {}
    inputs = {"tracks": {"result": result, "label": ""}}

    executor = (_exec_analyze_events if kind == "events"
                else _exec_analyze_interactions)
    out = executor(step, run, run.context or {}, inputs, index)
    return out.get("rows") or []


def _exec_analyze_events(step, run, context, inputs, index):
    """Primitive 1 — every boundary crossing in the clip.

    Two sources, one table: crossings of the user's references, derived from the
    episode pass, and the worker's own Entry/Exit classifications against nests
    it detected. Both normalise into the Event schema and each row says which it
    came from, so neither can quietly stand in for the other.
    """
    from . import ops, primitives

    tidy, refs, result, fps, fps_source, ref_source, taxa = _analysis_inputs(
        step, run, context, inputs, index)

    rows = primitives.events_from_gpu(ops.load_events_df(result), fps)
    if tidy is not None and refs:
        episodes = ops.compute_episodes(tidy, refs, gap_frames=_gap_frames(step))
        rows = primitives.events_from_episodes(episodes, fps) + rows
        rows.sort(key=lambda r: (r["frame"], r["action"], str(r["subject"])))
    primitives.label_references(rows, refs, key="target")

    out = {
        "artifact": "events", "table_kind": "events",
        "fps": fps, "fps_source": fps_source,
        "reference_source": ref_source, "reference_count": len(refs),
        "taxa": taxa["taxa"], "taxa_source": taxa["how"],
        "csv": result.get("events_csv_path", ""),
        **primitives.summarize_events(rows),
    }
    if tidy is None:
        out["note"] = ("No readable tracking table — only the worker's own nest "
                       "events are listed.")
    elif not refs:
        out["note"] = ("No reference upstream and none detected, so only the "
                       "worker's own nest events are listed. Draw an ROI, use the "
                       "device nest layout, or wire a Detect node for the "
                       "reference class into the analyzer.")
    return out


def _exec_analyze_interactions(step, run, context, inputs, index):
    """Primitive 2 — every episode of two things being together.

    Insect-to-reference episodes come from the local pass over the user's ROI;
    insect-to-insect proximity comes from the worker, which computes it while
    the tracks are already in memory. A visit is simply an insect-to-reference
    row, which is why Visitation Count is no longer a block of its own.
    """
    from . import ops, primitives

    tidy, refs, result, fps, fps_source, ref_source, taxa = _analysis_inputs(
        step, run, context, inputs, index)
    want = (step.get("config") or {}).get("interaction_type", "all")

    gap = _gap_frames(step)
    rows = primitives.interactions_from_gpu(ops.load_interactions_df(result), fps)

    if tidy is not None:
        # Both halves are computed here when the tracks are readable, so the
        # whole table honours one gap tolerance and one set of thresholds. The
        # worker's rows are the fallback for jobs whose tracking CSV is gone.
        #
        # This is also a correctness fix, not just tidiness: the worker matches
        # an insect to a reference by centroid-to-centroid distance under a flat
        # 50 px, which discards the reference's size entirely. A bee sitting on
        # the edge of a 400 px flower is 200 px from its centre and was simply
        # never recorded — visibly inside the box in the annotated video, absent
        # from the table. compute_episodes asks containment instead.
        local = []
        if refs:
            local += primitives.interactions_from_episodes(
                ops.compute_episodes(tidy, refs, gap_frames=gap), fps)
        local += primitives.interactions_from_proximity(
            ops.compute_proximity_episodes(
                tidy, radius=_proximity_radius(step), gap_frames=gap,
                aspect=ops.frame_aspect(result.get("summary_stats") or result)),
            fps)
        # Keep only what the local pass could not produce: reference rows when
        # no reference was defined, and nothing else.
        keep_gpu = [r for r in rows
                    if r["b_kind"] == primitives.REFERENCE and not refs]
        rows = local + keep_gpu

    if want == "organism_organism":
        rows = [r for r in rows if r["b_kind"] == primitives.ORGANISM]
    elif want == "organism_reference":
        rows = [r for r in rows if r["b_kind"] == primitives.REFERENCE]
    rows.sort(key=lambda r: (r["start_frame"] is None, r["start_frame"], str(r["a"])))
    primitives.label_references(rows, refs, key="b")

    out = {
        "artifact": "table", "table_kind": "interactions",
        "fps": fps, "fps_source": fps_source,
        "reference_source": ref_source, "reference_count": len(refs),
        "taxa": taxa["taxa"], "taxa_source": taxa["how"],
        "csv": result.get("interactions_csv_path", ""),
        **primitives.summarize_interactions(rows),
    }
    if tidy is None and not rows:
        out["note"] = "No readable tracking or interactions table for this job."
    return out


def _exec_analyze_interaction(step, run, context, inputs, index):
    """Module 3 — proximity interactions, computed on the GPU during tracking."""
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    want = (step.get("config") or {}).get("interaction_type", "all")
    # The worker writes these literals into the interactions CSV.
    kinds = {
        "organism_organism": "organism-to-organism",
        "organism_reference": "organism-to-reference",
    }

    df = ops.load_interactions_df(result)
    if df is not None:
        summary = ops.summarize_interactions(df, kind=kinds.get(want))
        return {
            "artifact": "table", "table_kind": "interaction",
            "csv": result.get("interactions_csv_path", ""),
            **summary,
        }
    return {
        "artifact": "table", "table_kind": "interaction",
        "interaction_count": result.get("interaction_count", 0),
        "rows": [],
        "csv": result.get("interactions_csv_path", ""),
        "note": "Derived from the job summary (interactions CSV not available).",
    }


def _exec_analyze_detection_count(step, run, context, inputs, index):
    """Module 3 — count detections instead of trips/visits.

    Prefers the raw detections CSV (every detection the detector emitted, before
    the tracker associated anything). Jobs run before the worker started writing
    that file have none, so they fall back to the *tracked* table — which counts
    only detections that made it into a confirmed track, and therefore
    undercounts. The output says which source was used.
    """
    from . import ops

    up = inputs.get("detections") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    cfg = step.get("config") or {}
    metric = cfg.get("metric", "total")
    reference = find_reference(run.steps, index, context, run)
    boxes = ops.roi_shapes(reference)

    label = _upstream_label(inputs)

    # Sampled detection returns per-frame boxes rather than CSVs. Distinct and
    # modal only mean anything there: they answer "how many objects are there",
    # not "how many detections were made".
    frames = (result.get("summary_stats") or {}).get("sampled_frames")
    if metric in ("distinct", "modal"):
        if frames is None:
            return {
                "artifact": "table", "table_kind": "detection_count",
                "metric": metric, "rows": [],
                "note": "This metric needs sampled detection. Set the Detect "
                        "node's 'Frames to analyse' to 'Sampled frames' and "
                        "re-run — counting distinct objects is not meaningful "
                        "over a full tracking pass.",
            }
        detector = _upstream_detector(run.steps, index)
        want = label or (detector_label(detector) if detector else "")
        if metric == "distinct":
            try:
                iou = float(cfg.get("iou_threshold", 0.5) or 0.5)
            except (TypeError, ValueError):
                iou = 0.5
            summary = ops.count_distinct_objects(frames, want, iou_threshold=iou)
            note = (f"{summary['distinct_objects']} distinct object(s) across "
                    f"{summary['frames_sampled']} sampled frame(s); boxes "
                    f"overlapping by IoU >= {iou} counted once.")
        else:
            summary = ops.modal_frame_count(frames, want)
            note = (f"Most common per-frame count across "
                    f"{summary['frames_sampled']} sampled frame(s); "
                    f"{summary.get('frames_agreeing', 0)} frame(s) agreed.")
        # No frame rate here: this branch counts sampled frames, not time.
        return {"artifact": "table", "table_kind": "detection_count",
                "metric": metric, "note": note, **summary}

    df = ops.filter_by_label(ops.load_detections_df(result), label)
    raw = df is not None
    if raw:
        note = ("Counted from the raw detector output — every detection, "
                "including ones the tracker never associated into a track.")
        # The raw table has no track_id; give normalized_tracks a stand-in so
        # its shared column/coordinate handling still applies. It is never
        # reported as a track count (count_tracks=False below).
        if "track_id" not in {c.lower() for c in df.columns}:
            df = df.copy()
            df["track_id"] = range(len(df))
    else:
        df = ops.filter_by_label(ops.load_tracking_df(result), _upstream_label(inputs))
        note = ("Counted from the tracked-detection table — this job predates "
                "raw-detection export, so detections the tracker discarded are "
                "not included. Re-run it to count raw detections.")

    tidy = (ops.normalized_tracks(df, _frame_scale(run, result))
            if df is not None else None)
    if tidy is not None:
        fps, fps_source = ops.fps_with_source(result)
        if metric == "over_time":
            try:
                bin_sec = float(cfg.get("bin_seconds", 5) or 5)
            except (TypeError, ValueError):
                bin_sec = 5.0
            series = ops.compute_colony_activity(
                tidy, boxes, fps, metric="motion", bin_sec=bin_sec,
            )
            return {"artifact": "table", "table_kind": "detection_count",
                    "metric": metric, "note": note,
                    "fps": fps, "fps_source": fps_source, **series}
        summary = ops.compute_detection_counts(
            tidy, boxes, fps, per_frame=(metric == "per_frame"), count_tracks=not raw,
        )
        return {"artifact": "table", "table_kind": "detection_count",
                "metric": metric, "note": note,
                "fps": fps, "fps_source": fps_source, **summary}

    return {
        "artifact": "table", "table_kind": "detection_count",
        "metric": metric, "rows": [],
        "unique_tracks": result.get("unique_tracks", 0),
        "note": "Derived from the job summary (tracking CSV not available).",
    }


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
    """Retired — kept runnable, and now sharing the primitives' reference
    resolution so an existing Visitation pipeline benefits from the detected-
    reference fallback without having to be rewired."""
    from . import ops

    tidy, refs, result, fps, fps_source, ref_source, _taxa = _analysis_inputs(
        step, run, context, inputs, index)
    if tidy is not None:
        if not refs:
            return {"artifact": "table", "table_kind": "visitation",
                    "reference_source": ref_source,
                    "note": "No reference upstream, and the detector found none — "
                            "draw an ROI, use the device nest layout, or wire a "
                            "Detect node for the reference class into the analyzer."}
        summary = ops.compute_visitation(tidy, refs, fps)
        # Carried so the page can say "dwell times assume 30 fps" instead of
        # presenting a guessed rate as a measurement.
        return {"artifact": "table", "table_kind": "visitation",
                "fps": fps, "fps_source": fps_source,
                "reference_source": ref_source, "reference_count": len(refs),
                **summary}

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
    roi = find_reference(run.steps, index, context, run)
    boxes = ops.roi_shapes(roi)

    df = ops.filter_by_label(ops.load_tracking_df(result), _upstream_label(inputs))
    tidy = (ops.normalized_tracks(df, _frame_scale(run, result))
            if df is not None else None)
    if tidy is not None:
        fps, fps_source = ops.fps_with_source(result)
        series = ops.compute_colony_activity(tidy, boxes, fps, metric=metric)
        return {"artifact": "table", "table_kind": "colony_activity",
                "fps": fps, "fps_source": fps_source, **series}
    return {
        "artifact": "table",
        "table_kind": "colony_activity",
        "metric": metric,
        "rows": [],
        "note": "Derived from job summary (tracking CSV not available).",
    }


def _exec_identify_species(step, run, context, inputs, index):
    """Report the species each track was classified as.

    The GPU wrote the winner into the tracking CSV's taxon column during the
    tracking pass (voting across every frame), so this is a pure read — the
    taxon_confidence / taxon_votes columns say how firm each call was.
    """
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    df = ops.filter_by_label(ops.load_tracking_df(result), _upstream_label(inputs))
    summary = ops.species_identities(df) if df is not None else None
    if summary is not None and summary.get("rows"):
        return {"artifact": "table", "table_kind": "species", **summary}
    return {
        "artifact": "table", "table_kind": "species",
        "identified_tracks": 0, "unique_taxa": 0, "rows": [],
        "note": "No species labels in the tracking data. Species classification "
                "runs during tracking, so this pipeline has to be re-run with "
                "this node present for the worker to produce them.",
    }


def _exec_identify_marker(step, run, context, inputs, index):
    """Identity add-on — per-individual marker IDs on top of tracks.

    Two sources, in order of quality:

    1. ``bee_id`` columns in the tracking CSV, if the tracker decoded markers
       itself. Nothing sets those today (the tracker's identifier hook is not
       wired on the GPU worker), but reading them first means this step needs no
       change when it is.
    2. Otherwise, decode the per-track crops the job already uploaded to S3.
       Runs on CPU here, works on videos analysed long before the decoder
       existed, and votes across a track's crops rather than trusting one read.
    """
    from . import markers, ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    marker_type = (step.get("config") or {}).get("marker_type", "auto")

    df = ops.filter_by_label(ops.load_tracking_df(result), _upstream_label(inputs))
    ident = ops.marker_identities(df) if df is not None else None
    if ident is not None:
        return {"artifact": "table", "table_kind": "marker_id",
                "source": "tracker", **ident}

    ident = markers.identify_from_crops(result, marker_type=marker_type)
    if ident is not None and ident.get("rows"):
        return {"artifact": "table", "table_kind": "marker_id", **ident}

    if ident is None and marker_type not in ("auto", "color"):
        note = (f"No decoder for '{marker_type}' markers yet — only colour marks "
                "can be read at the moment. Set Marker Type to Colour or Auto.")
    elif ident is None:
        note = ("No per-track crops are stored for this job, so there is nothing "
                "to read markers from. Crops are saved during tracking — re-run "
                "the pipeline to produce them.")
    else:
        note = ("Crops were read but no marker was legible in any of them. If "
                "your bees are marked, the paint may be too small or too dim in "
                "these crops to classify.")
    return {
        "artifact": "table", "table_kind": "marker_id",
        "identified_tracks": 0, "unique_markers": 0, "rows": [], "note": note,
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
    # ── The three-module palette ──
    "reference.layout": lambda s, r, c, i, idx: resolve_reference(s, r, c, idx) or {
        "artifact": "roi", "source": (s.get("config") or {}).get("source", "device_layout"),
    },
    "track.mot": _exec_mot,
    "analyze.events": _exec_analyze_events,
    "analyze.interactions": _exec_analyze_interactions,
    "analyze.detection_count": _exec_analyze_detection_count,
    # ── Legacy (still runnable; not in the palette) ──
    "input.image_set": lambda s, r, c, i, idx: {"artifact": "frames", "source": (s.get("config") or {}).get("source", "device_crops")},
    "roi.nest_layout": _exec_roi_nest_layout,
    "roi.draw": _exec_roi_draw,
    "analyze.interaction": _exec_analyze_interaction,
    "analyze.foraging_trips": _exec_analyze_foraging_trips,
    "analyze.visitation": _exec_analyze_visitation,
    "analyze.colony_activity": _exec_analyze_colony_activity,
    "identify.marker": _exec_identify_marker,
    "identify.species": _exec_identify_species,
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

_DETECT_TRACK_BLOCKS = {"detect.objects", "detect.bee", "detect.nest", "track.bee"}


def downstream_ids(step_id, steps):
    """Yield every step transitively downstream of ``step_id``.

    Reverse of ``upstream_ids``, including the legacy linear fallback (a step with
    no ``inputs`` map implicitly consumes the previous one). Used for the
    lookahead flags below: some job settings are chosen on a *later* node because
    the GPU produces their output during tracking. Walking real edges rather than
    scanning the whole step list matters on a branched graph — one Detector's job
    must not pick up config from a sibling branch's analyzer.
    """
    by_id = {s.get("id"): (i, s) for i, s in enumerate(steps)}
    children = {}
    for i, s in enumerate(steps):
        for up in upstream_ids(s, steps, i):
            children.setdefault(up, []).append(s.get("id"))

    seen, frontier = set(), list(children.get(step_id, []))
    while frontier:
        sid = frontier.pop()
        if sid in seen or sid not in by_id:
            continue
        seen.add(sid)
        yield by_id[sid][1]
        frontier.extend(children.get(sid, []))


def _resolve_custom_models(cfg, run, config):
    """Resolve node-picked custom models (pk -> S3 key) into ``config``.

    Empty selection means the built-in models. Resolved by pk against the run
    owner's own models — same contract as the New Analysis form's
    custom_bee_model / custom_nest_model selects. The Detect module names these
    object_model/reference_model; the legacy nodes bee/nest_model.

    Returns an error string, or None on success.
    """
    for fields, config_key, label in (
        (("object_model", "bee_model"), "custom_bee_model_path", "object"),
        (("reference_model", "nest_model"), "custom_nest_model_path", "reference"),
    ):
        raw = next((cfg[f] for f in fields if cfg.get(f)), None)
        if not raw:
            continue
        from apps.training.models import CustomModel
        try:
            model_pk = int(raw)
        except (TypeError, ValueError):
            return f"Invalid {label} model selection on this node."
        cm = CustomModel.usable(run.user).filter(pk=model_pk).first()
        if not cm:
            return f"The selected {label} model is unavailable (removed or deactivated)."
        config[config_key] = cm.storage_key
    return None


def is_sampled(step):
    """True when a Detect node analyses sampled frames rather than every one."""
    return (step.get("config") or {}).get("analyse") == "sampled"


def build_sampled_detection_config(step, run, context, index):
    """Job config for sampled detection — the worker's `pre_annotate` task.

    Static objects don't move, so counting them doesn't need the tracking pass
    over every frame. The worker already samples every Nth frame, runs the
    detector and returns per-frame boxes; this just points a Detect node at it.
    Far cheaper: 20 frames instead of ~18,000 for a 10-minute clip.

    Deliberately NOT merged with the tracking config — the two produce different
    result shapes, and a shared builder would blur which one a step gets.
    """
    video_out = find_artifact("video", run.steps, index, context)
    if not video_out:
        return None, "No upstream video for this GPU step."

    cfg = step.get("config") or {}
    label = detector_label(step)
    if not label:
        return None, "This Detect node needs a label to sample for."

    def _clamp(name, default, lo, hi):
        # `or default` would be wrong here: an explicit 0 is a real value that
        # should clamp to the floor, not silently fall back to the default.
        raw = cfg.get(name)
        if raw in (None, ""):
            raw = default
        try:
            return max(lo, min(hi, int(float(raw))))
        except (TypeError, ValueError):
            return default

    raw_family = cfg.get("model_family") or "yolo"
    config = {
        "task": "pre_annotate",
        # Only this node's own class — sampled detection is per-node, so unlike
        # the tracking path there is no shared result to filter afterwards.
        "classes": [p.strip() for p in label.split(",") if p.strip()],
        "detector_kind": "sam3" if str(raw_family).lower() == "sam3" else "yolo",
        "sample_interval": _clamp("sample_interval", 30, 1, 600),
        "max_frames": _clamp("max_frames", 20, 1, 300),
        "confidence_threshold": float(cfg.get("confidence", 0.4) or 0.4),
        "selection": "uniform",
    }
    model_err = _resolve_custom_models(cfg, run, config)
    if model_err:
        return None, model_err
    return {"video_id": video_out["video_id"], "config": config}, None


def _all_detector_labels(steps):
    """Every label any *full-frame* Detect node asks for, de-duped, in order.

    All Detect nodes on a video share one GPU pass. Collecting their labels here
    means each node's job config comes out **identical**, so the first one submits
    and the rest hit the StepResult cache — the existing cache does the
    de-duplication for free, and adding a class costs no extra compute.
    """
    labels = []
    for s in steps or []:
        if s.get("block_type") != "detect.objects" or is_sampled(s):
            continue  # sampled nodes run their own job; they share nothing
        label = detector_label(s)
        if label and label not in labels:
            labels.append(label)
    return labels


def _upstream_detector(steps, index):
    """The Detect node feeding this step, if any — for label filtering."""
    for _sid, i, step in _walk_upstream(steps, index):
        if step is not None and step.get("block_type") == "detect.objects":
            return step
    return None


def _run_tracking_for(step):
    """Whether this GPU step's job should run detection + tracking.

    A Detect node always runs the full pass. ``run_tracking=False`` sends the
    worker down a nest-only path that writes no CSVs at all — not even
    detections — so it can never serve a Detect node, whose entire output is the
    detections table. Only the legacy nest-only block uses that path.
    """
    block_type = step.get("block_type")
    if block_type == "detect.objects":
        # Legacy graphs may still carry run_scope from before the class split.
        return (step.get("config") or {}).get("run_scope", "full") != "reference_only"
    if block_type == "detect.nest":  # legacy nest-only fast path
        return False
    return block_type in ("track.bee", "detect.bee")


def _pipeline_species(step, steps):
    """Species-classification settings from a downstream Identify Species node.

    Unlike the marker flag this was replaced with, these keys are consumed all
    the way down: _spawn_gpu_job forwards them, the handler passes them to
    CloudPipeline, and the tracker classifies with them. They belong in the
    hashed job config precisely *because* they change the output — the taxon
    column differs — so adding the node correctly forces a re-run instead of
    serving a cached result computed without it.
    """
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") != "identify.species":
            continue
        cfg = s.get("config") or {}
        try:
            floor = float(cfg.get("min_confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            floor = 0.5
        return True, floor
    return False, 0.5


def _pipeline_tracker(step, steps):
    """Tracking algorithm, read from a downstream MOT node (default BeeTrack).

    Inert on the worker today — there is only one tracker — but the value rides
    the job config so a second algorithm is a worker change, not a builder change.
    """
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") == "track.mot":
            return (s.get("config") or {}).get("tracker", "beetrack") or "beetrack"
    return "beetrack"


# Blocks that carry the Entry/Exit classifier cutoff. Events are computed
# during tracking, so the Track step reads it from whichever analyzer is
# downstream — the Events primitive, or the retired Foraging Trips node on
# pipelines saved before it.
_EVENT_CONFIDENCE_BLOCKS = ("analyze.events", "analyze.foraging_trips")


def _pipeline_event_confidence(step, steps):
    """Entry/Exit event-classifier cutoff, from a downstream Events node.
    Defaults to 0.6 when no such node is downstream."""
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") in _EVENT_CONFIDENCE_BLOCKS:
            try:
                return float((s.get("config") or {}).get("event_confidence", 0.6) or 0.6)
            except (TypeError, ValueError):
                return 0.6
    return 0.6


def build_job_config(step, run, context, index):
    """The job config for a GPU step, whichever kind it is.

    Single entry point on purpose: engine._gpu_cache_key and submit_gpu_step both
    call this, so the hashed config is always exactly the one submitted. A
    sampled Detect node produces a completely different job (and result shape)
    from a tracking one, and routing it anywhere else would let the two drift.
    """
    if step.get("block_type") == "detect.objects" and is_sampled(step):
        return build_sampled_detection_config(step, run, context, index)
    return build_detect_and_track_config(step, run, context, index)


def build_detect_and_track_config(step, run, context, index):
    """Assemble the ``detect_and_track`` Job config for a detect/track GPU step.

    Pulls the source video + any upstream ROI/nest layout. Tracking always
    uploads per-track crops for later species ID.
    """
    video_out = find_artifact("video", run.steps, index, context)
    if not video_out:
        return None, "No upstream video for this GPU step."

    cfg = (step.get("config") or {})
    # Detector choice: "model_family" on the Detect module, "detector" on the
    # legacy detect.bee/track.bee nodes.
    raw_family = cfg.get("model_family") or cfg.get("detector") or "yolo"
    detector_kind = "sam3" if str(raw_family).lower() == "sam3" else "yolo"
    # Every Detect node's label rides one job. For SAM 3 the union becomes the
    # grounding prompt (the detector already takes a comma-separated list and
    # labels each detection with the prompt that matched). For YOLO the model
    # detects its trained classes regardless; nodes filter by taxon on read.
    labels = _all_detector_labels(run.steps)
    prompt = ",".join(labels) if labels else (cfg.get("text_prompt", "") or "").strip()
    config = {
        "detection_mode": "yolo",
        "confidence_threshold": float(cfg.get("confidence", 0.4) or 0.4),
        # Entry/Exit event-classifier cutoff — set on the downstream Foraging
        # Trips node (events are produced during tracking).
        "ml_threshold": _pipeline_event_confidence(step, run.steps),
        "run_tracking": _run_tracking_for(step),
        # Detector: SAM 3 text-prompt tracking, else YOLO.
        "detector_kind": detector_kind,
        "text_prompt": prompt,
        # Informational for the worker + part of the cache key, so adding or
        # renaming a class re-runs rather than serving a stale result.
        "detect_labels": labels,
        # Never render the overlay video. It roughly doubled runtime, could time
        # out long clips on its own, and was unavailable for chunked runs
        # anyway — the merge cannot stitch one. Pinned False rather than read
        # from config so pipelines saved while the option existed don't keep
        # paying for it.
        "visualize": False,
        # Selected on the downstream MOT node; inert on the worker for now.
        "tracker": _pipeline_tracker(step, run.steps),
    }
    wants_species, species_floor = _pipeline_species(step, run.steps)
    if wants_species:
        config["identify_species"] = True
        config["species_min_confidence"] = species_floor
    legacy_ref = cfg.get("reference_source")
    if step.get("block_type") == "detect.objects" and legacy_ref:
        # Legacy graphs only: part of the hashed config so two pre-split Detectors
        # differing solely in how they got their reference don't share a result.
        config["reference_source"] = legacy_ref
    model_err = _resolve_custom_models(cfg, run, config)
    if model_err:
        return None, model_err
    # NOTE: this used to set identify_bees/marker_method when an identify.marker
    # node was present. Nothing consumed them — analysis.views._spawn_gpu_job
    # builds the SageMaker payload key-by-key and drops them — while
    # engine._gpu_cache_key hashes this whole dict, so their only observable
    # effect was busting the StepResult cache and re-billing a full GPU run for a
    # bit-identical result. Re-add them together with an actual marker decoder.
    # The Detector module carries its own reference config, so resolve it from
    # this step rather than walking upstream (nothing upstream of a Detector
    # produces an ROI). Legacy detect/track nodes still take theirs from an
    # upstream roi.* step.
    if step.get("block_type") == "detect.objects":
        roi_out = resolve_reference(step, run, context, index)
    else:
        roi_out = find_reference(run.steps, index, context, run)
    if roi_out:
        if roi_out.get("hotel_roi"):
            config["hotel_roi"] = roi_out["hotel_roi"]
        if roi_out.get("hotel_polygon"):
            config["hotel_polygon"] = roi_out["hotel_polygon"]
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

    built, err = build_job_config(step, run, context, index)
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
