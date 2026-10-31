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
    source, source_label = "device", ""
    # A clip's, or a photo's, layout — photos share the table (memory/45).
    video_out = (find_artifact("video", run.steps, index, context)
                 or find_artifact("photo", run.steps, index, context))
    if video_out:
        try:
            from apps.videos.models import Video

            video = Video.everything.select_related("device", "site").get(pk=video_out["video_id"])
            # The clip's own layout, its site's, or its unit's as in use when
            # it was recorded (memory/52).
            from apps.devices.layouts import layout_for_video
            layout = layout_for_video(video)
            hotel_roi = layout["roi_override"]
            hotel_polygon = layout["roi_polygon"]
            nest_layout = layout["nest_layout"] or None
            source, source_label = layout["source"] or "device", layout["source_label"]
        except Exception as exc:  # defensive — layouts are optional
            logger.info("detect.objects: could not read the clip's layout: %s", exc)
    return {
        "artifact": "roi",
        "hotel_roi": hotel_roi,
        "hotel_polygon": hotel_polygon,
        "nest_layout": nest_layout,
        "source": source,
        "source_label": source_label,
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
    node's label — the same label-filtering every Detect consumer does. Finding
    none of its class means no references.
    """
    from . import ops

    out = context.get(step.get("id")) or {}
    result = out.get("result") or {}
    label = detector_label(step)

    photo = (result.get("summary_stats") or {}).get("photo")
    if photo is not None:
        boxes = photo_reference_boxes(photo, label)
        return ({"artifact": "roi", "source": "detected", "label": label,
                 "regions": [{"box": b} for b in boxes]} if boxes else {})

    # A sampled Detect (static objects: flags, nest holes) returns frames, not
    # a detections table. Boxes seen on several frames are one object; numbered
    # by position (top to bottom, then left to right) so the same flag keeps
    # its id across a site's clips.
    sampled = (result.get("summary_stats") or {}).get("sampled_frames")
    if sampled:
        refs = _sampled_references(sampled, label, run, index, context)
        return ({"artifact": "roi", "source": "detected", "label": label, "nest_layout": refs}
                if refs else {})

    df = ops.load_detections_df(result)
    if df is None:
        df = ops.load_tracking_df(result)
    boxes = ops.boxes_for_label(df, label) if df is not None else []
    if boxes:
        return {"artifact": "roi", "source": "detected", "label": label,
                "regions": [{"box": b} for b in boxes]}
    # None of this node's class found: no references. Not the worker's own
    # nest boxes — the pipeline asked for this class, not for those (memory/60).
    return {}


def _sampled_references(frames, label, run, index, context):
    """Distinct objects in sampled detections, as 0..1 references with stable
    ids. Empty when the clip's frame size was never measured (the boxes are in
    pixels and can't be placed)."""
    from apps.videos.models import Video

    from . import ops

    video_out = find_artifact("video", run.steps, index, context) or {}
    video = Video.everything.filter(pk=video_out.get("video_id")).only("width", "height").first()
    if not (video and video.width and video.height):
        return []
    w, h = float(video.width), float(video.height)
    rows = ops.count_distinct_objects(frames, label)["rows"]
    return [{"id": r["object"],
             "box": [round(max(0.0, min(1.0, v)), 5)
                     for v in (r["x1"] / w, r["y1"] / h, r["x2"] / w, r["y2"] / h)]}
            for r in rows]


# ── Local executors ───────────────────────────────────────────────────────────

def _exec_input_photo(step, run, context, inputs, index):
    """The run's photo (memory/45) — a row of the videos table, kind=photo."""
    from apps.videos.models import Video

    photo_id = (step.get("config") or {}).get("video_id")
    if not photo_id:
        return {"error": "No photo selected."}
    try:
        photo = Video.manageable(run.user, photos=True).get(pk=photo_id, kind=Video.Kind.PHOTO)
    except (Video.DoesNotExist, ValueError, TypeError):
        return {"error": f"Photo {photo_id} not found, or not yours to analyze."}
    return {"artifact": "photo", "video_id": photo.pk, "storage_key": photo.storage_key,
            "title": photo.title or f"Photo {photo.pk}"}


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
    out = {
        "artifact": "video",
        "video_id": video.pk,
        "storage_key": getattr(video, "storage_key", ""),
        "title": getattr(video, "title", "") or f"Video {video.pk}",
    }
    # Part of the clip, chosen at launch (memory/53): GPU steps analyse only it.
    cfg = step.get("config") or {}
    for key in ("start_seconds", "end_seconds"):
        if cfg.get(key) not in (None, ""):
            out[key] = float(cfg[key])
    return out


def _exec_roi_nest_layout(step, run, context, inputs, index):
    video_out = find_artifact("video", run.steps, index, context)
    if not video_out:
        return {"error": "No upstream video for the ROI layout."}
    hotel_roi, hotel_polygon, nest_layout = None, None, None
    try:
        from apps.videos.models import Video

        video = Video.objects.select_related("device", "site").get(pk=video_out["video_id"])
        # The clip's own layout, its site's, or its unit's (memory/52).
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


def _tracks_label(run, inputs):
    """The class to take out of the shared tracking table, or "" for all of it.

    One GPU pass serves every Detect node, so a table can hold several classes
    and each analyzer takes its own. With ONE Detect class in the pipeline
    every track is the subject and nothing is filtered: the table's taxon
    column carries the voted SPECIES once Identify species has run, and
    matching the Detect class ("bee") against species names emptied every
    analyzer (memory/60).
    """
    if len(_all_detector_labels(run.steps)) <= 1:
        return ""
    return _upstream_label(inputs)


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

    # References come only from the pipeline (memory/60): a Reference node (the
    # saved layout) or a Detect node for the reference class. Nothing the GPU
    # worker found on its own, and the ROI is the tracking area, never one.
    roi = find_reference(run.steps, index, context, run)
    refs = ops.roi_references(roi)
    ref_source = _reference_source(roi) if refs else "none"

    # What the reference branch claims, so the subject branch can be "the rest"
    # when the detector's own class names differ from the configured ones. That
    # is what makes the pipeline taxon-agnostic: renaming a class moves rows
    # between branches by role, it does not empty one.
    raw = ops.load_tracking_df(result)
    taxa, taxa_how = ops.resolve_taxa(raw, _tracks_label(run, inputs),
                                      exclude=[(roi or {}).get("label")])
    df = ops.filter_by_label(raw, _tracks_label(run, inputs),
                             exclude=[(roi or {}).get("label")])
    # Tracks an Identify Species node left out by group (memory/65). The full
    # table is kept so the analyzer can say what the filter took out.
    dropped = set((up or {}).get("dropped_tracks") or [])
    unfiltered = None
    if dropped and df is not None and len(df):
        id_col = ops._pick(df, ops._ID_COLS)
        if id_col is not None:
            unfiltered = df
            df = df[~df[id_col].map(_int_or_none).isin(dropped)]
    tidy = ops.normalized_tracks(df, scale) if df is not None else None
    fps, fps_source = ops.fps_with_source(result, video)
    return tidy, refs, result, fps, fps_source, ref_source, {
        "taxa": sorted(taxa) if taxa else ([] if taxa == set() else None),
        "how": taxa_how,
        "aspect": ops.frame_aspect(scale),
        "video": video,
        "dropped": dropped if unfiltered is not None else set(),
        "unfiltered": (ops.normalized_tracks(unfiltered, scale)
                       if unfiltered is not None else None),
        "track_groups": (up or {}).get("track_groups") or {},
    }


def _reference_source(roi):
    """Where a pipeline's references came from, as results report it:
    ``layout`` (a saved layout or drawn regions) or ``detect:<class>``."""
    roi = roi or {}
    if roi.get("source") == "detected":
        return f"detect:{roi.get('label') or 'reference'}"
    return "layout"


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


DEFAULT_GAP_SECONDS = 0.6


def _gap_frames(step, fps):
    """The node's gap tolerance in frames of this clip.

    Set in seconds (``gap_seconds``) so it means the same time on any camera.
    A run launched before that holds ``gap_frames`` in its frozen steps, and is
    honoured as it was.
    """
    cfg = step.get("config") or {}
    try:
        if cfg.get("gap_seconds") not in (None, ""):
            return max(int(round(float(cfg["gap_seconds"]) * float(fps or 25))), 0)
        if cfg.get("gap_frames") not in (None, ""):
            return max(int(float(cfg["gap_frames"])), 0)
    except (TypeError, ValueError):
        pass
    return int(round(DEFAULT_GAP_SECONDS * float(fps or 25)))


# No minimum by default (memory/60): a quick pass-over is still a pass-over, and
# its duration is in the CSV for the reader to filter on.
DEFAULT_MIN_SECONDS = 0.0


def _min_seconds(step):
    raw = (step.get("config") or {}).get("min_seconds")
    if raw in (None, ""):
        return DEFAULT_MIN_SECONDS
    try:
        return max(float(raw), 0.0)
    except (TypeError, ValueError):
        return DEFAULT_MIN_SECONDS


def _proximity_radius(step):
    """Insect-to-insect radius in body lengths (memory/60)."""
    from . import ops

    return _number((step.get("config") or {}).get("proximity_bl"), ops.DEFAULT_PROXIMITY_BL)


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

    pinned = getattr(job, "_pinned_run", None)
    if pinned is not None:
        return pinned
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


def pin_run(job, run_pk):
    """Make ``run_pk`` the run behind this job object for everything that reads
    it (``_run_for_job``), when it really is one of the job's runs. A re-run
    re-uses the clip's GPU job, so a page opened from a batch names its run.
    Returns the pinned run or None."""
    import uuid

    from . import aggregate
    from .models import PipelineRun

    try:
        run = PipelineRun.objects.filter(pk=uuid.UUID(str(run_pk))).first() if run_pk else None
    except ValueError:
        run = None
    if run is None or aggregate.run_job_id(run) != job.pk:
        return None
    job._pinned_run = run
    return run


def primitives_for_job(job, kind):
    """Events or interactions for one job, however it was launched.

    The per-clip results page shows a job, not a run, and used to render the
    worker's own interactions CSV — so it disagreed with the batch export,
    which computes the primitives. Two tables of the same clip saying different
    things is worse than either being wrong on its own.

    A pipeline run's rows are what its own steps saved — the same rows the
    batch table, its CSVs and studies read. A pipeline with no step for this
    table has none: nothing is computed that the pipeline didn't ask for, or
    the clip page and the batch would disagree (memory/60).

    A job no pipeline launched (the old analysis form) has no steps to read,
    so it is counted against the clip's saved layout (``_job_run``).
    """
    result = getattr(job, "result", None)
    if result is None:
        return []

    run = _run_for_job(job)
    if run is not None:
        from .aggregate import analyzer_outputs

        return [row for k, out in analyzer_outputs(run) if k == kind
                for row in (out.get("rows") or [])]
    return recompute_primitive(_job_run(job), kind)


def measured_kinds(job):
    """Which of events / interactions this job's pipeline has a step for.
    A job no pipeline launched is counted for both (``primitives_for_job``)."""
    run = _run_for_job(job)
    if run is None:
        return {"events", "interactions"}
    from .aggregate import analyzer_outputs

    return {k for k, _out in analyzer_outputs(run) if k in ("events", "interactions")}


def _job_run(job):
    """The run behind a job, or a stand-in carrying just its result and video
    with the clip's saved layout as its reference (``primitives_for_job``)."""
    from .models import PipelineRun

    run = _run_for_job(job)
    if run is not None:
        return run
    result = job.result
    payload = {f: getattr(result, f, None) for f in (
        "tracking_csv_path", "detections_csv_path", "summary_stats", "unique_tracks")}
    run = PipelineRun(user_id=job.user_id)
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
    return run


def geometry_for_job(job):
    """The pipeline's reference output for a job — its ROI and reference
    objects — resolved the way the analyzer resolves them. {} when none."""
    if getattr(job, "result", None) is None:
        return {}
    run = _job_run(job)
    steps = list(run.steps or [])
    index = next((i for i in range(len(steps) - 1, -1, -1)
                  if str(steps[i].get("block_type", "")).startswith("analyze.")),
                 max(len(steps) - 1, 0))
    try:
        return find_reference(steps, index, run.context or {}, run) or {}
    except Exception:  # noqa: BLE001 - a half-built graph draws nothing
        logger.exception("geometry_for_job: could not resolve job %s", job.pk)
        return {}


def references_for_job(job):
    """The references a job's events were counted against."""
    from . import ops

    return ops.roi_references(geometry_for_job(job))


def analysis_notes_for_job(job):
    """What the run's events/interactions steps said about how they measured:
    no reference, tracking from before the detected column, overlapping
    references, rules the rows were counted under."""
    from .aggregate import analyzer_outputs

    run = _run_for_job(job)
    if run is None:
        return []
    notes = []
    for kind, out in analyzer_outputs(run):
        if kind not in ("events", "interactions"):
            continue
        rules = out.get("counting_rules") or 0
        if rules < _COUNTING_RULES and out.get("rows") is not None:
            # The reconcile sweep recounts every run below the current rules
            # from its saved tracks (recount.py).
            note = ("Waiting to be recounted with the current rules from this run's "
                    "saved tracks; this happens automatically.")
            if note not in notes:
                notes.append(note)
        if out.get("note") and out["note"] not in notes:
            notes.append(out["note"])
    return notes


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
    """Primitive 1 — exits, entries and stays (memory/60).

    One rule, over each track's detected rows, against the references the
    pipeline defines: an insect first seen in a reference came out of it, one
    last seen in a reference went in, one that started and ended in it without
    leaving stayed. Nothing from the worker; no duration filter.
    """
    from . import ops, primitives

    tidy, refs, result, fps, fps_source, ref_source, extra = _analysis_inputs(
        step, run, context, inputs, index)
    cfg = step.get("config") or {}
    rows, notes = [], _measurement_notes(tidy, refs, extra)
    filtered_out = None
    if tidy is not None and refs:
        first, last = _frames_analysed(run, fps, extra.get("video"))
        events = ops.compute_events(
            tidy, refs, fps, padding_bl=_number(cfg.get("padding_bl"), 0.0),
            aspect=extra["aspect"], first_frame=first, last_frame=last)
        rows = primitives.events_from_computed(events, fps)
        if extra["dropped"] and extra["unfiltered"] is not None:
            every = primitives.events_from_computed(ops.compute_events(
                extra["unfiltered"], refs, fps, padding_bl=_number(cfg.get("padding_bl"), 0.0),
                aspect=extra["aspect"], first_frame=first, last_frame=last), fps)
            filtered_out = _filtered_out(every, extra)
        if cfg.get("ml_scoring") in (True, "true", "on", "1", 1):
            from . import ml_scoring

            note = ml_scoring.score(rows, result, refs, extra)
            if note:
                notes.append(note)

    out = {
        "artifact": "events", "table_kind": "events",
        "fps": fps, "fps_source": fps_source,
        "reference_source": ref_source, "reference_count": len(refs),
        "taxa": extra["taxa"], "taxa_source": extra["how"],
        "tracking_has_detected": ops.has_detected(tidy) if tidy is not None else None,
        "counting_rules": _COUNTING_RULES,
        **primitives.summarize_events(rows),
    }
    if filtered_out:
        out["filtered_out"] = filtered_out
    if notes:
        out["note"] = " ".join(notes)
    return out


def _filtered_out(every, extra):
    """What an Identify Species group filter took out of this count: the
    tracks, the events they would have made (by group) and those events' keys,
    so reviewed events can be checked against it (memory/65)."""
    from . import review

    dropped, groups = extra["dropped"], extra["track_groups"]
    gone = [e for e in every if _int_or_none(e.get("track_id")) in dropped]
    by_group = {}
    for t in dropped:
        g = (groups.get(str(t)) or {}).get("group") or "?"
        by_group.setdefault(g, {"group": g, "tracks": 0, "events": 0})["tracks"] += 1
    for e in gone:
        g = (groups.get(str(_int_or_none(e.get("track_id")))) or {}).get("group") or "?"
        by_group.setdefault(g, {"group": g, "tracks": 0, "events": 0})["events"] += 1
    return {"tracks": len(dropped), "events": len(gone),
            "by_group": sorted(by_group.values(), key=lambda g: -g["tracks"]),
            "event_keys": [review.event_key(e) for e in gone]}


# Which event/interaction rules a saved output was counted under (memory/60).
# recount.CURRENT_RULES follows this.
_COUNTING_RULES = 4


def _measurement_notes(tidy, refs, extra, need_refs=True):
    """What a reader must know about how these rows were measured."""
    from . import ops

    notes = []
    if tidy is None:
        notes.append("No readable tracking table for this clip.")
        return notes
    if need_refs and not refs:
        notes.append("No reference in this pipeline, so no events or insect-to-"
                     "reference interactions. Add a Reference (saved layout) node, "
                     "or a Detect node for the reference class wired into the analyzer.")
    if not ops.has_detected(tidy):
        notes.append("This clip was tracked before the tracker recorded which rows "
                     "were detections, so track starts and ends may include predicted "
                     "positions. Re-run tracking to measure from detections only.")
    pairs = ops.overlapping_references(refs)
    if pairs:
        named = ", ".join(f"{a} and {b}" for a, b in pairs)
        notes.append(f"Overlapping references: {named}. A point in both counts once, "
                     "for the reference whose centre is nearest. Redraw them so they "
                     "don't overlap.")
    return notes


def _frames_analysed(run, fps, video=None):
    """(first, last) frame analysed: a chosen window's edges, else the clip's
    own (0 and its last frame). ``last`` is None when the clip's length is
    unknown — ``at_edge`` then says "unknown" rather than guessing."""
    start, end = _window_frames(run, fps)
    first = start or 0
    if end is not None:
        return first, end
    duration = getattr(video, "duration_seconds", None)
    if duration and fps:
        return first, max(int(float(duration) * float(fps)) - 1, 0)
    return first, None


def _window_frames(run, fps):
    """(first, last) frame of a chosen part of the clip (memory/53), from the
    input step's start/end seconds; None for an edge that is the clip's own."""
    cfg = {}
    for st in run.steps or []:
        if st.get("block_type") == "input.video":
            cfg = st.get("config") or {}
            break
    fps = float(fps or 25)
    try:
        start = float(cfg.get("start_seconds") or 0)
        end = cfg.get("end_seconds")
        return (int(start * fps) if start > 0 else None,
                int(float(end) * fps) if end not in (None, "") else None)
    except (TypeError, ValueError):
        return None, None


def _exec_analyze_interactions(step, run, context, inputs, index):
    """Primitive 2 — time spent in a reference, and time near another insect.

    Both halves computed here from detected rows: insect-in-reference against
    the pipeline's references (grown by padding, in body lengths), insect-near-
    insect within a radius in body lengths. No minimum length unless the node
    sets one — a quick pass-over is still a pass-over, with its duration.
    """
    from . import ops, primitives

    tidy, refs, result, fps, fps_source, ref_source, extra = _analysis_inputs(
        step, run, context, inputs, index)
    cfg = step.get("config") or {}
    want = cfg.get("interaction_type", "all")
    gap = _gap_frames(step, fps)

    rows = []
    notes = _measurement_notes(tidy, refs, extra, need_refs=want != "organism_organism")
    if tidy is not None:
        if refs and want != "organism_organism":
            rows += primitives.interactions_from_episodes(
                ops.compute_episodes(tidy, refs, gap_frames=gap,
                                     padding_bl=_number(cfg.get("padding_bl"), 0.0),
                                     aspect=extra["aspect"]), fps)
        if want != "organism_reference":
            rows += primitives.interactions_from_proximity(
                ops.compute_proximity_episodes(
                    tidy, radius_bl=_proximity_radius(step), gap_frames=gap,
                    aspect=extra["aspect"]), fps)

    floor = _min_seconds(step)
    if floor:
        rows = [r for r in rows
                if r.get("duration_sec") is not None and float(r["duration_sec"]) >= floor]
    rows.sort(key=lambda r: (r["start_frame"], str(r["track_id"])))

    out = {
        "artifact": "table", "table_kind": "interactions",
        "fps": fps, "fps_source": fps_source,
        "reference_source": ref_source, "reference_count": len(refs),
        "taxa": extra["taxa"], "taxa_source": extra["how"],
        "tracking_has_detected": ops.has_detected(tidy) if tidy is not None else None,
        "counting_rules": _COUNTING_RULES,
        **primitives.summarize_interactions(rows),
    }
    if notes:
        out["note"] = " ".join(notes)
    return out


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

    photo = (result.get("summary_stats") or {}).get("photo")
    if photo is not None:
        return photo_detection_count(photo, boxes)

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
        df = ops.filter_by_label(ops.load_tracking_df(result), _tracks_label(run, inputs))
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


def _exec_analyze_colony_activity(step, run, context, inputs, index):
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    metric = (step.get("config") or {}).get("metric", "occupancy")
    roi = find_reference(run.steps, index, context, run)
    boxes = ops.roi_shapes(roi)

    df = ops.filter_by_label(ops.load_tracking_df(result), _tracks_label(run, inputs))
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

    The GPU worker classified every crop of every track after tracking and
    wrote each track's winner into the tracking CSV's taxon column, so this is
    a pure read — taxon_votes / taxon_vote_share say how firm each call was.
    When there is nothing, the worker's own account of why
    (summary_stats["identification"]["species"]) is the note.
    """
    from . import ops

    up = inputs.get("tracks") or _first_upstream_result(inputs)
    result = (up or {}).get("result", {})
    photo = (result.get("summary_stats") or {}).get("photo")
    if photo is not None:
        return {"artifact": "table", "table_kind": "species", **photo_species(photo)}
    df = ops.filter_by_label(ops.load_tracking_df(result), _tracks_label(run, inputs))
    summary = ops.species_identities(df) if df is not None else None
    if summary is None or not summary.get("rows"):
        summary = {"identified_tracks": 0, "unique_taxa": 0, "rows": [],
                   "note": _species_note(result)}
    # Tracks pass through, so analyzers wired after this node count only the
    # groups it keeps (memory/65). Wired off to the side, nothing reads them.
    out = {"artifact": "tracks", "table_kind": "species", **summary,
           "result": result, "label": (up or {}).get("label", "")}
    out.update(_species_groups(step, run, result, df, summary["rows"]))
    return out


def _species_groups(step, run, result, df, rows):
    """Each track's insect group and whether the filter counts it (memory/65).

    Worked out here, on the web, from the per-crop readings the worker saved:
    changing the groups or the rules is a recount, never a GPU run."""
    from . import aggregate, taxon_groups

    cfg = taxon_groups.settings(step.get("config"))
    try:
        readings = taxon_groups.crop_readings(result, df)
        groups = taxon_groups.track_groups(readings, cfg, aggregate.run_job_id(run))
    except Exception:  # noqa: BLE001 - no groups must never lose the count
        logger.exception("insect groups failed for run %s", run.pk)
        return {"groups_note": "Insect groups could not be worked out; every track is counted."}
    for row in rows:
        g = groups.get(_int_or_none(row.get("track")))
        if g:
            row.update(group=g["group"], group_share=g["share"], counted=g["counted"])
    counts = {}
    for g in groups.values():
        c = counts.setdefault(g["group"], {"group": g["group"], "tracks": 0, "counted": g["counted"]})
        c["tracks"] += 1
    order = taxon_groups.all_groups(cfg["custom"]) + [taxon_groups.UNCLEAR]
    return {
        "group_filter": cfg["on"],
        "leave_out": cfg["leave_out"] if cfg["on"] else [],
        "group_options": taxon_groups.all_groups(cfg["custom"]),
        "track_groups": {str(t): g for t, g in groups.items()},
        "group_counts": sorted(counts.values(), key=lambda c: (
            order.index(c["group"]) if c["group"] in order else len(order))),
        "dropped_tracks": sorted(t for t, g in groups.items() if not g["counted"]),
        "names_seen": taxon_groups.names_seen(readings),
    }


def _int_or_none(value):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if f == int(f) else None


def photo_detection_count(photo, regions=None):
    """Detection count on one photo: per class, and per reference region when
    the pipeline has one — the device's saved layout or a Detect node's boxes
    (a box's centre inside the region's shape)."""
    from . import ops

    dets = photo.get("detections") or []
    w, h = float(photo.get("width") or 1), float(photo.get("height") or 1)
    by_class = {}
    for d in dets:
        by_class[d.get("class") or "object"] = by_class.get(d.get("class") or "object", 0) + 1
    rows = [{"class": c, "count": n} for c, n in sorted(by_class.items())]
    out = {"artifact": "table", "table_kind": "detection_count", "metric": "photo",
           "detections": len(dets), "rows": rows, "tiles": photo.get("tiles")}
    if regions:
        per_region = []
        for i, shape in enumerate(regions):
            inside = sum(1 for d in dets if ops.in_any_box(
                (d["x"] + d["w"] / 2) / w, (d["y"] + d["h"] / 2) / h, [shape]))
            per_region.append({"region": i + 1, "count": inside})
        out["regions"] = per_region
    return out


def photo_reference_boxes(photo, label=""):
    """A photo's detections of ``label`` as 0..1 reference boxes — the flowers
    or tubes a Detect node found, for the analyzers to measure against. The job
    only looked for its own classes, so with no class match every box counts."""
    dets = photo.get("detections") or []
    want = (label or "").strip().lower()
    hits = [d for d in dets if (d.get("class") or "").lower() == want] if want else []
    w, h = float(photo.get("width") or 1), float(photo.get("height") or 1)
    return [[d["x"] / w, d["y"] / h, (d["x"] + d["w"]) / w, (d["y"] + d["h"]) / h]
            for d in (hits or dets)]


def photo_species(photo):
    """One row per insect in a photo: its species and the model's confidence.
    No floor (memory/60): the confidence is a column, the reader filters."""
    rows, taxa = [], set()
    for d in photo.get("detections") or []:
        taxon, conf = d.get("species"), d.get("species_confidence")
        rows.append({"insect": d.get("id"), "class": d.get("class"), "taxon": taxon or "",
                     "confidence": conf, "crop_key": d.get("crop_key")})
        if taxon:
            taxa.add(taxon)
    out = {"identified_tracks": sum(1 for r in rows if r["taxon"]), "unique_taxa": len(taxa),
           "rows": rows, "unit": "insect"}
    status = photo.get("species_status")
    if not rows:
        out["note"] = "No insects were detected in this photo."
    elif status and not status.get("loaded"):
        out["note"] = f"Species model could not be loaded: {status.get('error') or 'unknown error'}."
    return out


def _number(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _species_note(result):
    """Why a run has no species, from what the worker recorded."""
    stats = (result or {}).get("summary_stats") or {}
    status = (stats.get("identification") or {}).get("species")
    if not status:
        return ("No species labels in the tracking data. Species are identified "
                "on the GPU after tracking, so this pipeline has to be re-run "
                "with this node present.")
    model = "BioCLIP" if status.get("model") == "bioclip" else "BeeMachine"
    if not status.get("loaded"):
        return f"{model} could not be loaded: {status.get('error') or 'unknown error'}."
    if not status.get("tracks"):
        return f"{model} was ready, but the run saved no track crops to classify."
    return (f"{model} classified {status.get('crops', 0)} crops from "
            f"{status.get('tracks', 0)} tracks but named no species.")


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

    df = ops.filter_by_label(ops.load_tracking_df(result), _tracks_label(run, inputs))
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
    "input.photo": _exec_input_photo,
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
    # Retired questions run as the primitives they were reads over (memory/60):
    # trips and visits are left to the reader.
    "analyze.interaction": _exec_analyze_interactions,
    "analyze.foraging_trips": _exec_analyze_events,
    "analyze.visitation": _exec_analyze_interactions,
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


def _sample_interval_frames(cfg, video_out, clamp):
    """The node's sampling interval in frames of its clip.

    Set in seconds (``sample_seconds``); a node saved before that holds
    ``sample_interval`` in frames and keeps it.
    """
    if cfg.get("sample_seconds") in (None, "") and cfg.get("sample_interval") not in (None, ""):
        return clamp("sample_interval", 30, 1, 600)
    from apps.videos.models import Video

    fps = 25.0
    video = Video.everything.filter(pk=video_out.get("video_id")).only("fps").first()
    try:
        fps = float(getattr(video, "fps", None) or 25.0)
    except (TypeError, ValueError):
        pass
    try:
        seconds = float(cfg.get("sample_seconds") or 1.2)
    except (TypeError, ValueError):
        seconds = 1.2
    return max(1, min(600, int(round(seconds * fps))))


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
        "sample_interval": _sample_interval_frames(cfg, video_out, _clamp),
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
    """Species model from a downstream Identify Species node, or None.

    Consumed all the way down: _spawn_gpu_job forwards it, the worker votes
    every crop of every track with it after tracking. It belongs in the hashed
    job config because it changes the output (the taxon column), so adding or
    switching the node forces a re-run instead of serving a cached result.
    """
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") != "identify.species":
            continue
        model = str((s.get("config") or {}).get("model") or "beemachine").lower()
        return model if model in ("beemachine", "bioclip") else "beemachine"
    return None


def _pipeline_markers(step, steps):
    """Marker type from a downstream Read Marker node, or None. Same contract as
    species: the worker decodes and votes every crop after tracking."""
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") == "identify.marker":
            return (s.get("config") or {}).get("marker_type", "auto") or "auto"
    return None


# old frame-count field -> (its seconds field, the worker's frame parameter)
LEGACY_FRAME_SETTINGS = {
    "byte_track_buffer": ("byte_track_buffer_seconds", "track_buffer"),
    "ocsort_max_age": ("ocsort_max_age_seconds", "max_age"),
    "ocsort_min_hits": ("ocsort_min_hits_seconds", "min_hits"),
    "ocsort_delta_t": ("ocsort_delta_t_seconds", "delta_t"),
}


def tracker_settings(mot_config: dict) -> dict:
    """The selected tracker's settings from a MOT node's config, prefix
    stripped, coerced to numbers, and only where they differ from the default.
    """
    from .registry import TRACKER_FIELDS, TRACKER_SETTING_PREFIX

    cfg = mot_config or {}
    kind = (cfg.get("tracker") or "beetrack").lower()
    prefixes = TRACKER_SETTING_PREFIX.get(kind, ())
    out = {}
    # A config saved before durations moved to seconds (a run's frozen steps)
    # holds frame counts under the old names; the worker still takes those.
    legacy = set()
    for old, (new, param) in LEGACY_FRAME_SETTINGS.items():
        if any(old.startswith(p) for p in prefixes) and cfg.get(old) not in (None, "") \
                and cfg.get(new) in (None, ""):
            try:
                out[param] = int(float(cfg[old]))
                legacy.add(new)
            except (TypeError, ValueError):
                pass
    for field in TRACKER_FIELDS:
        name = field["name"]
        prefix = next((p for p in prefixes if name.startswith(p)), None)
        if prefix is None or name in legacy:
            continue
        value = cfg.get(name)
        # Compared with what the worker would use unsent: where the builder's
        # default differs from it (worker_default), an unset field still sends
        # the builder's default.
        default = field.get("worker_default", field["default"])
        if value in (None, ""):
            if "worker_default" not in field:
                continue
            value = field["default"]
        if field["field_type"] == "number":
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value.is_integer() and isinstance(default, int):
                value = int(value)
            if default is not None and float(value) == float(default):
                continue
        elif value == default:
            continue
        out[name[len(prefix):]] = value
    if kind == "beetrack":
        # The revive stage is retired: "Keep a lost track" covers the whole
        # time (registry.TRACKER_FIELDS). The worker's own default would
        # revive for 0.3 s more, so say 0.
        out["max_resurrection_seconds"] = 0.0
    return out


def _pipeline_tracker_settings(step, steps):
    """Settings for the tracker chosen on the downstream MOT node.

    Without one, the defaults: a Detect node sharing the GPU pass with a
    tracked one (nests beside bees) must hash the same, or the pass runs twice.
    """
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") == "track.mot":
            return tracker_settings(s.get("config") or {})
    return tracker_settings({})


def _pipeline_tracker(step, steps):
    """Tracking algorithm, read from a downstream MOT node (default BeeTrack).

    Inert on the worker today — there is only one tracker — but the value rides
    the job config so a second algorithm is a worker change, not a builder change.
    """
    for s in downstream_ids(step.get("id"), steps):
        if s.get("block_type") == "track.mot":
            return (s.get("config") or {}).get("tracker", "beetrack") or "beetrack"
    return "beetrack"


    return 0.6


def build_job_config(step, run, context, index):
    """The job config for a GPU step, whichever kind it is.

    Single entry point on purpose: engine._gpu_cache_key and submit_gpu_step both
    call this, so the hashed config is always exactly the one submitted. A
    sampled Detect node produces a completely different job (and result shape)
    from a tracking one, and routing it anywhere else would let the two drift.
    """
    if step.get("block_type") == "detect.objects" and find_artifact("photo", run.steps, index, context):
        return build_photo_detection_config(step, run, context, index)
    if step.get("block_type") == "detect.objects" and is_sampled(step):
        return build_sampled_detection_config(step, run, context, index)
    return build_detect_and_track_config(step, run, context, index)


def build_photo_detection_config(step, run, context, index):
    """Job config for Detect on a photo — the worker's ``detect_photo`` task
    (memory/45): tiled detection, a crop per insect, and species on the crops
    when an Identify species node follows. One node, one class, like sampled
    detection."""
    photo = find_artifact("photo", run.steps, index, context)
    cfg = step.get("config") or {}
    label = detector_label(step)
    if not label:
        return None, "This Detect node needs to know what to detect."
    raw_family = cfg.get("model_family") or "yolo"
    config = {
        "task": "detect_photo",
        "classes": [p.strip() for p in label.split(",") if p.strip()],
        "detector_kind": "sam3" if str(raw_family).lower() == "sam3" else "yolo",
        "confidence_threshold": float(cfg.get("confidence", 0.25) or 0.25),
    }
    species_model = _pipeline_species(step, run.steps)
    if species_model:
        config["identify_species"] = True
        config["species_classifier"] = species_model
    model_err = _resolve_custom_models(cfg, run, config)
    if model_err:
        return None, model_err
    return {"video_id": photo["video_id"], "config": config}, None


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
        # Only values changed from the tracker's defaults, so an untouched
        # BeeTrack node hashes exactly as it did before trackers were tunable.
        **({"tracker_params": settings}
           if (settings := _pipeline_tracker_settings(step, run.steps)) else {}),
    }
    species_model = _pipeline_species(step, run.steps)
    if species_model:
        config["identify_species"] = True
        config["species_classifier"] = species_model
    marker_type = _pipeline_markers(step, run.steps)
    if marker_type:
        config["identify_markers"] = True
        config["marker_type"] = marker_type
    legacy_ref = cfg.get("reference_source")
    if step.get("block_type") == "detect.objects" and legacy_ref:
        # Legacy graphs only: part of the hashed config so two pre-split Detectors
        # differing solely in how they got their reference don't share a result.
        config["reference_source"] = legacy_ref
        if legacy_ref == "detect":
            # The pipeline asked for the bee-hotel nest model. The worker never
            # runs it otherwise (memory/60).
            config["detect_nests"] = True
    model_err = _resolve_custom_models(cfg, run, config)
    if model_err:
        return None, model_err
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
    # Part of the clip: in the config, so it is also in the cache key and a
    # 5-minute result never stands in for the whole clip's.
    for key in ("start_seconds", "end_seconds"):
        if video_out.get(key) is not None:
            config[key] = video_out[key]
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
        # .everything: a Photo Input's photo is a row of the same table.
        video = Video.everything.get(pk=built["video_id"])
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
