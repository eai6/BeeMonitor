"""Re-track one run's clip on the web and point the run at the new tracks
(memory/69).

The GPU's own result is never changed: one GPU result can serve several
batches. The new tracking table, crop votes and crop index are written beside
it under ``retrack/<run>/`` and the run's copy of the result points to them;
``gpu_result`` keeps the GPU's, which every later re-track starts from. Pages
that read a clip through its job get the run's view with ``effective_result``.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

# A crop goes to the new track whose box overlaps its old one at least this much.
CROP_IOU = 0.5


class NotReplayable(Exception):
    """The clip can't be re-tracked on the web (it needs the GPU once)."""


def _job(run):
    from apps.analysis.models import Job

    from ..aggregate import run_job_id

    job_id = run_job_id(run)
    return Job.objects.filter(pk=job_id).select_related("video").first() if job_id else None


def tracking_outputs(run):
    """Step ids of the run's outputs that carry the tracking result."""
    return [sid for sid, out in (run.context or {}).items()
            if isinstance(out, dict) and isinstance(out.get("result"), dict)
            and (out["result"].get("tracking_csv_path") or out.get("gpu_result"))]


def gpu_result(run):
    """The GPU's own result for the run's clip (before any re-track)."""
    for sid in tracking_outputs(run):
        out = run.context[sid]
        return dict(out.get("gpu_result") or out["result"])
    return None


def mot_step(run):
    for s in run.steps or []:
        if s.get("block_type") == "track.mot":
            return s
    return None


def settings_of(run):
    """The BeeTracker settings the run's MOT node asks for now."""
    from .. import executors
    from .replay import tracker_params

    step = mot_step(run)
    return tracker_params(executors.tracker_settings((step or {}).get("config") or {}))


# Where "Update this batch" keeps the MOT settings the GPU tracked with, when
# it changes the run's MOT node (regroup.applied_steps).
GPU_MOT_KEY = "_gpu_mot"
# The MOT settings the web can change by re-tracking (BeeTrack's time windows).
REPLAYABLE = ("beetrack_max_age_seconds", "beetrack_min_hits_seconds",
              # How a lost track searches (memory/71).
              "beetrack_lost_search", "beetrack_stay_in_reference",
              "beetrack_match_distance_multiplier")


def tracked_with(run):
    """The settings the run's current tracks were made with: a re-track's, else
    the GPU's — as recorded when the MOT node was changed, else the node's own
    (unchanged since the GPU ran)."""
    from .. import executors
    from .replay import tracker_params

    for sid in tracking_outputs(run):
        retracked = (run.context[sid].get("result") or {}).get("retracked")
        if retracked:
            return retracked["settings"]
    step = mot_step(run)
    cfg = (step or {}).get("config") or {}
    if isinstance(cfg.get(GPU_MOT_KEY), dict):
        return tracker_params(executors.tracker_settings(cfg[GPU_MOT_KEY]))
    return settings_of(run)


def needs_retrack(run) -> bool:
    """Whether the run's MOT settings differ from those its tracks were made
    with. Only BeeTrack is replayed; a run with no MOT node never is."""
    step = mot_step(run)
    if step is None or ((step.get("config") or {}).get("tracker") or "beetrack") != "beetrack":
        return False
    return settings_of(run) != tracked_with(run)


# ── Reading and writing ─────────────────────────────────────────────────────

def _rows(path):
    from .. import ops

    df = ops._read_csv(path or "")
    if df is None:
        return None
    return df.where(df.notna(), None).to_dict("records")


def _store(base_path, name, rows, fieldnames):
    """Write rows as CSV beside the GPU result: a local file when the result is
    local (tests, development), else the processed bucket."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    folder = base_path.rsplit("/", 1)[0] if "/" in base_path else ""
    if os.path.exists(base_path):
        path = os.path.join(folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            fh.write(buf.getvalue())
        return path
    from config.storage import get_s3_client

    key = f"{folder}/{name}" if folder else name
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as fh:
        fh.write(buf.getvalue())
        local = fh.name
    try:
        get_s3_client().upload_file("processed", key, local)
    finally:
        os.unlink(local)
    return key


# ── Clip and chunks ─────────────────────────────────────────────────────────

def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _frame_range(run, job, fps, detections, params):
    """First and last frame analysed. When the clip's length is unknown the
    replay runs until every track could have been lost."""
    from .. import executors

    first, last = executors._frames_analysed(run, fps, getattr(job, "video", None))
    if last is None:
        frames = [_int(d.get("frame")) for d in detections]
        frames = [f for f in frames if f is not None]
        last = (max(frames) if frames else first) + int(fps * params["max_age_seconds"]) + 2
    return first, last


def _chunks(job):
    """``[(index, start, end_exclusive)]`` for a clip the worker split, else []."""
    chunks = ((job.config or {}).get("chunks") or []) if job else []
    starts = [_int(ch.get("start_frame")) for ch in chunks]
    if not chunks or None in starts:
        return []
    # A chunk ends where the next begins (analysis/views.py plans them so).
    return [(_int(ch.get("i")) if ch.get("i") is not None else i, starts[i],
             starts[i + 1] if i + 1 < len(starts) else None)
            for i, ch in enumerate(chunks)]


def replay_clip(run, job, base, params):
    """New tracking rows for the run's clip: the whole clip, or each chunk
    with its own tracker, then joined as the GPU merge joins them."""
    from .replay import replay

    detections = _rows(base.get("detections_csv_path"))
    if detections is None:
        raise NotReplayable("no saved detections for this clip")
    stats = base.get("summary_stats") or {}
    fps = float(stats.get("video_fps") or 0) or None
    if not fps:
        from .. import ops
        fps = ops.fps_with_source(base, getattr(job, "video", None))[0] or 25.0
    video = getattr(job, "video", None)
    height = getattr(video, "height", None)
    log = _rows(stats.get("update_log_csv")) if stats.get("update_log_csv") else None
    refs = _reference_boxes(run, video) if params.get("stay_in_reference") else None

    chunks = _chunks(job)
    if not chunks:
        first, last = _frame_range(run, job, fps, detections, params)
        return replay(detections, fps, height, params, first, last, log, refs), bool(log)

    from apps.analysis import chunk_stitch
    from apps.analysis.views import _remap_chunk_track_id

    rows, boundaries = [], []
    for i, start, end in chunks:
        part = [d for d in detections if start <= (_int(d.get("frame")) or -1)
                and (end is None or (_int(d.get("frame")) or 0) < end)]
        part_log = [e for e in (log or []) if start <= (_int(e.get("frame")) or -1)
                    and (end is None or (_int(e.get("frame")) or 0) < end)] or None
        last = (end - 1) if end is not None else _frame_range(run, job, fps, part, params)[1]
        for r in replay(part, fps, height, params, start, last, part_log, refs):
            r["track_id"] = _remap_chunk_track_id(r["track_id"], i)
            rows.append(r)
        if end is not None:
            boundaries.append(end)
    chunk_stitch.stitch(rows, boundaries)
    return rows, bool(log)


def _reference_boxes(run, video):
    """The run's references (its Events step's) as boxes in pixels."""
    from .. import executors

    steps = run.steps or []
    for i, step in enumerate(steps):
        if step.get("block_type") != "analyze.events":
            continue
        roi = executors.find_reference(steps, i, run.context or {}, run)
        from .. import ops
        w, h = getattr(video, "width", None), getattr(video, "height", None)
        if not (w and h):
            return None
        return [(r["box"][0] * w, r["box"][1] * h, r["box"][2] * w, r["box"][3] * h)
                for r in ops.roi_references(roi)]
    return None


# ── Crops and species follow the new tracks ────────────────────────────────

def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _boxes(rows):
    """``{(track, frame): box}`` for detected rows."""
    out = {}
    for r in rows:
        if str(r.get("detected", 1)) in ("0", "0.0", "False"):
            continue
        t, f = str(_int(r.get("track_id")) if _int(r.get("track_id")) is not None
                   else r.get("track_id")), _int(r.get("frame"))
        try:
            out[(t, f)] = tuple(float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        except (TypeError, ValueError, KeyError):
            continue
    return out


def crop_mapping(old_rows, new_rows):
    """``{(old track, frame): new track}``: each detected box of the old
    tracks, given to the new track whose box at that frame overlaps it most
    (at least CROP_IOU)."""
    old = _boxes(old_rows)
    new_by_frame = {}
    for (t, f), box in _boxes(new_rows).items():
        new_by_frame.setdefault(f, []).append((t, box))
    out = {}
    for (t, f), box in old.items():
        best, score = None, CROP_IOU
        for nt, nbox in new_by_frame.get(f, []):
            s = _iou(box, nbox)
            if s >= score:
                best, score = nt, s
        if best is not None:
            out[(t, f)] = best
    return out


def _tid(value):
    i = _int(value)
    return str(i) if i is not None else str(value)


def remap_votes(votes, mapping):
    """Crop vote rows with their new track ids; crops no new track took are
    left out."""
    out = []
    for v in votes or []:
        new = mapping.get((_tid(v.get("track_id")), _int(v.get("frame"))))
        if new is not None:
            out.append({**v, "track_id": new})
    return out


def vote_tracks(votes):
    """Each new track's species call from its crops' readings, as the worker
    votes them (identification/track_vote.py)."""
    from beemonitor.identification.species import SpeciesVote
    from beemonitor.identification.track_vote import _summary

    by_track = {}
    for v in votes:
        by_track.setdefault(str(v["track_id"]), []).append(v)
    out = {}
    for tid, rows in by_track.items():
        vote = SpeciesVote()
        for r in rows:
            if r.get("species"):
                vote.add(str(r["species"]), float(r.get("species_confidence") or 0.0))
        out[tid] = {"crops": len(rows), **_summary("taxon", vote, len(rows))}
    return out


def _with_species(rows, by_track):
    for r in rows:
        call = by_track.get(str(r["track_id"]))
        if call and call.get("taxon"):
            r["taxon"] = call["taxon"]
            r["taxon_confidence"] = call.get("taxon_confidence")
        r["taxon_votes"] = (call or {}).get("taxon_votes", 0)
        r["taxon_vote_share"] = (call or {}).get("taxon_vote_share", 0.0)
        r["taxon_crops"] = (call or {}).get("taxon_crops", 0)
    return rows


def remap_manifest(manifest, crops_rows, mapping):
    """The crop index (``{track: [crop keys]}``) for the new tracks."""
    by_key = {}
    for c in crops_rows or []:
        by_key[str(c.get("crop_key"))] = (_tid(c.get("track_id")), _int(c.get("frame")))
    out = {}
    for old_tid, keys in (manifest or {}).items():
        for key in keys or []:
            where = by_key.get(str(key))
            new = mapping.get(where) if where else None
            if new is not None:
                out.setdefault(str(new), []).append(key)
    return out


# ── The whole job ───────────────────────────────────────────────────────────

def retrack_run(run) -> dict:
    """Re-track the run's clip with its MOT node's current settings and point
    the run's outputs at the new tracks. Returns ``{tracks, exact}``. Raises
    NotReplayable when the clip has no saved detections."""
    from django.utils import timezone

    from .replay import TRACKING_COLUMNS

    base = gpu_result(run)
    if not base or not base.get("tracking_csv_path"):
        raise NotReplayable("this run has no tracking result")
    job = _job(run)
    params = settings_of(run)
    new_rows, exact = replay_clip(run, job, base, params)

    old_rows = _rows(base.get("tracking_csv_path")) or []
    mapping = crop_mapping(old_rows, new_rows)
    stats = dict(base.get("summary_stats") or {})
    folder = f"retrack/{run.pk}"
    votes = remap_votes(_rows(stats.get("track_votes_csv")) or [], mapping)
    by_track_votes = vote_tracks(votes) if votes else {}
    new_rows = _with_species(new_rows, by_track_votes)

    tracking_path = _store(base["tracking_csv_path"], f"{folder}/tracking.csv", new_rows,
                           TRACKING_COLUMNS + ["taxon_vote_share", "taxon_crops"])
    if votes:
        stats["track_votes_csv"] = _store(base["tracking_csv_path"], f"{folder}/track_votes.csv",
                                          votes, list(votes[0].keys()))
    identification = dict(stats.get("identification") or {})
    if by_track_votes:
        keep = ("taxon", "taxon_confidence", "taxon_votes", "taxon_crops", "taxon_vote_share")
        identification["by_track"] = {t: {k: v for k, v in c.items() if k in keep}
                                      for t, c in by_track_votes.items()}
    elif identification.get("by_track"):
        identification["by_track"] = {}
    if identification:
        stats["identification"] = identification
    crops_rows = _rows(base.get("crops_csv_path")) or []
    crops_path = base.get("crops_csv_path")
    if crops_rows:
        moved = [{**c, "track_id": mapping[(_tid(c.get("track_id")), _int(c.get("frame")))]}
                 for c in crops_rows
                 if (_tid(c.get("track_id")), _int(c.get("frame"))) in mapping]
        crops_path = _store(base["tracking_csv_path"], f"{folder}/track_crops.csv", moved,
                            list(crops_rows[0].keys()))
    if stats.get("crops_manifest"):
        manifest = remap_manifest(stats["crops_manifest"], crops_rows, mapping)
        stats["crops_manifest"] = manifest
        stats["crops_per_track"] = {t: len(k) for t, k in manifest.items()}

    tracks = len({r["track_id"] for r in new_rows})
    result = {**base, "tracking_csv_path": tracking_path, "summary_stats": stats,
              "crops_csv_path": crops_path,
              "unique_tracks": tracks,
              "retracked": {"settings": params, "at": timezone.now().isoformat(),
                            "exact": exact, "tracks": tracks,
                            "gpu_tracks": base.get("unique_tracks")}}
    context = dict(run.context or {})
    for sid in tracking_outputs(run):
        out = dict(context[sid])
        out.setdefault("gpu_result", out["result"])
        out["result"] = result
        context[sid] = out
    run.context = context
    type(run).objects.filter(pk=run.pk).update(context=context)
    return {"tracks": tracks, "exact": exact}


def effective_result(job, run=None):
    """The clip's result as its run sees it: the run's re-tracked tables when
    it has them, else the job's own JobResult."""
    from .. import executors

    result = getattr(job, "result", None)
    run = run if run is not None else executors._run_for_job(job)
    if run is None or result is None:
        return result
    for sid in tracking_outputs(run):
        mine = run.context[sid].get("result") or {}
        if mine.get("retracked"):
            return _ResultView(result, mine)
    return result


class _ResultView:
    """A JobResult with the run's re-tracked fields over it."""

    _OVER = ("tracking_csv_path", "summary_stats", "unique_tracks", "crops_csv_path")

    def __init__(self, result, over):
        self._result, self._over = result, over

    def __getattr__(self, name):
        if name in self._OVER and name in self._over:
            return self._over[name]
        return getattr(self._result, name)
