"""Aggregation for a batch of pipeline runs: the per-run table, the analyzers'
rollups, and the combined CSVs (tracking, detections, tracks, events,
interactions) with provenance columns first.

Trips, visit rates and other higher-level analyses are not computed here
(memory/60): they are reads over the events and interactions tables, left to
the user.
"""

import csv
import io
import logging
from datetime import datetime, timezone as dt_timezone

from apps.pipelines.ops import DEFAULT_FPS, fps_with_source  # noqa: F401  (re-exported)

logger = logging.getLogger(__name__)

# ── Batch sources ─────────────────────────────────────────────────────────────


def run_video_id(run):
    """The video (or photo) pk this run was launched on, from its frozen input
    step — photos are rows of the same table (memory/45)."""
    for s in (run.steps or []):
        if s.get("block_type") in ("input.video", "input.photo"):
            vid = (s.get("config") or {}).get("video_id")
            if str(vid).isdigit():
                return int(vid)
    return None


def run_gpu_result(run):
    """The detect/track step's JobResult summary dict stored in run.context.

    A result carrying tracks wins, because in a detect -> track pipeline the
    detect step's own result would otherwise shadow it. A detection-only result
    is the fallback rather than nothing: this used to recognise tracks and
    events only, so a detect-only batch looked resultless and its Detections
    download was never offered.
    """
    fallback = {}
    for out in (run.context or {}).values():
        result = (out or {}).get("result") or {}
        if result.get("events_csv_path") or result.get("tracking_csv_path"):
            return result
        if not fallback and any(result.get(key) for _k, key, _l, _h in BASE_TABLES):
            fallback = result
    return fallback


# Sort sentinel: clips with no timestamp go last without ever being compared
# against None.
_EPOCH = datetime(1970, 1, 1, tzinfo=dt_timezone.utc)


def collect_sources(runs):
    """Pair each completed run with its video + CSV paths.

    Returns (sources, skipped) where sources are dicts with run/video/result/
    recorded_at/fps and skipped is a list of {video, reason} for runs that
    can't join the aggregate (no timestamp, no result yet, ...).
    """
    from apps.videos.models import Video

    video_ids = [vid for r in runs if (vid := run_video_id(r)) is not None]
    videos = {v.pk: v for v in
              Video.everything.filter(pk__in=video_ids).select_related("device")}

    sources, skipped = [], []
    for run in runs:
        vid = run_video_id(run)
        video = videos.get(vid)
        title = getattr(video, "title", None) or (f"Video {vid}" if vid else "unknown video")
        if video is None:
            skipped.append({"video": title, "reason": "video no longer exists"})
            continue
        if run.status != run.Status.COMPLETED:
            skipped.append({"video": title, "reason": f"run {run.status}"})
            continue
        result = run_gpu_result(run)
        # Any base table is enough to join. This used to demand an events CSV,
        # from when events was the only download — so a detection or tracking
        # pipeline contributed nothing to the Detections and Tracking buttons
        # that were later added beside it, and the page offered no button at all
        # for data it was holding.
        if not any(result.get(k) for _kind, k, _l, _h in BASE_TABLES):
            skipped.append({"video": title, "reason": "no CSV in run output"})
            continue
        # A clip with no timestamp still measured what it measured. It used to be
        # dropped from every download for want of a timeline position, which
        # silently shrank the export; it now joins with an empty timestamp.
        stats = result.get("summary_stats") or {}
        # One resolver for every derived duration in the system — this used to
        # read a "fps" key the backend never writes, so it always fell through
        # to 30 while the run page used the real 25.
        fps, fps_source = fps_with_source(stats, video)
        sources.append({
            "run": run,
            "video": video,
            "title": title,
            "result": result,
            "recorded_at": video.recorded_at,
            "fps": max(float(fps), 1.0),
            "fps_source": fps_source,
        })
    sources.sort(key=lambda s: (s["recorded_at"] is None, s["recorded_at"] or _EPOCH))
    return sources, skipped


# ── CSV plumbing ──────────────────────────────────────────────────────────────


def read_processed_csv(blob_path, use_cache=False):
    """Rows (as dicts) of a CSV in the processed bucket; [] on any failure.

    ``use_cache=True`` memoizes the parsed rows (LocMem, 6 h) — safe only for
    blobs that are immutable once written (a completed job's events CSV) and
    worth it only for SMALL ones; events CSVs are a handful of rows per video,
    tracking CSVs are not cached."""
    if use_cache:
        from django.core.cache import cache

        key = f"prccsv:{blob_path}"
        hit = cache.get(key)
        if hit is not None:
            return hit
    try:
        from config.storage import get_s3_client

        buf = io.BytesIO()
        get_s3_client().download_to_stream("processed", blob_path, buf)
        text = buf.getvalue().decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(text)))
    except Exception as e:
        logger.warning("aggregate: could not read %s: %s", blob_path, e)
        return []  # failures are NOT cached — next call retries
    if use_cache:
        cache.set(key, rows, 6 * 60 * 60)
    return rows


def _frame_number(row):
    for key in ("frame_number", "frame", "frame_num"):
        if row.get(key, "") not in ("", None):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                return None
    return None


# The base tables a batch can hand back — the primitives everything else is
# derived from. Order is the order a person reads them in: what was seen, what
# it was, where it went, who it met.
BASE_TABLES = (
    ("detections", "detections_csv_path", "Detections",
     "Every raw detection, before the tracker associated them into trajectories."),
    ("tracking", "tracking_csv_path", "Tracking",
     "One row per track per frame — the trajectories everything else is read from."),
    ("events", "events_csv_path", "Events",
     "Exits from, entries into and stays in a reference, with their times."),
    ("interactions", "interactions_csv_path", "Interactions",
     "Episodes of two things together, with durations."),
)


def available_downloads(sources, runs=()):
    """The base tables this batch actually produced, as template rows.

    Offering a download that returns nothing is worse than not offering it: the
    user cannot tell a missing pipeline step from a broken button. Only tables
    something actually wrote are listed.

    ``analyzed`` marks the two the pipeline's own analyzers computed. Those
    downloads carry the analyzers' answer rather than the worker's raw file,
    which matters because the two disagree: the worker matches an insect to a
    reference by centroid distance under a flat 50 px and misses anything
    sitting inside a large one.
    """
    # Events and interactions only where the pipeline has the step (memory/60).
    analyzed = set()
    for run in runs:
        for kind, out in analyzer_outputs(run):
            if kind in PRIMITIVE_KINDS:
                analyzed.add(kind)

    from . import tracks

    out = []
    for kind, path_key, label, hint in BASE_TABLES:
        clips = sum(1 for s in sources if (s.get("result") or {}).get(path_key))
        if kind in analyzed:
            out.append({"kind": kind, "label": label,
                        "clips": len(runs) or clips,
                        "analyzed": True,
                        "hint": hint + " From this pipeline's own step, computed "
                                       "from the batch's tracking."})
        elif clips and kind not in PRIMITIVE_KINDS:
            out.append({"kind": kind, "label": label, "hint": hint,
                        "clips": clips, "analyzed": False})
        if kind == "tracking":
            with_tracks = sum(1 for s in sources
                              if (s.get("result") or {}).get("tracking_csv_path")
                              or tracks.by_track_of(s.get("result") or {}))
            if with_tracks:
                out.append({"kind": "tracks", "label": "Tracks", "clips": with_tracks,
                            "analyzed": False,
                            "hint": "One row per track: when it was seen, its species "
                                    "and marker ID (by vote over its crops)."})
    # Trips reviewers accepted or rejected (memory/62 §4b): only once someone
    # has reviewed one, since a suggestion alone is not data.
    from .review import trip_reviews_for
    if any(getattr(r, "pk", None) for r in runs) and trip_reviews_for(
            [r for r in runs if getattr(r, "pk", None)]):
        out.append({"kind": "trips", "label": "Reviewed trips", "clips": len(runs),
                    "analyzed": False,
                    "hint": "Exit → entry pairs a reviewer called: accepted, rejected or "
                            "re-linked. Unreviewed suggestions are never included."})
    return out


# Provenance columns on every combined export. A batch can span devices and
# sites, and a row that cannot say where it came from is not analysable: you
# cannot compare a treatment against a control if the two are indistinguishable
# once concatenated. Written ahead of the source columns so they are the first
# thing a spreadsheet or an R session sees.
PROVENANCE_FIELDS = ["video_title", "clip_start", "device_id", "device_name",
                     "site_name", "location"]


def _provenance(src):
    """Where one clip's rows came from: which clip, which device, which site."""
    video = src.get("video")
    device = getattr(video, "device", None)
    from apps.videos import timing

    return {
        "video_title": src["title"],
        # The clip's recorded start; blank when it is unknown (memory/60), so
        # the reader can fill it in and add clip_sec themselves.
        "clip_start": timing.iso(timing.clip_start(video)) if video else "",
        "device_id": getattr(device, "id", "") or "",
        "device_name": getattr(device, "name", "") or "",
        # The clip's own site wins: a device can be moved between sites, and the
        # clip records where it actually was.
        "site_name": getattr(video, "site_name", "") or "",
        "location": getattr(device, "location", "") or "",
    }


# Primitive tables the pipeline's own analyzers computed, keyed by the
# table_kind they tag themselves with.
PRIMITIVE_KINDS = {"events": "events", "interactions": "interactions"}


def _primitive_fieldnames(kind, rows):
    from . import primitives, tracks

    if kind == "events":
        return PROVENANCE_FIELDS + list(primitives.EVENT_FIELDS) + tracks.IDENTITY_FIELDS
    return (PROVENANCE_FIELDS + list(primitives.INTERACTION_FIELDS) + tracks.IDENTITY_FIELDS
            + ["with_species", "with_marker_id"])


def primitive_csv(runs, kind):
    """(fieldnames, rows) for the events/interactions the ANALYZERS computed.

    Not the worker's own CSV. That one matches an insect to a reference by
    centroid-to-centroid distance under a flat 50 px, which throws the
    reference's size away: a bee resting inside a 400 px flower sits ~200 px
    from its centre and is never recorded. Exporting it meant the download
    disagreed with the annotated video the user was looking at.

    The analyzers ask containment instead, and their rows live in run.context.
    Returns (None, []) when no run in the batch produced this primitive — the
    caller then falls back to the worker's file rather than offering nothing.
    """
    from apps.videos.models import Video

    video_ids = [vid for r in runs if (vid := run_video_id(r)) is not None]
    videos = {v.pk: v for v in
              Video.everything.filter(pk__in=video_ids).select_related("device")}

    all_rows = []
    for run in runs:
        video = videos.get(run_video_id(run))
        if video is None:
            continue
        # Only what the run's own steps saved: a pipeline with no step for this
        # table contributes nothing (memory/60).
        outputs = [out for k, out in analyzer_outputs(run) if k == kind]
        if not outputs:
            continue
        result = run_gpu_result(run)
        fps = max(fps_with_source(result, video)[0], 1.0)
        from apps.videos import timing

        from . import tracks
        start = timing.clip_start(video)
        provenance = _provenance({"video": video, "title": video.title or f"Video {video.pk}"})
        from .review import added_rows, event_summaries
        reviews = event_summaries(run) if kind == "events" else None
        rows = [r for out in outputs for r in (out.get("rows") or [])]
        for out in [{"rows": rows}]:
            for row in tracks.primitive_with_identity(
                    kind, out.get("rows") or [], tracks.by_track_of(result), fps, start,
                    reviews=reviews, added=added_rows(run) if kind == "events" else None):
                all_rows.append({**provenance, **row})

    if not all_rows:
        return None, []
    return _primitive_fieldnames(kind, all_rows), all_rows


def combined_csv(sources, path_key):
    """Every clip's tracking or detections rows, as the columns people download
    (memory/60): times instead of bare frames, the worker's internal columns
    translated, provenance first.

    Returns (fieldnames, rows); fieldnames is None when nothing was readable.
    """
    from apps.videos import timing

    from . import tracks

    tracking = path_key == "tracking_csv_path"
    fields = tracks.TRACKING_FIELDS if tracking else tracks.DETECTION_FIELDS
    all_rows, read_any = [], False
    for src in sources:
        blob = src["result"].get(path_key)
        if not blob:
            continue
        rows = read_processed_csv(blob)
        if not rows:
            continue
        read_any = True
        start = timing.clip_start(src["video"]) if src.get("video") else None
        if tracking:
            rows = tracks.tracking_export(rows, tracks.by_track_of(src["result"]),
                                          src["fps"], start)
        else:
            rows = tracks.detections_export(rows, src["fps"], start)
        provenance = _provenance(src)
        all_rows.extend({**provenance, **row} for row in rows)
    if not read_any:
        return None, []
    return PROVENANCE_FIELDS + list(fields), all_rows


def run_job_id(run):
    """The GPU job pk this run submitted, from its frozen context."""
    for out in (run.context or {}).values():
        job_id = (out or {}).get("job_id")
        if job_id:
            return job_id
    return None


def run_error(run):
    """The message that explains a failed run.

    Prefers the step's own error (the GPU job's text, which names the real
    cause) over the run-level summary, which is usually just "a step failed".
    """
    if run.status != run.Status.FAILED:
        return ""
    candidates = []
    for out in (run.context or {}).values():
        error = (out or {}).get("error")
        if error:
            candidates.append(error)
    # "Upstream step failed." is a consequence, never the cause — keep it only
    # if nothing else explains the run.
    real = [c for c in candidates if "Upstream step failed" not in c]
    return (real or candidates or [run.error_message or ""])[0]


def batch_rows(runs):
    """One row per run: clip, outcome, what it produced, what it cost.

    The batch page could only say "failed"; a row now carries the reason and
    the numbers, so a batch reads without opening anything.
    """
    from apps.analysis.models import Job, JobResult
    from apps.videos.models import Video

    video_ids = [v for r in runs if (v := run_video_id(r)) is not None]
    videos = {v.pk: v for v in Video.everything.filter(pk__in=video_ids).select_related("device")}
    job_ids = [j for r in runs if (j := run_job_id(r)) is not None]
    jobs = {j.pk: j for j in Job.objects.filter(pk__in=job_ids)}
    results = {r.job_id: r for r in JobResult.objects.filter(job_id__in=job_ids)}

    from .review import batch_review_counts
    reviews = batch_review_counts(runs)

    rows = []
    for run in runs:
        video = videos.get(run_video_id(run))
        job = jobs.get(run_job_id(run))
        result = results.get(job.pk) if job else None
        rows.append({
            "run": run,
            "video": video,
            "job": job,
            "result": result,
            # What the analyzers measured, so the row agrees with the clip's own
            # page and with this batch's CSV downloads.
            "primitives": primitive_counts(run),
            # Reviewers' calls on its events (memory/62); None = no Events step.
            "review": reviews.get(run.pk),
            "status": run.status,
            "error": run_error(run),
            # Stopped by its owner (memory/49): shown apart from failures.
            "cancelled": run.status == run.Status.FAILED and "Cancelled by user" in run_error(run),
            "when": video.recorded_at if video else None,
            # GPU time is spent whether or not the run produced anything — a
            # failure that burned an hour is worth seeing next to one that did
            # not. Reported as time rather than money: seconds are a fact about
            # the work; a price is a claim about a rate card that drifts.
            "gpu_seconds": float(job.execution_seconds) if job and job.execution_seconds else 0.0,
        })
    rows.sort(key=lambda r: (r["when"] is None, r["when"]), reverse=True)
    return rows


def batch_devices(rows):
    """The distinct devices this batch's clips came from, in row order.

    The header used to print ``rows.0.video.device.name`` — the FIRST clip's
    device, labelled as if it were the batch's. A batch is a set of clips, and
    nothing stops those clips coming from several units, so that header was a
    guess that happened to be right whenever a batch was single-device and
    silently wrong when it was not. Which unit a result came from is the first
    thing you need to trust it.
    """
    seen, out = set(), []
    for row in rows:
        device = getattr(row.get("video"), "device", None)
        name = (getattr(device, "name", "") or "").strip()
        key = getattr(device, "pk", None) or name
        if device is None or not name or key in seen:
            continue
        seen.add(key)
        out.append({"pk": device.pk, "name": name,
                    "location": (device.location or "").strip()})
    return out


def batch_days(rows):
    """The local days this batch's clips were recorded on, newest first: the
    options of the batch page's Day filter."""
    from django.utils import timezone

    return sorted({timezone.localtime(r["when"]).date() for r in rows if r.get("when")},
                  reverse=True)


def batch_summary(rows):
    """Counts and money for the whole batch, including what failures cost."""
    completed = [r for r in rows if r["status"] == "completed"]
    cancelled = [r for r in rows if r["status"] == "failed" and r.get("cancelled")]
    failed = [r for r in rows if r["status"] == "failed" and not r.get("cancelled")]
    pct = (lambda n: round(100 * n / len(rows))) if rows else (lambda n: 0)
    return {
        "total": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "cancelled": len(cancelled),
        "running": len(rows) - len(completed) - len(failed) - len(cancelled),
        "pct_ok": pct(len(completed)),
        "pct_failed": pct(len(failed)),
        "pct_cancelled": pct(len(cancelled)),
        "gpu_seconds": round(sum(r["gpu_seconds"] for r in rows), 1),
        "gpu_seconds_failed": round(sum(r["gpu_seconds"] for r in failed), 1),
        "failed_video_ids": [r["video"].pk for r in failed if r["video"]],
        "all_video_ids": [r["video"].pk for r in rows if r["video"]],
    }


# ── Analyzer-shaped results ──────────────────────────────────────────────────
# The batch page only ever aggregated foraging trips: it read events CSVs and
# rendered trips, entries/exits and nest chips whatever the pipeline computed. Run
# a Visitation pipeline and you got a page about a question you never asked.
#
# Each analyzer already tags its output with `table_kind` in run.context. Nothing
# read it at batch level. These functions merge each kind across the batch's runs,
# so the page can render what was actually computed.

def analyzer_outputs(run):
    """Every analyzer output in one run, as ``(kind, output)``.

    Most analyzers tag themselves with ``table_kind``. Foraging trips does not —
    it predates the table analyzers and returns ``artifact: "events"`` — so it is
    mapped here rather than left undetectable, which would make a trips pipeline
    look like it ran no analyzer at all.
    """
    out = []
    for value in (run.context or {}).values():
        if not isinstance(value, dict):
            continue
        kind = value.get("table_kind")
        if kind:
            out.append((kind, value))
    return out


# The key each primitive's analyzer reports its own total under. Reading the
# analyzer's count rather than len(rows) keeps this free: the number is already
# in run.context, saved when the analyzer ran.
_PRIMITIVE_COUNT_KEYS = {"events": "event_count", "interactions": "interaction_count"}


def primitive_counts(run):
    """What this run's own analyzers measured, per primitive.

    The batch table printed ``JobResult.total_events`` and
    ``JobResult.interaction_count`` — the WORKER's counters — while the table
    and the CSV for the same clip carried the analyzers' computed rows. The
    worker matches an insect to a reference by centroid distance under a flat
    50 px, so the two legitimately disagree, and the batch said one thing while
    every download said another.

    ``None``, not zero, when this pipeline ran no analyzer for that primitive.
    Zero is a measurement; printing it for a question the batch never asked is
    the column of zeros this page already refuses to render elsewhere.
    """
    counts = dict.fromkeys(_PRIMITIVE_COUNT_KEYS, None)
    for kind, out in analyzer_outputs(run):
        key = _PRIMITIVE_COUNT_KEYS.get(kind)
        if key is None:
            continue
        total = out.get(key)
        if total is None:
            total = len(out.get("rows") or [])
        counts[kind] = (counts[kind] or 0) + int(total)
    return counts


def _merge_per_reference(bucket, rows, count_key):
    """Add one run's per-reference rows into the batch-wide tally.

    Keyed on the reference id, which is stable within a layout — that is what
    makes the same tube comparable across clips. Labels come along so the page
    never has to invent one.
    """
    from .ops import _reference_label
    from .primitives import reference_id

    for row in rows or []:
        ref_id = str(reference_id(row.get("id", "")) or "")
        if not ref_id:
            continue
        entry = bucket.setdefault(ref_id, {
            "id": ref_id, "label": _reference_label(ref_id),
            count_key: 0, "visitors": 0, "partners": 0, "dwell_sec": 0.0,
            "duration_sec": 0.0, "clips": 0,
        })
        entry[count_key] += row.get(count_key, 0) or 0
        entry["visitors"] += row.get("visitors", 0) or 0
        entry["partners"] += row.get("partners", 0) or 0
        entry["dwell_sec"] += float(row.get("dwell_sec") or 0)
        entry["duration_sec"] += float(row.get("duration_sec") or 0)
        if row.get(count_key):
            entry["clips"] += 1
    return bucket


def aggregate_visitation(outputs):
    """Visits across the batch, and per reference.

    ``unique_visitors`` is summed rather than deduplicated: track ids are only
    unique within one clip, so the same bee in two clips is two visitors and
    there is no way to know otherwise. Stated on the page rather than hidden.
    """
    per_ref, totals = {}, {"unique_visitors": 0, "total_visits": 0, "total_dwell_sec": 0.0}
    for out in outputs:
        totals["unique_visitors"] += out.get("unique_visitors", 0) or 0
        totals["total_visits"] += out.get("total_visits", 0) or 0
        totals["total_dwell_sec"] += float(out.get("total_dwell_sec") or 0)
        _merge_per_reference(per_ref, out.get("per_reference"), "visits")

    rows = sorted(per_ref.values(), key=lambda r: (-r["visits"], r["id"]))
    for r in rows:
        r["dwell_sec"] = round(r["dwell_sec"], 1)
    totals["total_dwell_sec"] = round(totals["total_dwell_sec"], 1)
    totals["per_reference"] = rows
    totals["clips"] = len(outputs)
    return totals


def aggregate_interaction(outputs):
    per_ref, totals = {}, {"interaction_count": 0, "organism_organism": 0,
                           "organism_reference": 0, "total_duration_sec": 0.0}
    for out in outputs:
        for key in ("interaction_count", "organism_organism", "organism_reference"):
            totals[key] += out.get(key, 0) or 0
        totals["total_duration_sec"] += float(out.get("total_duration_sec") or 0)
        _merge_per_reference(per_ref, out.get("per_reference"), "interactions")

    rows = sorted(per_ref.values(), key=lambda r: (-r["interactions"], r["id"]))
    for r in rows:
        r["duration_sec"] = round(r["duration_sec"], 1)
    totals["total_duration_sec"] = round(totals["total_duration_sec"], 1)
    totals["per_reference"] = rows
    totals["clips"] = len(outputs)
    return totals


def aggregate_detection_count(outputs):
    totals = {"total": 0, "distinct": 0, "clips": len(outputs), "with_any": 0}
    for out in outputs:
        rows = out.get("rows") or []
        total = out.get("total") or sum(r.get("count", 0) or 0 for r in rows)
        distinct = out.get("distinct") or out.get("distinct_objects") or 0
        totals["total"] += total or 0
        totals["distinct"] += distinct or 0
        if total:
            totals["with_any"] += 1
    return totals


# Colony activity is deliberately absent: its computation stays, but it does not
# get a section of its own — a timeline belongs inside whichever analyzer ran.
def aggregate_events(outputs):
    """Boundary crossings across the batch, and per target.

    ``subjects`` is summed rather than deduplicated for the same reason
    visitors are: track ids are unique only within one clip. Stated on the page
    rather than hidden.
    """
    from .ops import _reference_label
    from .primitives import reference_id

    per_ref, totals = {}, {"event_count": 0, "enter_count": 0,
                           "exit_count": 0, "stay_count": 0, "subjects": 0}
    for out in outputs:
        for key in ("event_count", "enter_count", "exit_count", "stay_count", "subjects"):
            totals[key] += out.get(key, 0) or 0
        for row in out.get("per_reference") or []:
            ref_id = str(reference_id(row.get("id", "")) or "")
            if not ref_id:
                continue
            entry = per_ref.setdefault(ref_id, {
                "id": ref_id, "label": _reference_label(ref_id),
                "kind": "reference" if row.get("kind") in (None, "nest") else row["kind"],
                "events": 0, "enter": 0, "exit": 0, "stay": 0, "subjects": 0, "clips": 0,
            })
            for key in ("events", "enter", "exit", "stay", "subjects"):
                entry[key] += row.get(key, 0) or 0
            if row.get("events"):
                entry["clips"] += 1

    totals["per_reference"] = sorted(
        per_ref.values(), key=lambda r: (-r["events"], r["id"]))
    totals["clips"] = len(outputs)
    return totals


def aggregate_interactions(outputs):
    """Interaction episodes across the batch, and per reference.

    The insect-to-reference count *is* the batch's visitation count — a visit
    is an insect interacting with a reference — so the page reports it from
    here rather than from a separate analyzer that could disagree.
    """
    per_ref, totals = {}, {"interaction_count": 0, "organism_organism": 0,
                           "organism_reference": 0, "total_duration_sec": 0.0}
    for out in outputs:
        for key in ("interaction_count", "organism_organism", "organism_reference"):
            totals[key] += out.get(key, 0) or 0
        totals["total_duration_sec"] += float(out.get("total_duration_sec") or 0)
        _merge_per_reference(per_ref, out.get("per_reference"), "interactions")

    rows = sorted(per_ref.values(), key=lambda r: (-r["interactions"], r["id"]))
    for r in rows:
        r["duration_sec"] = round(r["duration_sec"], 1)
    totals["total_duration_sec"] = round(totals["total_duration_sec"], 1)
    totals["per_reference"] = rows
    totals["clips"] = len(outputs)
    return totals


AGGREGATORS = {
    # The primitives.
    "events": aggregate_events,
    "interactions": aggregate_interactions,
    # Retired analyzers. `visitation` deliberately has NO aggregator: a visit
    # is an insect-to-reference interaction, so its panel was a second, worse
    # answer to a question the Interactions panel already answers — and on a
    # pipeline with no references it rendered four zeros and an apology.
    # Historical runs still list the analyzer they used; they just don't get a
    # panel of their own.
    "interaction": aggregate_interaction,
    "detection_count": aggregate_detection_count,
}

KIND_LABELS = {
    "events": "Events",
    "interactions": "Interactions",
    "visitation": "Visitation",
    "interaction": "Interactions",
    "detection_count": "Detection count",
}


def reference_summary(runs, rows):
    """What geometry the analyzers had to work with — the batch's blind spot.

    Whether a reference reached the analyzer decided everything downstream and
    was visible nowhere: a batch could report zero visits on footage full of
    them and look no different from a batch where nothing happened.

    Reads only what is already loaded — the graph's own geometry and the
    worker's detected boxes — so it costs no S3 and no probe. ``unplaceable``
    is the case that actually bit: the worker found reference boxes, but they
    are in pixels and the clip's frame size was never measured, so they cannot
    be placed against the tracks.
    """
    from . import executors, ops

    if not runs:
        return {"count": 0, "source": "none", "unplaceable": 0}

    run = next((r for r in runs if r.status == r.Status.COMPLETED), runs[0])
    steps = list(run.steps or [])
    index = next((i for i in range(len(steps) - 1, -1, -1)
                  if str(steps[i].get("block_type", "")).startswith("analyze.")),
                 max(len(steps) - 1, 0))
    try:
        roi = executors.find_reference(steps, index, run.context or {}, run)
    except Exception:                      # a half-built graph must not 500 the page
        logger.exception("reference summary: could not resolve the graph reference")
        roi = {}
    refs = ops.roi_references(roi)
    if refs:
        # Where the layout came from: this clip, its site or its unit (memory/52).
        return {"count": len(refs), "source": "graph", "unplaceable": 0,
                "label": (roi or {}).get("source_label", ""),
                "overlaps": ops.overlapping_references(refs)}
    # References come only from the pipeline (memory/60): none is none.
    return {"count": 0, "source": "none", "unplaceable": 0}


def coverage_of(outputs, attempted):
    """How much of the batch a total actually speaks for.

    Every aggregate used to report ``clips = len(outputs)`` — the clips that
    *succeeded*. A day where 9 of 12 runs failed rendered as a clean 3-clip day
    with nothing to say the other 75% of the footage is missing. A total
    without a denominator is not a usable number.

    ``assumed_fps`` counts the clips whose durations rest on a guessed frame
    rate rather than a measured one (see ``ops.fps_with_source``).
    """
    analysed = len(outputs)
    assumed = sum(1 for o in outputs if (o or {}).get("fps_source") == "assumed")
    return {
        "attempted": attempted,
        "analysed": analysed,
        "missing": max(attempted - analysed, 0),
        "pct": round(100 * analysed / attempted) if attempted else 0,
        "complete": analysed >= attempted,
        "assumed_fps": assumed,
    }


def analyzer_results(runs):
    """``[{kind, label, summary, coverage}]`` for every analyzer this batch ran.

    Ordered by how many runs produced each, so the pipeline's main analyzer
    leads when a graph has more than one.
    """
    by_kind = {}
    for run in runs:
        for kind, output in analyzer_outputs(run):
            by_kind.setdefault(kind, []).append(output)

    attempted = len(runs)
    results = []
    for kind, outputs in by_kind.items():
        aggregator = AGGREGATORS.get(kind)
        entry = {
            "kind": kind,
            "label": KIND_LABELS.get(kind, kind.replace("_", " ").title()),
            # foraging_trips is rendered by the existing cross-video machinery;
            # colony_activity deliberately has no section. Both still register
            # so the page knows which analyzers ran.
            "summary": aggregator(outputs) if aggregator else None,
            "clips": len(outputs),
            "coverage": coverage_of(outputs, attempted),
        }
        if not aggregator:
            entry["label"] = KIND_LABELS.get(kind, kind)
        results.append(entry)
    results.sort(key=lambda r: -r["clips"])
    return results



# ── Photo batches (memory/45) ─────────────────────────────────────────────────

def run_photo(run):
    """The photo result a photo run's Detect step wrote (memory/45), or None."""
    for step_out in (run.context or {}).values():
        if isinstance(step_out, dict):
            photo = ((step_out.get("result") or {}).get("summary_stats") or {}).get("photo")
            if photo:
                return photo
    return None


def photo_rows(runs):
    """One row per detected insect across a batch of photo runs:
    photo id, taken at, insect, class, detector confidence, species, its
    confidence. Empty when the batch is not photo runs."""
    from apps.videos.models import Video

    out = []
    per_run = []
    for run in runs:
        photo = run_photo(run)
        if photo is None:
            continue
        pid = next((int((s.get("config") or {}).get("video_id") or 0) for s in run.steps or []
                    if s.get("block_type") == "input.photo"), 0)
        per_run.append((run, pid, photo))
    videos = {v.pk: v for v in Video.everything.filter(pk__in=[p for _r, p, _ph in per_run])}
    for run, pid, photo in per_run:
        v = videos.get(pid)
        taken = v.recorded_at.isoformat() if v and v.recorded_at else ""
        for d in photo.get("detections") or []:
            out.append({"photo_id": pid, "taken_at": taken, "run_id": str(run.pk),
                        "insect": d.get("id"), "class": d.get("class"),
                        "confidence": d.get("confidence"), "species": d.get("species") or "",
                        "species_confidence": d.get("species_confidence"),
                        "x": d.get("x"), "y": d.get("y"), "w": d.get("w"), "h": d.get("h")})
        if not photo.get("detections"):
            out.append({"photo_id": pid, "taken_at": taken, "run_id": str(run.pk), "insect": None})
    return out


def photo_counts(run):
    """Insects detected and species named in one photo run, or None for a clip
    run — the photo batch's row numbers."""
    photo = run_photo(run)
    if photo is None:
        return None
    named = set()
    for d in photo.get("detections") or []:
        if d.get("species"):
            named.add(d["species"])
    return {"insects": len(photo.get("detections") or []), "named": len(named)}