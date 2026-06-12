"""BioCLIP perception pipeline (Phase 1).

When an ``ActivityFrame`` lands, classify the mover crop with BioCLIP on a
SageMaker Serverless (CPU) endpoint, record a ``Detection``, and once all of an
activity's frames are in, aggregate them into an ``Observation`` + stamp the
``Activity`` with its best taxon/status.

Runs off the request path on a bounded thread pool (mirrors
``apps.analysis.spawn_gpu_job_async``) so ``/devices/frames`` stays fast, and is a
**no-op when ``SAGEMAKER_BIOCLIP_ENDPOINT_NAME`` is unset** — Phase 0 (frames just
ingest + display) keeps working until the endpoint is deployed.

The endpoint returns ranked Tree-of-Life predictions, e.g.::

    [{"score": 0.82,
      "common_name": "common eastern bumble bee",
      "ranks": {"kingdom": "Animalia", ..., "genus": "Bombus",
                "species": "Bombus impatiens"}},
     ...]
"""

from __future__ import annotations

import io
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.db import connection

from .models import Activity, ActivityFrame, Detection, Observation, Taxon
from .priors import region_taxa

logger = logging.getLogger(__name__)

# Most-general -> most-specific. Matches Taxon.Rank and the endpoint's `ranks`.
RANK_ORDER = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

# Built lazily on first use so processes that never classify (admin, migrate,
# unrelated management commands, the no-endpoint Phase-0 case) don't spin up
# idle worker threads at import.
_POOL = None
_POOL_LOCK = threading.Lock()


def _pool() -> ThreadPoolExecutor:
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = ThreadPoolExecutor(
                    max_workers=getattr(settings, "MONITOR_CLASSIFY_MAX_WORKERS", 3),
                    thread_name_prefix="bioclip",
                )
    return _POOL


def enabled() -> bool:
    """True when a BioCLIP endpoint is configured (else the pipeline no-ops)."""
    return bool(getattr(settings, "SAGEMAKER_BIOCLIP_ENDPOINT_NAME", ""))


def classify_frame_async(frame_id: int) -> None:
    """Queue classification of one frame on the bounded pool (non-blocking)."""
    if not enabled():
        return
    _pool().submit(classify_frame, frame_id)


# ---------------------------------------------------------------------------
# SageMaker invocation
# ---------------------------------------------------------------------------

def _sagemaker_runtime():
    import boto3
    from botocore.config import Config
    return boto3.client(
        "sagemaker-runtime",
        region_name=getattr(settings, "AWS_REGION", "us-east-1"),
        # Serverless cold starts can take a while; allow for it but still bound.
        config=Config(connect_timeout=10, read_timeout=120, retries={"max_attempts": 2}),
    )


def _read_crop_bytes(storage_key: str) -> bytes:
    from config.storage import get_s3_client
    buf = io.BytesIO()
    get_s3_client().download_to_stream("raw-videos", storage_key, buf)
    return buf.getvalue()


def _invoke_bioclip(jpeg: bytes, candidate_taxa=None) -> list:
    """Send one crop to the endpoint; return its ranked predictions (a list).

    With ``candidate_taxa`` the request constrains BioCLIP to those labels (the
    location prior); without it, classification is unconstrained Tree-of-Life.
    The body is JSON so the candidate list rides alongside the base64 image (the
    container also still accepts a bare image/jpeg body for the unconstrained case).
    """
    import base64
    endpoint = settings.SAGEMAKER_BIOCLIP_ENDPOINT_NAME
    body = {"image_b64": base64.b64encode(jpeg).decode("ascii")}
    if candidate_taxa:
        body["candidate_taxa"] = list(candidate_taxa)
        body["rank"] = "species"
    resp = _sagemaker_runtime().invoke_endpoint(
        EndpointName=endpoint,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(body).encode("utf-8"),
    )
    data = json.loads(resp["Body"].read().decode("utf-8"))
    # Accept either a bare list or {"predictions": [...]}.
    preds = data.get("predictions", data) if isinstance(data, dict) else data
    return preds if isinstance(preds, list) else []


# ---------------------------------------------------------------------------
# Taxon resolution + persistence
# ---------------------------------------------------------------------------

def _resolve_taxon(ranks: dict) -> "Taxon | None":
    """Get-or-create the Taxon chain kingdom->...->species, return the deepest.

    Wires each node's ``parent`` so the tree is queryable. Missing/blank ranks
    are skipped (e.g. an order-level-only prediction stops there).
    """
    if not isinstance(ranks, dict):
        return None
    parent = None
    deepest = None
    for rank in RANK_ORDER:
        name = (ranks.get(rank) or "").strip()
        if not name:
            continue
        taxon, _ = Taxon.objects.get_or_create(
            rank=rank, name=name, defaults={"parent": parent},
        )
        if parent is not None and taxon.parent_id is None:
            taxon.parent = parent
            taxon.save(update_fields=["parent"])
        parent = taxon
        deepest = taxon
    return deepest


def classify_frame(frame_id: int) -> None:
    """Worker: classify one frame, write a Detection, maybe aggregate the activity.

    Closes the DB connection across the network call (the SageMaker round-trip can
    be slow on a serverless cold start) so we never pin a pooled connection.
    """
    try:
        frame = (ActivityFrame.objects
                 .select_related("activity").get(pk=frame_id))
        storage_key = frame.storage_key
        activity = frame.activity
        activity_id = activity.id
        lat, lon = activity.lat, activity.lon
        month = activity.started_at.month if activity.started_at else None
    except ActivityFrame.DoesNotExist:
        connection.close()
        return

    connection.close()  # release before the slow S3 + SageMaker calls

    err = None
    preds = []
    try:
        jpeg = _read_crop_bytes(storage_key)
        topk = int(getattr(settings, "MONITOR_BIOCLIP_TOPK", 5))
        candidates = []
        if getattr(settings, "MONITOR_USE_LOCATION_PRIORS", True):
            candidates = region_taxa(lat, lon, month)
        preds = _invoke_bioclip(jpeg, candidates)
        # If the location-constrained answer is weak, fall back to an unconstrained
        # Tree-of-Life pass and keep whichever is more confident — so a true species
        # not yet recorded near this site isn't forced into a wrong local neighbour.
        if candidates:
            floor = float(getattr(settings, "MONITOR_BIOCLIP_GENUS_FALLBACK", 0.15))
            top_score = _score(preds[0]) if preds else None
            if top_score is None or top_score < floor:
                unconstrained = _invoke_bioclip(jpeg, None)
                if unconstrained and (top_score is None
                                      or (_score(unconstrained[0]) or 0.0) > top_score):
                    preds = unconstrained
        preds = preds[:topk]
    except Exception as e:  # noqa: BLE001 - any failure must not crash the worker
        logger.exception("bioclip: frame %s classify failed", frame_id)
        err = str(e)

    try:
        if err is not None:
            # Record the failure AS a detection (taxon=None) so the activity can
            # still complete from its other frames — one transient error must not
            # strand the whole activity. update_or_create keeps re-runs idempotent.
            Detection.objects.update_or_create(
                frame_id=frame_id, model=Detection.Model.BIOCLIP,
                defaults={"taxon": None, "confidence": None, "raw": {"error": err}},
            )
        else:
            top = preds[0] if preds else None
            taxon = _resolve_taxon(top.get("ranks", {})) if top else None
            Detection.objects.update_or_create(
                frame_id=frame_id, model=Detection.Model.BIOCLIP,
                defaults={"taxon": taxon, "confidence": _score(top),
                          "raw": {"predictions": preds}},
            )
        _maybe_aggregate(activity_id)
    finally:
        connection.close()


def _score(pred) -> "float | None":
    if not isinstance(pred, dict):
        return None
    try:
        return float(pred.get("score"))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-activity aggregation
# ---------------------------------------------------------------------------

def _maybe_aggregate(activity_id: int) -> None:
    """Once every frame of the activity has a BioCLIP detection, roll them up.

    Consensus = the taxon with the highest mean score across frames; the activity
    is marked ``no_detection`` if even the best is below the confidence floor.
    Individual count is 1 for now (one mover per activity — refined later).
    """
    activity = Activity.objects.filter(pk=activity_id).first()
    if activity is None:
        return
    frames = list(activity.frames.all())
    if not frames:
        return
    dets = list(Detection.objects.filter(
        frame__activity_id=activity_id, model=Detection.Model.BIOCLIP))
    # Wait until every frame has been classified.
    if len({d.frame_id for d in dets}) < len(frames):
        return

    floor = float(getattr(settings, "MONITOR_BIOCLIP_MIN_CONFIDENCE", 0.2))
    scored = [d for d in dets if d.taxon_id is not None and d.confidence is not None]

    if not scored:
        # No usable ID. Distinguish "the model ran but found nothing" from
        # "every frame errored" so a transient outage isn't silently NO_DETECTION.
        had_error = any((d.raw or {}).get("error") for d in dets)
        activity.status = (Activity.Status.FAILED if had_error
                           else Activity.Status.NO_DETECTION)
        activity.save(update_fields=["status"])
        return

    # Mean score per taxon across the activity's frames.
    by_taxon: dict = {}
    for d in scored:
        by_taxon.setdefault(d.taxon_id, []).append(d)
    stats = {tid: (sum(d.confidence for d in ds) / len(ds), ds)
             for tid, ds in by_taxon.items()}
    best_taxon_id = max(stats, key=lambda tid: stats[tid][0])

    # Taxa clearing the floor each become an Observation (co-occurrence). If none
    # clear it, record the best as the headline but mark the activity NO_DETECTION.
    confident = {tid: s for tid, s in stats.items() if s[0] >= floor}
    if not confident:
        activity.status = Activity.Status.NO_DETECTION
        activity.best_taxon_id = best_taxon_id
        activity.best_confidence = stats[best_taxon_id][0]
        activity.save(update_fields=["status", "best_taxon", "best_confidence"])
        return

    # Prune observations from a prior run whose taxon is no longer confident.
    activity.observations.exclude(taxon_id__in=confident.keys()).delete()
    for tid, (mean_conf, ds) in confident.items():
        rep = max(ds, key=lambda d: d.confidence)
        Observation.objects.update_or_create(
            activity=activity, taxon_id=tid,
            defaults={"confidence": mean_conf, "individual_count": 1,
                      "representative_frame_id": rep.frame_id,
                      "status": Observation.Status.AUTO},
        )

    best_taxon_id = max(confident, key=lambda tid: confident[tid][0])
    activity.best_taxon_id = best_taxon_id
    activity.best_confidence = confident[best_taxon_id][0]
    activity.individual_count = 1  # per-taxon headline count (true counting deferred)
    activity.status = Activity.Status.ANALYZED
    activity.save(update_fields=["best_taxon", "best_confidence",
                                 "individual_count", "status"])
