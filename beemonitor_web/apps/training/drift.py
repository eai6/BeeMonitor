"""Domain-drift detection (memory/25, P2c).

Embeds a video's frames with DINOv3 (via the SAM 3 endpoint's task="embed") and
scores them against a stored baseline distribution (DriftReference). A high z-score
means this footage has shifted away from what the current detector was trained on —
the trigger to SAM 3-label + fine-tune. All heavy work runs in daemon threads.
"""

import json
import logging
import math
import time
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# z-score above which a video is flagged as domain-drifted (mean cosine distance to
# the reference centroid is > ref_mean + DRIFT_Z * ref_std).
DRIFT_Z = 2.0


def _embed_video(video_blob_path, sample_interval=20, max_frames=200, timeout_s=900):
    """Invoke the SAM 3 endpoint's task='embed' on a video → (embeddings, centroid).

    embeddings: list[list[float]] (L2-normalised per-frame DINOv3 vectors).
    Returns (None, None) on failure/timeout.
    """
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    from django.conf import settings

    endpoint = settings.SAGEMAKER_SAM3_ENDPOINT_NAME
    if not endpoint:
        logger.error("drift: SAGEMAKER_SAM3_ENDPOINT_NAME unset — SAM 3 endpoint not deployed?")
        return None, None
    in_bucket = settings.SAGEMAKER_INPUT_BUCKET
    region = settings.AWS_REGION

    if video_blob_path.startswith("s3://"):
        # embed accepts s3:// or a raw-videos key; pass through.
        pass

    cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2})
    s3 = boto3.client("s3", region_name=region, config=cfg)
    smrt = boto3.client("sagemaker-runtime", region_name=region, config=cfg)

    payload = {
        "task": "embed",
        "video_blob_path": video_blob_path,
        "sample_interval": sample_interval,
        "max_frames": max_frames,
    }
    key = f"drift/{abs(hash(video_blob_path)) % (10**10)}.json"
    s3.put_object(Bucket=in_bucket, Key=key,
                  Body=json.dumps(payload).encode("utf-8"),
                  ContentType="application/json")
    resp = smrt.invoke_endpoint_async(
        EndpointName=endpoint,
        InputLocation=f"s3://{in_bucket}/{key}",
        ContentType="application/json",
        # Default is 15 min; long videos need the platform max (1 h).
        InvocationTimeoutSeconds=3600,
    )
    out = urlparse(resp["OutputLocation"])
    out_bucket, out_key = out.netloc, out.path.lstrip("/")
    fail_key = out_key.replace(".out", ".failure")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(10)
        try:
            body = s3.get_object(Bucket=out_bucket, Key=out_key)["Body"].read()
            result = json.loads(body)
            return result.get("embeddings") or [], result.get("centroid") or []
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("NoSuchKey", "404", "NotFound"):
                raise
            try:
                s3.get_object(Bucket=out_bucket, Key=fail_key)
                logger.error("drift: endpoint failure embedding %s", video_blob_path)
                return None, None
            except ClientError:
                continue  # still running
    logger.warning("drift: embed timed out for %s", video_blob_path)
    return None, None


def _cos_dist_to(centroid, embeddings):
    """Cosine distances (1 - dot, both L2-normalised) of each embedding to centroid."""
    return [1.0 - sum(c * e for c, e in zip(centroid, emb)) for emb in embeddings]


def build_reference(user_id, video_blob_paths, scope="default", note="",
                    sample_interval=20, max_frames=150):
    """Embed baseline videos → save/update the user's DriftReference (centroid + spread)."""
    from django.db import connection

    from .models import DriftReference

    try:
        all_emb = []
        for path in video_blob_paths:
            if not path:
                continue
            embs, _ = _embed_video(path, sample_interval, max_frames)
            if embs:
                all_emb.extend(embs)
        if not all_emb:
            logger.error("drift: no embeddings collected for baseline (user %s)", user_id)
            return

        dim = len(all_emb[0])
        # L2-normalised mean = centroid.
        centroid = [sum(e[i] for e in all_emb) / len(all_emb) for i in range(dim)]
        norm = math.sqrt(sum(c * c for c in centroid)) or 1.0
        centroid = [c / norm for c in centroid]

        dists = _cos_dist_to(centroid, all_emb)
        mean = sum(dists) / len(dists)
        var = sum((d - mean) ** 2 for d in dists) / len(dists)
        std = math.sqrt(var)

        DriftReference.objects.update_or_create(
            user_id=user_id, scope=scope,
            defaults={
                "centroid": centroid, "ref_mean": mean, "ref_std": std,
                "dim": dim, "n_frames": len(all_emb), "note": note,
            },
        )
        logger.info("drift: reference for user %s/%s built from %d frames (mean=%.3f std=%.3f)",
                    user_id, scope, len(all_emb), mean, std)
    except Exception as e:  # noqa: BLE001
        logger.error("drift: build_reference failed (user %s): %s", user_id, e, exc_info=True)
    finally:
        connection.close()


def _maybe_auto_adapt(check, ref):
    """Start an AdaptationRun for a drifted video — unless one is already in
    flight for this scope (dedupe so repeated drift checks don't pile up runs)."""
    from . import orchestrator
    from .models import AdaptationRun

    active = [AdaptationRun.Status.RELABELING, AdaptationRun.Status.TRAINING,
              AdaptationRun.Status.EVALUATING, AdaptationRun.Status.AWAITING]
    if AdaptationRun.objects.filter(
        user_id=check.user_id, scope=ref.scope, status__in=active,
    ).exists():
        logger.info("drift: auto-adapt skipped for scope %s — run already active", ref.scope)
        return
    run = orchestrator.start_run(check.user, [check.video_id], scope=ref.scope)
    if run:
        logger.info("drift: auto-adapt started run %s for drifted video %s (scope %s)",
                    run.pk, check.video_id, ref.scope)


def run_check(check_id):
    """Score one DriftCheck's video against its reference; fill in the result."""
    from django.db import connection

    from .models import DriftCheck

    try:
        check = DriftCheck.objects.select_related("reference", "video").get(pk=check_id)
        ref = check.reference
        if not ref or not ref.centroid:
            check.status = "error"
            check.detail = "no baseline set"
            check.save(update_fields=["status", "detail"])
            return

        embs, _ = _embed_video(check.video.storage_key)
        if not embs:
            check.status = "error"
            check.detail = "embedding failed / no frames"
            check.save(update_fields=["status", "detail"])
            return

        dists = _cos_dist_to(ref.centroid, embs)
        drift_score = sum(dists) / len(dists)
        z = (drift_score - ref.ref_mean) / ref.ref_std if ref.ref_std > 1e-9 else 0.0

        check.drift_score = drift_score
        check.z_score = z
        check.is_drifted = z >= DRIFT_Z
        check.n_frames = len(embs)
        check.status = "done"
        check.detail = ("domain shift — SAM 3 relabel + fine-tune suggested"
                        if check.is_drifted else "within the baseline distribution")
        check.save()
        logger.info("drift: video %s scored drift=%.3f z=%.2f drifted=%s",
                    check.video_id, drift_score, z, check.is_drifted)

        # Close the loop: if the baseline opted into auto-adaptation and this
        # video drifted, kick off an AdaptationRun (deduped per scope). Promotion
        # of the resulting model is still user-approved, so this stays cost-safe.
        if check.is_drifted and ref.auto_adapt:
            _maybe_auto_adapt(check, ref)
    except Exception as e:  # noqa: BLE001
        logger.error("drift: run_check %s failed: %s", check_id, e, exc_info=True)
        try:
            check.status = "error"
            check.detail = str(e)[:255]
            check.save(update_fields=["status", "detail"])
        except Exception:  # noqa: BLE001
            pass
    finally:
        connection.close()
