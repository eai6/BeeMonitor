"""Frame sampling on the GPU endpoint: dispatch batches, collect results.

With ``SAMPLING_BACKEND=sagemaker`` sampled clips never decode in the web
process. Queued ``FrameSamplingTask`` rows are claimed into ``SamplingBatch``es
and sent to the SAM 3 endpoint's ``sample_label`` task, which samples and
pre-labels each clip in one pass (sagemaker_backend/inference.py,
memory/38_sampling_on_sagemaker.md). The reconciler tick drives it all:
``dispatch`` → ``collect`` → ``recover``.

Safe with several web processes (every gunicorn worker runs the reconciler,
App Runner runs several instances, deploys overlap them):

* tasks are claimed with a conditional UPDATE (QUEUED → PROCESSING, batch set),
  so each task lands in exactly one batch;
* a batch is collected only by the process that moves it INVOKED → COLLECTING;
* each task's results are applied only while it is still PROCESSING in *that*
  batch — a cancelled or superseded task is skipped, and collecting twice
  changes nothing.

The in-flight cap is soft: two processes may overshoot it by a batch each.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import timedelta

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from .models import Annotation, FrameSamplingTask, SamplingBatch

logger = logging.getLogger(__name__)

METHOD = "sample_label"
INVOCATION_TIMEOUT_S = 1800          # SageMaker writes a failure record past this
REQUEST_TTL_S = 3 * 3600             # …or if the request waited in the queue this long
NEVER_INVOKED_AFTER = timedelta(minutes=10)
NO_ANSWER_AFTER = timedelta(seconds=REQUEST_TTL_S + INVOCATION_TIMEOUT_S) + timedelta(minutes=30)
MAX_ATTEMPTS = 2
ARCHIVED_MESSAGE = ("This clip is archived in cold storage and can't be read until it "
                    "is restored.")


def enabled() -> bool:
    return getattr(settings, "SAMPLING_BACKEND", "local") == "sagemaker"


def params_for(classes, max_frames=20, min_gap_s=0.5, roi="device", replace=True) -> dict:
    """Task params for GPU sample-and-pre-label (method ``sample_label``)."""
    return {"method": METHOD, "classes": [c for c in (classes or []) if c],
            "max_frames": max(1, min(int(max_frames), 200)),
            "min_gap_s": float(min_gap_s), "roi": "frame" if roi == "frame" else "device",
            "replace": bool(replace)}


def start(tasks) -> None:
    """Hand newly created tasks to whichever backend samples them."""
    from . import sampling
    for task in tasks:
        if (task.params or {}).get("method") == METHOD:
            continue            # stays QUEUED; the reconciler's dispatch() sends it
        sampling.spawn_sampling_async(task.pk)


def supersede(project, video_ids) -> int:
    """A clip being sampled again cancels its older queued tasks."""
    return FrameSamplingTask.objects.filter(
        project=project, video_id__in=list(video_ids),
        status=FrameSamplingTask.Status.QUEUED,
    ).update(status=FrameSamplingTask.Status.CANCELLED, completed_at=timezone.now())


# ── dispatch ────────────────────────────────────────────────────────────────

def _in_flight() -> int:
    return SamplingBatch.objects.filter(status__in=[
        SamplingBatch.Status.CLAIMED, SamplingBatch.Status.INVOKED,
        SamplingBatch.Status.COLLECTING]).count()


def _claim_batch():
    """Reserve one batch of queued tasks sharing the same params, or None."""
    queued = FrameSamplingTask.objects.filter(
        status=FrameSamplingTask.Status.QUEUED, params__method=METHOD)
    first = queued.order_by("created_at").first()
    if first is None:
        return None, []
    size = 1 if (first.params or {}).get("solo") else max(1, settings.SAMPLING_BATCH_CLIPS)
    same = queued.filter(params=first.params).order_by("created_at")
    ids = list(same.values_list("pk", flat=True)[:size])
    batch = SamplingBatch.objects.create(
        batch_id=f"sl-{uuid.uuid4().hex[:16]}", params=first.params or {})
    batch.result_key = f"sampling-results/{batch.batch_id}.json"
    batch.save(update_fields=["result_key"])
    claimed = FrameSamplingTask.objects.filter(
        pk__in=ids, status=FrameSamplingTask.Status.QUEUED,
    ).update(status=FrameSamplingTask.Status.PROCESSING, batch=batch,
             started_at=timezone.now())
    tasks = list(FrameSamplingTask.objects.filter(batch=batch)
                 .select_related("video", "video__device").order_by("created_at", "pk"))
    if not claimed or not tasks:
        SamplingBatch.objects.filter(pk=batch.pk).update(
            status=SamplingBatch.Status.FAILED, error="nothing claimed",
            finished_at=timezone.now())
        return None, []
    return batch, tasks


def _clip_payload(task) -> dict | None:
    """The GPU's view of one clip, or None if the GPU can't read it."""
    from apps.devices.layouts import layout_for_video

    video = task.video
    blob = video.storage_key or ""
    if not blob or blob.startswith("s3://"):
        FrameSamplingTask.objects.filter(pk=task.pk).update(
            status=FrameSamplingTask.Status.FAILED, completed_at=timezone.now(),
            error_message=("This video still lives in an external bucket. Open it in "
                           "the editor once to ingest it, then sample."))
        return None
    roi = polygon = None
    if (task.params or {}).get("roi") == "device" and video.device_id:
        layout = layout_for_video(video)
        roi, polygon = layout["roi_override"], layout["roi_polygon"]
    return {"task_id": task.pk, "video_blob_path": blob,
            "max_frames": (task.params or {}).get("max_frames", 20),
            "roi": roi, "polygon": polygon}


def _invoke(batch, clips) -> None:
    import boto3
    from botocore.config import Config

    params = batch.params or {}
    payload = {
        "task": METHOD, "batch_id": batch.batch_id, "job_id": batch.batch_id,
        "result_bucket": settings.SAGEMAKER_OUTPUT_BUCKET, "result_key": batch.result_key,
        "classes": params.get("classes") or ["bee"],
        "confidence": 0.3, "candidates": settings.SAMPLING_CANDIDATES,
        "min_gap_s": params.get("min_gap_s", 0.5), "clips": clips,
    }
    region = getattr(settings, "AWS_REGION", "us-east-1")
    cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2})
    key = f"sampling/{batch.batch_id}.json"
    boto3.client("s3", region_name=region, config=cfg).put_object(
        Bucket=settings.SAGEMAKER_INPUT_BUCKET, Key=key,
        Body=json.dumps(payload).encode("utf-8"), ContentType="application/json")
    resp = boto3.client("sagemaker-runtime", region_name=region, config=cfg).invoke_endpoint_async(
        EndpointName=settings.SAGEMAKER_SAM3_ENDPOINT_NAME,
        InputLocation=f"s3://{settings.SAGEMAKER_INPUT_BUCKET}/{key}",
        ContentType="application/json", InferenceId=batch.batch_id,
        InvocationTimeoutSeconds=INVOCATION_TIMEOUT_S, RequestTTLSeconds=REQUEST_TTL_S)
    out = resp["OutputLocation"]
    SamplingBatch.objects.filter(pk=batch.pk, status=SamplingBatch.Status.CLAIMED).update(
        status=SamplingBatch.Status.INVOKED, invoked_at=timezone.now(), output_uri=out,
        failure_uri=resp.get("FailureLocation") or out.replace(".out", ".failure"))


def dispatch() -> int:
    """Send queued clips to the GPU in batches, up to the in-flight cap."""
    if not enabled() or not settings.SAGEMAKER_SAM3_ENDPOINT_NAME:
        return 0
    sent = 0
    room = settings.SAMPLING_MAX_BATCHES_IN_FLIGHT - _in_flight()
    while room > 0:
        with transaction.atomic():
            batch, tasks = _claim_batch()
        if batch is None:
            break
        clips = [c for c in (_clip_payload(t) for t in tasks) if c]
        if not clips:
            SamplingBatch.objects.filter(pk=batch.pk).update(
                status=SamplingBatch.Status.FAILED, error="no readable clips",
                finished_at=timezone.now())
            continue
        try:
            _invoke(batch, clips)
            sent += 1
            room -= 1
        except Exception as exc:
            # Left CLAIMED: recover() re-queues it after NEVER_INVOKED_AFTER.
            logger.exception("sampling batch %s invoke failed", batch.batch_id)
            SamplingBatch.objects.filter(pk=batch.pk).update(error=str(exc)[:500])
            break
    return sent


# ── collect ─────────────────────────────────────────────────────────────────

def _read(s3, uri_or_key, bucket=None):
    from urllib.parse import urlparse
    from botocore.exceptions import ClientError

    if bucket is None:
        u = urlparse(uri_or_key)
        bucket, key = u.netloc, u.path.lstrip("/")
    else:
        key = uri_or_key
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise


def _friendly(error: str) -> str:
    if "InvalidObjectState" in (error or "") or "storage class" in (error or "").lower():
        return ARCHIVED_MESSAGE
    return error or "Sampling failed."


def _write_frames(task, clip) -> int:
    """Upsert the clip's pre-labelled frames. Frames a person reviewed or
    labelled keep their boxes; only a missing image path is filled in."""
    classes = task.project.classes or []
    class_id = {c: i for i, c in enumerate(classes)}
    picked = {f["n"] for f in clip.get("frames", [])}
    if (task.params or {}).get("replace"):
        (Annotation.objects.filter(project=task.project, video=task.video, sampled_only=True,
                                   reviewed=False, boxes=[])
         .exclude(frame_number__in=picked).delete())
    written = 0
    for f in clip.get("frames", []):
        boxes = [{**b, "class_id": class_id[b["class"]]} for b in f.get("boxes", [])
                 if b.get("class") in class_id]
        existing = Annotation.objects.filter(project=task.project, video=task.video,
                                             frame_number=f["n"]).first()
        if existing and (existing.reviewed or existing.boxes):
            if not existing.frame_image_path:
                Annotation.objects.filter(pk=existing.pk).update(frame_image_path=f["key"])
            written += 1
            continue
        if existing:
            Annotation.objects.filter(pk=existing.pk).update(
                boxes=boxes, frame_image_path=f["key"], image_width=f.get("w", 1920),
                image_height=f.get("h", 1080), sampled_only=False)
        else:
            Annotation.objects.create(
                project=task.project, video=task.video, frame_number=f["n"], boxes=boxes,
                frame_image_path=f["key"], image_width=f.get("w", 1920),
                image_height=f.get("h", 1080), sampled_only=False)
        written += 1
    return written


def _apply(batch, clip) -> float:
    """Apply one clip's result if its task still belongs to this batch.
    Returns GPU-ish seconds to charge (0 if skipped)."""
    task_id = clip.get("task_id")
    with transaction.atomic():
        task = (FrameSamplingTask.objects.select_for_update()
                .select_related("project", "video", "user")
                .filter(pk=task_id, batch=batch, status=FrameSamplingTask.Status.PROCESSING)
                .first())
        if task is None:
            return 0.0            # cancelled, superseded, or already collected
        if clip.get("error"):
            FrameSamplingTask.objects.filter(pk=task.pk).update(
                status=FrameSamplingTask.Status.FAILED, completed_at=timezone.now(),
                error_message=_friendly(clip["error"])[:500])
            return 0.0
        written = _write_frames(task, clip)
        FrameSamplingTask.objects.filter(pk=task.pk).update(
            status=FrameSamplingTask.Status.COMPLETED, completed_at=timezone.now(),
            frames_written=written, motion=clip.get("motion"))
    return float((clip.get("seconds") or {}).get("scan_and_detect") or 0.0)


def _charge(batch, seconds_by_user) -> None:
    from apps.accounts.models import UserProfile
    for user_id, secs in seconds_by_user.items():
        if secs <= 0:
            continue
        try:
            profile, _ = UserProfile.objects.get_or_create(user_id=user_id)
            profile.charge(int(round(secs)), gpu_seconds=secs)
        except Exception:
            logger.exception("sampling batch %s credit charge failed", batch.batch_id)


def _retry_or_fail(batch, reason: str) -> None:
    """The whole batch failed: each clip goes back alone, up to MAX_ATTEMPTS."""
    for task in FrameSamplingTask.objects.filter(batch=batch, status=FrameSamplingTask.Status.PROCESSING):
        if task.attempts + 1 < MAX_ATTEMPTS:
            FrameSamplingTask.objects.filter(pk=task.pk, batch=batch).update(
                status=FrameSamplingTask.Status.QUEUED, batch=None, attempts=task.attempts + 1,
                params={**(task.params or {}), "solo": True})
        else:
            FrameSamplingTask.objects.filter(pk=task.pk, batch=batch).update(
                status=FrameSamplingTask.Status.FAILED, attempts=task.attempts + 1,
                completed_at=timezone.now(), error_message=_friendly(reason)[:500])
    SamplingBatch.objects.filter(pk=batch.pk).update(
        status=SamplingBatch.Status.FAILED, error=reason[:1000], finished_at=timezone.now())


def collect(limit: int = 20) -> int:
    """Write the results of finished batches. Idempotent; never raises."""
    import boto3
    from botocore.config import Config

    batches = list(SamplingBatch.objects.filter(status=SamplingBatch.Status.INVOKED)
                   .order_by("invoked_at")[:limit])
    if not batches:
        return 0
    s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"),
                      config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}))
    done = 0
    for batch in batches:
        try:
            body = (_read(s3, batch.result_key, settings.SAGEMAKER_OUTPUT_BUCKET)
                    or (_read(s3, batch.output_uri) if batch.output_uri else None))
            failure = None if body else (_read(s3, batch.failure_uri) if batch.failure_uri else None)
            if body is None and failure is None:
                if batch.invoked_at and timezone.now() - batch.invoked_at > NO_ANSWER_AFTER:
                    if SamplingBatch.objects.filter(pk=batch.pk, status=SamplingBatch.Status.INVOKED).update(
                            status=SamplingBatch.Status.COLLECTING):
                        _retry_or_fail(batch, "No result from the GPU endpoint.")
                        done += 1
                continue
            if not SamplingBatch.objects.filter(pk=batch.pk, status=SamplingBatch.Status.INVOKED).update(
                    status=SamplingBatch.Status.COLLECTING):
                continue        # another process has it
            if failure is not None:
                _retry_or_fail(batch, failure.decode("utf-8", "replace"))
                done += 1
                continue
            result = json.loads(body)
            seconds_by_user = {}
            owners = dict(FrameSamplingTask.objects.filter(batch=batch).values_list("pk", "user_id"))
            for clip in result.get("clips") or []:
                secs = _apply(batch, clip)
                uid = owners.get(clip.get("task_id"))
                if uid:
                    seconds_by_user[uid] = seconds_by_user.get(uid, 0.0) + secs
            # Clips the GPU never mentioned (shouldn't happen): back in the queue.
            _retry_or_fail_leftovers(batch)
            SamplingBatch.objects.filter(pk=batch.pk).update(
                status=SamplingBatch.Status.COLLECTED, finished_at=timezone.now())
            _charge(batch, seconds_by_user)
            done += 1
        except Exception:
            logger.exception("collecting sampling batch %s failed", batch.batch_id)
    return done


def _retry_or_fail_leftovers(batch) -> None:
    left = FrameSamplingTask.objects.filter(batch=batch, status=FrameSamplingTask.Status.PROCESSING)
    if left.exists():
        _retry_or_fail(batch, "The GPU returned no result for this clip.")


# ── recover ─────────────────────────────────────────────────────────────────

def recover() -> int:
    """Batches claimed but never invoked (a crash between the two) re-queue."""
    stale = SamplingBatch.objects.filter(status=SamplingBatch.Status.CLAIMED,
                                         claimed_at__lt=timezone.now() - NEVER_INVOKED_AFTER)
    n = 0
    for batch in stale:
        if SamplingBatch.objects.filter(pk=batch.pk, status=SamplingBatch.Status.CLAIMED).update(
                status=SamplingBatch.Status.FAILED, error=batch.error or "never invoked",
                finished_at=timezone.now()):
            FrameSamplingTask.objects.filter(batch=batch, status=FrameSamplingTask.Status.PROCESSING).update(
                status=FrameSamplingTask.Status.QUEUED, batch=None)
            n += 1
    return n


def tick() -> dict:
    """One reconciler pass. Collect always runs, so switching back to the
    local backend still drains batches already on the GPU."""
    out = {"collected": 0, "recovered": 0, "dispatched": 0}
    try:
        out["collected"] = collect()
        if enabled():
            out["recovered"] = recover()
            out["dispatched"] = dispatch()
    except Exception:
        logger.exception("sampling tick failed")
    return out


def project_status(project) -> dict:
    """Counts for the project page's sampling status line."""
    qs = FrameSamplingTask.objects.filter(project=project)
    latest_ids = (qs.order_by("video_id", "-created_at").distinct("video_id").values("pk")
                  if _supports_distinct_on() else None)
    counts = {s: qs.filter(status=s).count() for s in ("queued", "processing", "failed")}
    empty_qs = qs.filter(status=FrameSamplingTask.Status.COMPLETED, frames_written=0)
    if latest_ids is not None:
        empty_qs = empty_qs.filter(pk__in=latest_ids)
    counts["no_activity"] = empty_qs.count()
    return counts


def _supports_distinct_on() -> bool:
    from django.db import connection
    return connection.features.can_distinct_on_fields


__all__ = ["enabled", "params_for", "start", "supersede", "dispatch", "collect",
           "recover", "tick", "project_status"]
