import json
import logging
import os
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse, reverse_lazy
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView

from apps.videos.models import Video
from config.storage import get_s3_client

from .analytics import (
    get_activity_over_time,
    get_cumulative_activity,
    get_foraging_trips_over_time,
    get_nest_activity_heatmap,
    get_period_averages,
    get_summary_stats,
    get_trips_per_nest,
    get_video_breakdown,
)
from .forms import JobCreateForm
from .models import Job, JobResult, GPU_TIERS

logger = logging.getLogger(__name__)

_NATALIES_RE = re.compile(r"natalies?", re.IGNORECASE)
_SITEA_RE = re.compile(r"SiteA", re.IGNORECASE)


def _sanitize_site(value: str) -> str:
    """Replace occurrences of 'natalies' with 'SiteA' in display strings."""
    if not value:
        return value
    return _NATALIES_RE.sub("SiteA", value)


def _unsanitize_site(value: str) -> str:
    """Reverse-map 'SiteA' back to 'natalies' for DB queries."""
    if not value:
        return value
    return _SITEA_RE.sub("natalies", value)


# GET/POST params the Processing-hub video filter understands.
VIDEO_FILTER_KEYS = ("device", "site", "year", "month", "day", "hour",
                     "from", "to", "q", "confirmed")


def apply_video_filters(qs, params):
    """Apply the Processing-hub video filters to a Video queryset. ``params`` is
    any dict-like with .get() (a GET or POST QueryDict). Shared by the hub list
    and the pipeline "run on all filtered videos" path so they never diverge."""
    from datetime import datetime, time
    from django.utils import timezone as _tz
    from django.utils.dateparse import parse_date, parse_datetime

    q = (params.get("q") or "").strip()
    if q:
        qs = qs.filter(title__icontains=q)
    if params.get("device"):
        qs = qs.filter(device_id=params.get("device"))
    if params.get("site"):
        qs = qs.filter(site_name=_unsanitize_site(params.get("site")))
    for field in ("year", "month", "day", "hour"):
        val = params.get(field)
        if val:
            try:
                qs = qs.filter(**{field: int(val)})
            except (ValueError, TypeError):
                pass

    def _parse_dt(s):
        dt = parse_datetime(s)
        if dt is None:
            d = parse_date(s)
            if d:
                dt = datetime.combine(d, time.min)
        if dt and _tz.is_naive(dt):
            dt = _tz.make_aware(dt, _tz.get_current_timezone())
        return dt

    if params.get("from"):
        dt = _parse_dt(params.get("from"))
        if dt:
            qs = qs.filter(recorded_at__gte=dt)
    if params.get("to"):
        dt = _parse_dt(params.get("to"))
        if dt:
            qs = qs.filter(recorded_at__lte=dt)

    confirmed = params.get("confirmed")
    if confirmed == "yes":
        qs = qs.filter(metadata__bee_confirmed=True)
    elif confirmed == "no":
        qs = qs.filter(metadata__bee_confirmed=False)
    return qs


def _generate_presigned_url(blob_path: str, container: str = "processed") -> str:
    """Time-limited URL for a blob in S3. Empty string on any error."""
    if not blob_path:
        return ""
    try:
        return get_s3_client().generate_presigned_url(container, blob_path)
    except Exception as e:
        logger.error("Failed to presign %s/%s: %s", container, blob_path, e)
        return ""


def _ingest_external_s3_to_storage(video) -> str:
    """Copy a video from an external S3 source into our raw-videos bucket.

    Returns the new ``storage_key`` (the key in our bucket). If the video is
    already in our storage (``storage_key`` doesn't start with ``s3://``),
    this is a no-op and returns the existing key.
    """
    storage_key = video.storage_key
    if not storage_key.startswith("s3://"):
        return storage_key  # already in our bucket

    import boto3
    import tempfile
    from pathlib import Path

    meta = video.metadata or {}
    remote_key = meta.get("remote_key", storage_key.replace("s3://", ""))
    bucket = meta.get("remote_bucket", "")
    if not bucket:
        raise ValueError("No S3 bucket in video metadata")
    if not video.source:
        raise ValueError("Video has no linked data source for S3 credentials")

    from apps.sources.views import _decrypt_credentials
    creds = _decrypt_credentials(video.source)

    external_s3 = boto3.client(
        "s3",
        aws_access_key_id=creds["access_key_id"],
        aws_secret_access_key=creds["secret_access_key"],
        region_name=creds.get("region", "us-east-1"),
    )

    filename = remote_key.split("/")[-1]
    new_key = f"{video.user_id}/{video.pk}/{filename}"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        logger.info(
            "Ingesting external s3://%s/%s -> raw-videos/%s",
            bucket, remote_key, new_key,
        )
        external_s3.download_file(bucket, remote_key, tmp.name)
        file_size = Path(tmp.name).stat().st_size
        get_s3_client().upload_file(
            "raw-videos", new_key, tmp.name, content_type="video/mp4",
        )

    video.storage_key = new_key
    video.file_size_bytes = file_size
    video.save(update_fields=["storage_key", "file_size_bytes"])
    creds.clear()

    logger.info("Ingest complete: %s (%d MB)", new_key, file_size // (1024 * 1024))
    return new_key


# ---------------------------------------------------------------------------
# SageMaker Async Inference spawn (Phase 4)
# ---------------------------------------------------------------------------
# The Job model's ``modal_*`` fields are repurposed (no migration needed):
#   modal_job_id  -> our own UUID, passed through as SageMaker InferenceId.
#                    Also the key under which payload + output land in S3.
#   modal_call_id -> the s3:// URI of the expected output JSON. PollJobsView
#                    head_objects this URI to detect completion.


def _sagemaker_runtime():
    """Boto3 sagemaker-runtime client, region-aware."""
    import boto3
    from botocore.config import Config
    return boto3.client(
        "sagemaker-runtime",
        region_name=getattr(settings, "AWS_REGION", "us-east-1"),
        # Fail fast. Without this, a network gap (e.g. a missing VPC endpoint)
        # blocks the worker — and the DB connection it holds — for 60s+ per call.
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 1}),
    )


# Bounded pool for single-job SageMaker spawns. A burst of uploads (e.g. a Pi
# video backlog draining over WiFi) previously started one thread per upload,
# each holding a DB connection across a multi-second SageMaker call — enough to
# exhaust the database's connection slots. Cap the concurrency instead.
_SPAWN_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="gpu-spawn")


def spawn_gpu_job_async(job_pk: int) -> None:
    """Queue a single-job SageMaker spawn on the bounded pool (non-blocking)."""
    _SPAWN_POOL.submit(_spawn_gpu_job, job_pk)


def _spawn_gpu_job(job_pk: int) -> None:
    """Invoke the SageMaker Async endpoint for a single job.

    Steps:
      1. External-S3 ingested videos are copied into our raw-videos bucket
         (the GPU container only reads from there).
      2. Payload JSON is written to the SM input bucket.
      3. ``invoke_endpoint_async`` returns immediately with an ``OutputLocation``
         — that s3:// URI is stashed on the job so PollJobsView can detect
         completion by HEAD-ing it.
    """
    import django
    django.setup()
    from django.db import connection

    try:
        job = Job.objects.select_related("video", "video__source").get(pk=job_pk)
        user_id = str(job.user_id)
        video = job.video
        blob_path = video.storage_key
    except Job.DoesNotExist:
        connection.close()
        return
    except Exception as e:
        Job.objects.filter(pk=job_pk).update(status="failed", error_message=str(e))
        connection.close()
        return

    # Release the DB connection during the slow S3/SageMaker round-trips so we
    # never hold a pooled connection across a multi-second network call.
    connection.close()

    try:
        if blob_path.startswith("s3://"):
            blob_path = _ingest_external_s3_to_storage(video)

        payload = {
            "job_id": job.modal_job_id,
            "user_id": user_id,
            "video_blob_path": blob_path,
            "detection_mode": job.config.get("detection_mode", "yolo"),
            "confidence_threshold": job.config.get("confidence_threshold", 0.25),
            # Annotated-video rendering is opt-in: it writes + uploads every
            # frame, which roughly doubles runtime and can blow the async time
            # cap on long clips. Tracking/events don't need it.
            "visualize": bool(job.config.get("visualize", False)),
        }
        if job.config.get("custom_nest_model_path"):
            payload["custom_nest_model_path"] = job.config["custom_nest_model_path"]
        if job.config.get("custom_bee_model_path"):
            payload["custom_bee_model_path"] = job.config["custom_bee_model_path"]
        # Device hotel ROI + nest tubes (set via "use device ROI") → worker uses
        # them; the nest model is the backup.
        if job.config.get("hotel_roi"):
            payload["hotel_roi"] = job.config["hotel_roi"]
        if job.config.get("nest_layout"):
            payload["nest_layout"] = job.config["nest_layout"]
        payload["run_tracking"] = bool(job.config.get("run_tracking", True))
        if job.config.get("ml_threshold") is not None:
            payload["ml_threshold"] = float(job.config["ml_threshold"])
        # Detector: SAM 3 text-prompt tracking, else YOLO.
        if job.config.get("detector_kind") == "sam3":
            payload["detector_kind"] = "sam3"
            payload["text_prompt"] = job.config.get("text_prompt", "") or "bee"
        # Recording start metadata → event timestamps; no filename convention needed.
        if video.recorded_at:
            payload["recorded_at"] = video.recorded_at.isoformat()

        modal_call_id, extra_cfg = _launch_gpu(payload, video, job.modal_job_id)
        if extra_cfg:
            job.config.update(extra_cfg)
            Job.objects.filter(pk=job_pk).update(modal_call_id=modal_call_id, config=job.config)
        else:
            Job.objects.filter(pk=job_pk).update(modal_call_id=modal_call_id)
        logger.info("Job %s spawned on SageMaker: %s", job_pk, modal_call_id)

    except Exception as e:
        Job.objects.filter(pk=job_pk).update(status="failed", error_message=str(e))
        logger.exception("Job %s spawn failed", job_pk)
    finally:
        connection.close()


def _spawn_gpu_batch(jobs_data: list, detection_mode: str, confidence: float,
                     custom_nest_model_path: str = "", custom_bee_model_path: str = "") -> None:
    """Spawn each job in the batch individually.

    BeeMonitor's SageMaker endpoint processes one video per invocation —
    the chunked-Modal pattern (one container handling many videos) is gone
    because the SageMaker pricing model doesn't reward it: scale-to-zero +
    per-instance billing means parallelism happens via the endpoint's own
    autoscaling, not by packing videos into one call.
    """
    import django
    django.setup()
    from django.db import connection

    try:
        # Ingest any external-S3 videos first so SM only sees our buckets.
        for jd in jobs_data:
            video = jd["video"]
            if video.storage_key.startswith("s3://"):
                try:
                    _ingest_external_s3_to_storage(video)
                except Exception as e:
                    Job.objects.filter(pk=jd["job_pk"]).update(
                        status="failed", error_message=f"Ingest failed: {e}",
                    )

        for jd in jobs_data:
            video = jd["video"]
            if video.storage_key.startswith("s3://"):
                continue  # ingest failed; already marked failed above
            try:
                payload = {
                    "job_id": jd["job_id"],
                    "user_id": jd["user_id"],
                    "video_blob_path": video.storage_key,
                    "detection_mode": detection_mode,
                    "confidence_threshold": confidence,
                    "visualize": bool((jd.get("config") or {}).get("visualize", False)),
                }
                if custom_nest_model_path:
                    payload["custom_nest_model_path"] = custom_nest_model_path
                if custom_bee_model_path:
                    payload["custom_bee_model_path"] = custom_bee_model_path
                # Per-video device hotel ROI + nest tubes (model is the backup).
                vcfg = jd.get("config") or {}
                if vcfg.get("hotel_roi"):
                    payload["hotel_roi"] = vcfg["hotel_roi"]
                if vcfg.get("nest_layout"):
                    payload["nest_layout"] = vcfg["nest_layout"]
                payload["run_tracking"] = bool(vcfg.get("run_tracking", True))
                if vcfg.get("ml_threshold") is not None:
                    payload["ml_threshold"] = float(vcfg["ml_threshold"])
                if vcfg.get("detector_kind") == "sam3":
                    payload["detector_kind"] = "sam3"
                    payload["text_prompt"] = vcfg.get("text_prompt", "") or "bee"
                # Recording start metadata → event timestamps; no filename convention needed.
                if video.recorded_at:
                    payload["recorded_at"] = video.recorded_at.isoformat()

                modal_call_id, extra_cfg = _launch_gpu(payload, video, jd["job_id"])
                if extra_cfg:
                    vcfg.update(extra_cfg)
                    Job.objects.filter(pk=jd["job_pk"]).update(
                        modal_call_id=modal_call_id, config=vcfg)
                else:
                    Job.objects.filter(pk=jd["job_pk"]).update(modal_call_id=modal_call_id)
                logger.info("Job %s spawned: %s", jd["job_pk"], modal_call_id)
            except Exception as e:
                Job.objects.filter(pk=jd["job_pk"]).update(
                    status="failed", error_message=str(e),
                )
                logger.exception("Job %s batch-spawn failed", jd["job_pk"])

        logger.info("Batch spawn complete: %d jobs", len(jobs_data))

    except Exception as e:
        logger.exception("Batch spawn failed: %s", e)
        job_pks = [jd["job_pk"] for jd in jobs_data]
        Job.objects.filter(pk__in=job_pks, status="processing").update(
            status="failed", error_message=f"Spawn error: {e}",
        )
    finally:
        connection.close()


def _put_inference_payload(job_id: str, payload: dict) -> str:
    """Write the SageMaker request payload to the SM input bucket. Returns s3:// URI."""
    import json as _json
    bucket = settings.SAGEMAKER_INPUT_BUCKET
    if not bucket:
        raise RuntimeError("SAGEMAKER_INPUT_BUCKET is not configured")
    key = f"{job_id}.json"
    # Direct boto3 PUT; the input bucket isn't one of the 4 we wrap in S3StorageClient.
    import boto3
    s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=_json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )
    return f"s3://{bucket}/{key}"


def _invoke_endpoint_async(job_id: str, input_uri: str) -> tuple[str, str]:
    """Call invoke_endpoint_async. Returns (output_uri, failure_uri) — the s3://
    URIs where the result / a platform-level failure (container crash, request
    expiry) will land. failure_uri is "" unless the endpoint config has an
    S3FailurePath."""
    endpoint = settings.SAGEMAKER_ENDPOINT_NAME
    if not endpoint:
        raise RuntimeError("SAGEMAKER_ENDPOINT_NAME is not configured")
    response = _sagemaker_runtime().invoke_endpoint_async(
        EndpointName=endpoint,
        InputLocation=input_uri,
        ContentType="application/json",
        InferenceId=job_id,
        # Platform default is 15 MINUTES — long clips (e.g. continuous-mode
        # 10-min videos) blow past it and die as "Timed out uploading object".
        # 3600 is the async platform maximum.
        InvocationTimeoutSeconds=3600,
    )
    # OutputLocation = s3://<output-bucket>/<inference-id>.out  (typical layout)
    return response["OutputLocation"], response.get("FailureLocation", "") or ""


# ── Chunked tracking for long videos ─────────────────────────────────────────
# The async platform hard-caps one invocation at 1 h. Long videos are split
# into frame ranges, one invocation each, and the CSVs merged when all land.
# Chunk sizes keep worst-case per-chunk GPU time comfortably under the cap
# (SAM 3 runs a heavy transformer on EVERY frame, hence the tiny chunks).
#
# GATED behind BEEMONITOR_CHUNK_TRACKING=1: the GPU image must support
# start_frame/end_frame first — an older image would silently ignore them and
# process the WHOLE video once per chunk (duplicated results, N× cost). Flip
# the env on only after the endpoint runs the range-aware image.
_CHUNK_LIMIT_SECONDS = {"yolo": 1200.0, "sam3": 60.0}


def _chunk_ranges(video, detector_kind):
    """[(start_frame, end_frame_or_None), ...] for a long video, else None.

    Built from duration/fps metadata. Boundaries are shared (chunk i's end ==
    chunk i+1's start) so coverage has no gap or overlap even if the fps
    estimate is off, and the LAST chunk runs to EOF so an imprecise duration
    can never truncate the tail."""
    import math

    if os.environ.get("BEEMONITOR_CHUNK_TRACKING") != "1":
        return None
    limit = _CHUNK_LIMIT_SECONDS["sam3" if detector_kind == "sam3" else "yolo"]
    dur = float(getattr(video, "duration_seconds", 0) or 0)
    if dur <= limit:
        return None  # fits one invocation (also: unknown duration -> unchunked)
    fps = float(getattr(video, "fps", 0) or 30.0)
    total_frames = int(dur * fps)
    n = math.ceil(dur / limit)
    per = math.ceil(total_frames / n)
    return [(i * per, (i + 1) * per if i < n - 1 else None) for i in range(n)]


def _launch_gpu(payload, video, mid):
    """Fire one invocation — or N chunked ones for a long video.

    Returns (modal_call_id, cfg_updates): cfg_updates carries either the
    single invocation's failure_location or the chunk manifest the poller
    drives to completion."""
    ranges = _chunk_ranges(video, payload.get("detector_kind", "yolo"))
    if not ranges:
        input_uri = _put_inference_payload(mid, payload)
        output_uri, failure_uri = _invoke_endpoint_async(mid, input_uri)
        return output_uri, ({"failure_location": failure_uri} if failure_uri else {})

    chunks = []
    for i, (start, end) in enumerate(ranges):
        cp = dict(payload)
        cp["job_id"] = f"{mid}-c{i}"
        cp["start_frame"] = start
        if end is not None:
            cp["end_frame"] = end
        cp["visualize"] = False  # a merged annotated video isn't supported
        input_uri = _put_inference_payload(cp["job_id"], cp)
        out_uri, fail_uri = _invoke_endpoint_async(cp["job_id"], input_uri)
        chunks.append({"i": i, "output_uri": out_uri,
                       "failure_uri": fail_uri, "result": None})
    logger.info("job %s: video %.0fs chunked into %d invocations",
                mid, float(video.duration_seconds or 0), len(chunks))
    return chunks[0]["output_uri"], {"chunks": chunks}


ACTIVE_JOB_STATUSES = [
    Job.Status.QUEUED, Job.Status.INGESTING,
    Job.Status.PROCESSING, Job.Status.POST_PROCESSING,
]


class JobCancelView(LoginRequiredMixin, View):
    """Cancel an in-flight job to free a GPU slot.

    SageMaker async has no API to abort a queued/running request, so this marks
    the Job cancelled (poller stops tracking it; a late result is ignored) and
    fails any pipeline-run step waiting on it so the run doesn't hang. GPU time
    already spent still bills.
    """

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.utils import timezone

        job = get_object_or_404(Job, pk=pk, user=request.user)
        if job.status in ACTIVE_JOB_STATUSES:
            job.status = Job.Status.CANCELLED
            job.completed_at = timezone.now()
            job.error_message = "Cancelled by user."
            job.save(update_fields=["status", "completed_at", "error_message"])
            try:
                from apps.pipelines import engine
                engine.on_job_finished(job)
            except Exception:
                logger.exception("cancel: pipeline hook failed for job %s", pk)
            messages.info(request, f"Cancelled analysis of '{job.video.title}'.")
        else:
            messages.warning(request, "That job already finished.")
        return redirect(request.META.get("HTTP_REFERER") or "analysis:processing")


class JobCancelAllView(LoginRequiredMixin, View):
    """Cancel every in-flight job for this user (free all GPU slots)."""

    def post(self, request):
        from django.shortcuts import redirect
        from django.utils import timezone

        jobs = list(Job.objects.filter(user=request.user, status__in=ACTIVE_JOB_STATUSES))
        for job in jobs:
            job.status = Job.Status.CANCELLED
            job.completed_at = timezone.now()
            job.error_message = "Cancelled by user."
            job.save(update_fields=["status", "completed_at", "error_message"])
            try:
                from apps.pipelines import engine
                engine.on_job_finished(job)
            except Exception:
                logger.exception("cancel-all: pipeline hook failed for job %s", job.pk)
        messages.info(request, f"Cancelled {len(jobs)} job(s).")
        return redirect("analysis:processing")


class ProcessingHubView(LoginRequiredMixin, View):
    """Phase 0 'Processing' hub — the home of video processing.

    Pick a video and run analysis with one click, then jump to the ecological
    outputs it produces (tracking, foraging trips, interactions, species ID).
    Wraps the existing Job pipeline + analytics; the visual workflow builder
    grows from here (see memory/19_processing_workflow_builder_design).
    """

    template_name = "analysis/processing.html"
    MODES = {"detect_and_track", "detect_only", "count_only"}

    def get(self, request):
        from urllib.parse import urlencode
        from django.db.models import OuterRef, Subquery
        from apps.devices.models import Device
        from apps.training.models import CustomModel
        from apps.pipelines.models import Pipeline

        # Own videos + videos from devices shared with me (viewer or manager).
        user_videos = Video.accessible(request.user)
        qs = user_videos

        # Comprehensive filter: device · site · year · month · day · hour · date
        # range · confirmation · title. `f` mirrors the GET params for the form +
        # the CSV download links; apply_video_filters does the actual filtering
        # (shared with the pipeline "run on all filtered" path).
        f = {k: request.GET.get(k, "") for k in VIDEO_FILTER_KEYS}
        qs = apply_video_filters(qs, request.GET)

        latest = (Job.objects.filter(video=OuterRef("pk"))
                  .order_by("-id").values("status")[:1])
        latest_id = (Job.objects.filter(video=OuterRef("pk"))
                     .order_by("-id").values("id")[:1])
        videos = (qs.annotate(job_status=Subquery(latest), job_id=Subquery(latest_id))
                  .select_related("device")
                  .order_by("-recorded_at", "-uploaded_at")[:200])
        recent_jobs = (Job.objects.filter(user=request.user)
                       .select_related("video").order_by("-id")[:8])

        devices = list(Device.accessible(request.user).order_by("name"))
        roi_devices = [d.name for d in devices if d.roi_override and d.nest_layout]
        models = CustomModel.objects.filter(user=request.user, is_active=True)

        # Dropdown options from the user's actual videos.
        opts = {
            "sites": sorted({_sanitize_site(s) for s in
                             user_videos.exclude(site_name="").values_list("site_name", flat=True)}),
            "years": sorted(set(user_videos.exclude(year=None).values_list("year", flat=True))),
            "months": sorted(set(user_videos.exclude(month=None).values_list("month", flat=True))),
            "days": sorted(set(user_videos.exclude(day=None).values_list("day", flat=True))),
            "hours": sorted(set(user_videos.exclude(hour=None).values_list("hour", flat=True))),
        }
        # Query string for the CSV downloads — the download views filter on these.
        dl = {k: f[k] for k in ("device", "site", "year", "month", "day", "hour", "confirmed", "from", "to") if f[k]}
        download_qs = ("?" + urlencode(dl)) if dl else ""

        # Everything currently in flight, with GPU-slot usage — cancellable to
        # free slots for new runs.
        from apps.accounts.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        active_jobs = list(
            Job.objects.filter(user=request.user, status__in=ACTIVE_JOB_STATUSES)
            .select_related("video").order_by("started_at", "id")
        )

        return render(request, self.template_name, {
            "videos": videos,
            "recent_jobs": recent_jobs,
            "active_jobs": active_jobs,
            "slots_used": sum(1 for j in active_jobs if j.status == Job.Status.PROCESSING),
            "slots_max": profile.max_concurrent_jobs,
            "video_count": qs.count(),
            "devices": devices,
            "roi_devices": roi_devices,
            "f": f, "opts": opts, "download_qs": download_qs,
            "has_filter": any(f.values()),
            # Pure viewers (manage no videos) see a read-only hub: no run controls.
            "can_manage_any": Video.manageable(request.user).exists(),
            "custom_nest_models": models.filter(model_type__in=["nest_detection", "custom"]),
            "custom_bee_models": models.filter(model_type__in=["bee_tracking", "custom"]),
            "est_credits_per_video": 349,
            # Pipelines to run on the selected videos (replaces the fixed recipe).
            "pipelines": Pipeline.objects.filter(user=request.user, is_template=False).order_by("title"),
            "pipeline_templates": Pipeline.objects.filter(is_template=True).order_by("title"),
        })

    def post(self, request):
        """One-click: run analysis on a video (creates a Job + spawns it)."""
        from django.utils import timezone
        # Owner or device-manager may run analysis; the Job is billed to the
        # runner (request.user). Viewers are excluded by Video.manageable.
        video = get_object_or_404(
            Video.manageable(request.user), pk=request.POST.get("video"))
        mode = request.POST.get("mode", "detect_and_track")
        if mode not in self.MODES:
            mode = "detect_and_track"
        job = Job.objects.create(
            user=request.user, video=video,
            config={"detection_mode": mode, "confidence_threshold": 0.25},
            status=Job.Status.PROCESSING, started_at=timezone.now(),
            modal_job_id=f"modal_{uuid.uuid4().hex[:12]}",
        )
        spawn_gpu_job_async(job.pk)
        messages.info(
            request,
            f"Analysis #{job.pk} started on “{video.title}”. It runs on the GPU; "
            "the job page auto-refreshes with results.")
        return redirect("analysis:detail", pk=job.pk)


class JobListView(LoginRequiredMixin, ListView):
    template_name = "analysis/list.html"
    context_object_name = "jobs"
    paginate_by = 50

    def get_queryset(self):
        qs = Job.objects.filter(user=self.request.user).select_related("video")

        # Filters
        status = self.request.GET.get("status", "")
        site = self.request.GET.get("site", "")
        year = self.request.GET.get("year", "")
        month = self.request.GET.get("month", "")
        day = self.request.GET.get("day", "")
        hour = self.request.GET.get("hour", "")

        if status:
            qs = qs.filter(status=status)
        if site:
            qs = qs.filter(video__site_name=_unsanitize_site(site))
        if year:
            try:
                qs = qs.filter(video__year=int(year))
            except (ValueError, TypeError):
                pass
        if month:
            try:
                qs = qs.filter(video__month=int(month))
            except (ValueError, TypeError):
                pass
        if day:
            try:
                qs = qs.filter(video__day=int(day))
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                qs = qs.filter(video__hour=int(hour))
            except (ValueError, TypeError):
                pass

        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user_videos = Video.objects.filter(user=self.request.user)

        ctx["site_names"] = sorted(set(
            _sanitize_site(s) for s in
            user_videos.exclude(site_name="").values_list("site_name", flat=True)
        ))
        ctx["years"] = sorted(set(
            user_videos.exclude(year=None).values_list("year", flat=True)
        ))
        ctx["months"] = sorted(set(
            user_videos.exclude(month=None).values_list("month", flat=True)
        ))
        ctx["days"] = sorted(set(
            user_videos.exclude(day=None).values_list("day", flat=True)
        ))
        ctx["hours"] = sorted(set(
            user_videos.exclude(hour=None).values_list("hour", flat=True)
        ))
        ctx["statuses"] = Job.Status.choices

        ctx["current_status"] = self.request.GET.get("status", "")
        ctx["current_site"] = self.request.GET.get("site", "")
        ctx["current_year"] = self.request.GET.get("year", "")
        ctx["current_month"] = self.request.GET.get("month", "")
        ctx["current_day"] = self.request.GET.get("day", "")
        ctx["current_hour"] = self.request.GET.get("hour", "")

        return ctx


class JobDetailView(LoginRequiredMixin, DetailView):
    template_name = "analysis/detail.html"
    context_object_name = "job"

    def get_queryset(self):
        return Job.objects.filter(user=self.request.user).select_related("video")


class JobCreateView(LoginRequiredMixin, FormView):
    template_name = "analysis/new.html"
    form_class = JobCreateForm

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def get_initial(self):
        initial = super().get_initial()
        video_id = self.request.GET.get("video")
        if video_id:
            initial["video"] = video_id
        return initial

    def form_valid(self, form):
        import threading
        from django.utils import timezone

        config = {
            "detection_mode": form.cleaned_data["detection_mode"],
            "confidence_threshold": form.cleaned_data["confidence_threshold"],
        }
        # Resolve custom model paths from POST (not form fields — these come from template selects)
        from apps.training.models import CustomModel
        for key, config_key in [("custom_nest_model", "custom_nest_model_path"), ("custom_bee_model", "custom_bee_model_path")]:
            model_id = self.request.POST.get(key, "")
            if model_id:
                try:
                    cm = CustomModel.objects.get(pk=model_id, user=self.request.user, is_active=True)
                    config[config_key] = cm.storage_key
                except CustomModel.DoesNotExist:
                    pass

        job = Job.objects.create(
            user=self.request.user,
            video=form.cleaned_data["video"],
            config=config,
            status=Job.Status.PROCESSING,
            started_at=timezone.now(),
            modal_job_id=f"modal_{uuid.uuid4().hex[:12]}",
        )

        # Spawn (non-blocking, bounded pool) — PollJobsView checks for results
        spawn_gpu_job_async(job.pk)

        messages.info(self.request, f"Job #{job.pk} submitted — processing on GPU. This page auto-refreshes.")
        self._job_pk = job.pk
        return super().form_valid(form)

    def get_success_url(self):
        return reverse("analysis:detail", kwargs={"pk": self._job_pk})


class PollJobsView(LoginRequiredMixin, View):
    """HTMX endpoint: check SageMaker for completed jobs and update DB.

    Called every 10s from the Analysis list page. For each in-flight Job,
    HEAD-checks the s3:// URI stored in ``modal_call_id`` (now the
    SageMaker async OutputLocation). If the object exists -> fetch + parse
    + mark complete. Also checks the parallel ``.failure`` location for
    inference errors.
    """

    def get(self, request):
        from django.http import JsonResponse

        # ALL in-flight jobs, however old and even if the spawn never finished
        # (empty modal_call_id) — an age cutoff here previously made jobs older
        # than 4h permanently invisible: forever "Processing" in the UI while
        # their results sat in S3.
        processing_jobs = list(Job.objects.filter(
            user=request.user,
            status=Job.Status.PROCESSING,
        ).order_by("-started_at")[:200])

        # Re-drive pipeline runs whose jobs already resolved but whose
        # completion notification was missed (restart mid-poll, old cancels).
        try:
            from apps.pipelines import engine as pipeline_engine
            pipeline_engine.reconcile_user_runs(request.user)
        except Exception:
            logger.exception("pipeline run reconciliation failed")

        try:
            poll_annotate_jobs(request.user)
        except Exception:
            logger.exception("annotate poll failed")

        if not processing_jobs:
            return JsonResponse({"checked": 0, "completed": 0, "jobs": []})

        completed = _poll_sagemaker_results(processing_jobs)
        # Post-poll status per video so a page can flip rows live (Processing hub).
        pks = [j.pk for j in processing_jobs]
        jobs = [{"job_id": s["pk"], "video_id": s["video_id"], "status": s["status"]}
                for s in Job.objects.filter(pk__in=pks).values("pk", "video_id", "status")]
        return JsonResponse({
            "checked": len(processing_jobs),
            "completed": completed,
            "jobs": jobs,
        })


# SageMaker async caps: 1h max processing + 6h request TTL. Anything still
# unresolved after this many hours can never complete — fail it out.
_JOB_TIMEOUT_HOURS = 8
# Grace period for the spawn thread before we consider a job "never spawned".
_SPAWN_GRACE_MINUTES = 10


def _handle_unspawned_job(job, age) -> int:
    """A PROCESSING job with no SageMaker request (empty modal_call_id): the
    spawn thread died before invoking — typically an app restart/deploy between
    Job creation and the invoke. Re-spawn once; fail if that doesn't take."""
    from datetime import timedelta
    from django.utils import timezone

    if age < timedelta(minutes=_SPAWN_GRACE_MINUTES):
        return 0  # spawn may still be in flight
    cfg = dict(job.config or {})
    if not cfg.get("respawn_attempted"):
        cfg["respawn_attempted"] = True
        Job.objects.filter(pk=job.pk).update(config=cfg)
        logger.warning("Job %s has no SageMaker request after %s — re-spawning", job.pk, age)
        spawn_gpu_job_async(job.pk)
        return 0
    Job.objects.filter(pk=job.pk).update(
        status="failed",
        error_message="Never reached the GPU (the server restarted during submission, "
                      "and one re-spawn attempt also failed). Re-run the analysis.",
        completed_at=timezone.now(),
    )
    _notify_pipeline(job.pk)
    return 1


def _poll_sagemaker_results(jobs) -> int:
    """Check each job's expected output S3 URI. Update DB on completion.

    Returns the number of jobs newly marked completed/failed in this call.
    """
    import boto3
    import json as _json
    from datetime import timedelta
    from urllib.parse import urlparse
    from django.utils import timezone
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
    now = timezone.now()
    n = 0

    for job in jobs:
        age = now - (job.started_at or now)
        if not job.modal_call_id:
            n += _handle_unspawned_job(job, age)
            continue
        # Chunked long videos: N invocations converge into one merged result.
        if (job.config or {}).get("chunks"):
            try:
                n += _poll_chunked_job(job, s3, age)
            except Exception as e:  # noqa: BLE001 - never kill the whole poll
                logger.warning("chunk poll failed for job %s: %s", job.pk, e)
            continue
        try:
            parsed = urlparse(job.modal_call_id)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")

            # Check for success first.
            try:
                body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
                result = _json.loads(body)
                _apply_result_to_job(job, result)
                n += 1
                continue
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in {"NoSuchKey", "404", "NotFound"}:
                    raise

            # Then platform-level failure (container crash, request expiry).
            # The real location comes from the invoke response (needs
            # S3FailurePath on the endpoint); the .failure guess is the legacy
            # fallback for jobs spawned before that existed.
            failure_uri = (job.config or {}).get("failure_location", "")
            if failure_uri:
                fparsed = urlparse(failure_uri)
                fbucket, fkey = fparsed.netloc, fparsed.path.lstrip("/")
            else:
                fbucket = bucket
                fkey = key.replace(".out", ".failure") if key.endswith(".out") else key + ".failure"
            try:
                body = s3.get_object(Bucket=fbucket, Key=fkey)["Body"].read()
                Job.objects.filter(pk=job.pk).update(
                    status="failed",
                    error_message=f"SageMaker inference failed: {body.decode('utf-8', errors='replace')[:500]}",
                    completed_at=timezone.now(),
                )
                _notify_pipeline(job.pk)
                n += 1
                continue
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in {"NoSuchKey", "404", "NotFound"}:
                    raise

            # Neither output nor failure exists. Past the async platform caps
            # (1h processing + 6h queue TTL) it can never complete — fail out
            # instead of showing "Processing" forever.
            if age > timedelta(hours=_JOB_TIMEOUT_HOURS):
                Job.objects.filter(pk=job.pk).update(
                    status="failed",
                    error_message=f"Timed out: no result after {_JOB_TIMEOUT_HOURS}h "
                                  "(the GPU request expired or was lost). Re-run the analysis.",
                    completed_at=timezone.now(),
                )
                _notify_pipeline(job.pk)
                n += 1
        except Exception as e:
            logger.error("Poll error for job %s (%s): %s", job.pk, job.modal_call_id, e)

    return n


def poll_annotate_jobs(user) -> int:
    """Resolve on-demand annotated-video renders (job.config['annotate']).

    Independent of the main analysis lifecycle — the Job is already completed;
    this only fills in annotated_video_path once the render output lands.
    Idempotent; safe to call from the poller and the reconciler.
    """
    import boto3
    import json as _json
    from datetime import timedelta
    from urllib.parse import urlparse
    from django.utils import timezone
    from botocore.exceptions import ClientError

    pending = list(Job.objects.filter(
        user=user, config__annotate__status="processing",
    )[:50])
    if not pending:
        return 0

    s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
    now = timezone.now()
    n = 0
    for job in pending:
        ann = dict((job.config or {}).get("annotate") or {})
        uri = ann.get("output_uri", "")
        if not uri:
            continue
        parsed = urlparse(uri)
        try:
            body = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"].read()
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code not in {"NoSuchKey", "404", "NotFound"}:
                logger.warning("annotate poll error job %s: %s", job.pk, e)
                continue
            # Not ready — give up after 1h (annotate is minutes of work).
            started = ann.get("started_at")
            try:
                age = now - timezone.datetime.fromisoformat(started) if started else timedelta(0)
            except (ValueError, TypeError):
                age = timedelta(0)
            if age > timedelta(hours=1):
                ann["status"] = "failed"
                cfg = dict(job.config or {}); cfg["annotate"] = ann
                Job.objects.filter(pk=job.pk).update(config=cfg)
                n += 1
            continue

        result = _json.loads(body)
        cfg = dict(job.config or {})
        if result.get("status") == "completed" and result.get("annotated_video_path"):
            JobResult.objects.filter(job_id=job.pk).update(
                annotated_video_path=result["annotated_video_path"])
            ann["status"] = "done"
        else:
            ann["status"] = "failed"
            ann["error"] = str(result.get("error_message", ""))[:300]
        cfg["annotate"] = ann
        Job.objects.filter(pk=job.pk).update(config=cfg)
        n += 1
    return n


def _poll_chunked_job(job, s3, age) -> int:
    """Advance one chunked job: collect landed chunk outputs, fail fast on any
    chunk failure, merge + complete when the last chunk lands. Returns 1 when
    the job reached a terminal state this pass."""
    import json as _json
    from datetime import timedelta
    from urllib.parse import urlparse
    from botocore.exceptions import ClientError
    from django.utils import timezone

    def _fail(msg):
        Job.objects.filter(pk=job.pk).update(
            status="failed", error_message=msg[:500], completed_at=timezone.now())
        _notify_pipeline(job.pk)
        return 1

    cfg = job.config or {}
    chunks = cfg["chunks"]
    changed = False
    for ch in chunks:
        if ch.get("result") is not None:
            continue
        p = urlparse(ch["output_uri"])
        try:
            body = s3.get_object(Bucket=p.netloc, Key=p.path.lstrip("/"))["Body"].read()
            res = _json.loads(body)
            if res.get("status") == "failed":
                return _fail(f"Chunk {ch['i']} failed: {res.get('error_message', 'unknown')}")
            ch["result"] = res
            changed = True
            continue
        except ClientError as e:
            if e.response.get("Error", {}).get("Code", "") not in {"NoSuchKey", "404", "NotFound"}:
                raise
        # No output yet — platform-level failure for this chunk?
        if ch.get("failure_uri"):
            fp = urlparse(ch["failure_uri"])
            try:
                body = s3.get_object(Bucket=fp.netloc, Key=fp.path.lstrip("/"))["Body"].read()
                return _fail(f"Chunk {ch['i']} — SageMaker inference failed: "
                             f"{body.decode('utf-8', errors='replace')}")
            except ClientError as e:
                if e.response.get("Error", {}).get("Code", "") not in {"NoSuchKey", "404", "NotFound"}:
                    raise
    if changed:
        Job.objects.filter(pk=job.pk).update(config=cfg)

    if all(ch.get("result") is not None for ch in chunks):
        merged = _merge_chunk_results(job, chunks)
        _apply_result_to_job(job, merged)
        return 1
    if age > timedelta(hours=_JOB_TIMEOUT_HOURS):
        pending = sum(1 for ch in chunks if ch.get("result") is None)
        return _fail(f"Timed out: {pending}/{len(chunks)} chunks never returned after "
                     f"{_JOB_TIMEOUT_HOURS}h. Re-run the analysis.")
    return 0


def _remap_chunk_track_id(tid, chunk_i):
    """Namespace a chunk-local track id so ids can't collide across chunks
    (each chunk's tracker restarts numbering). Numeric ids below 1M stay
    numeric via a per-chunk offset; anything else gets a prefixed string,
    keeping the scheme collision-free in all cases."""
    try:
        tid_i = int(float(tid))
    except (TypeError, ValueError):
        return f"{chunk_i}_{tid}"
    if 0 <= tid_i < 1_000_000:
        return str(chunk_i * 1_000_000 + tid_i)
    return f"{chunk_i}_{tid_i}"


def _merge_chunk_results(job, chunks) -> dict:
    """One result dict from N chunk results: concatenated events/tracking CSVs
    (frame numbers are already absolute; track ids get namespaced) uploaded to
    the job's canonical keys, plus summed counts. Annotated video, per-track
    crops and interactions aren't merged (documented in summary_stats) —
    day-level foraging trips are recomputed from the merged events anyway."""
    import csv as _csv
    import io

    from config.storage import get_s3_client

    s3c = get_s3_client()
    uid, mid = str(job.user_id), job.modal_job_id
    results = [ch["result"] for ch in chunks]

    def _read_rows(path):
        if not path:
            return []
        buf = io.BytesIO()
        try:
            s3c.download_to_stream("processed", path, buf)
        except Exception as e:  # noqa: BLE001 - a chunk with no rows is valid
            logger.warning("chunk csv read failed (%s): %s", path, e)
            return []
        return list(_csv.DictReader(io.StringIO(buf.getvalue().decode("utf-8", "replace"))))

    merged_paths = {}
    for kind, key in (("events_csv_path", f"{uid}/{mid}/events.csv"),
                      ("tracking_csv_path", f"{uid}/{mid}/tracking.csv")):
        fieldnames, all_rows = None, []
        for i, res in enumerate(results):
            rows = _read_rows(res.get(kind))
            if rows and fieldnames is None:
                fieldnames = list(rows[0].keys())
            for row in rows:
                if "track_id" in row:
                    row["track_id"] = _remap_chunk_track_id(row.get("track_id"), i)
                all_rows.append(row)
        if fieldnames:
            out = io.StringIO()
            w = _csv.DictWriter(out, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
            s3c.upload_stream("processed", key,
                              io.BytesIO(out.getvalue().encode("utf-8")),
                              content_type="text/csv")
            merged_paths[kind] = key
        else:
            merged_paths[kind] = ""

    def _sum(k):
        return sum(int(r.get(k) or 0) for r in results)

    stats0 = (results[0].get("summary_stats") or {}) if results else {}
    return {
        "status": "completed",
        "events_csv_path": merged_paths["events_csv_path"],
        "tracking_csv_path": merged_paths["tracking_csv_path"],
        "foraging_trips_csv_path": "",
        "interactions_csv_path": "",
        "crops_csv_path": "",
        "annotated_video_path": "",
        "total_events": _sum("total_events"),
        "entry_count": _sum("entry_count"),
        "exit_count": _sum("exit_count"),
        "unique_tracks": _sum("unique_tracks"),
        "nest_count": max((int(r.get("nest_count") or 0) for r in results), default=0),
        "foraging_trip_count": _sum("foraging_trip_count"),
        "avg_trip_duration_sec": None,
        "interaction_count": _sum("interaction_count"),
        "summary_stats": {
            "video_fps": stats0.get("video_fps"),
            "chunked": len(chunks),
            "note": "merged from chunked invocations; annotated video, "
                    "interactions and per-track crops are unavailable for chunked runs",
        },
        "execution_seconds": round(
            sum(float(r.get("execution_seconds") or 0) for r in results), 2),
    }


def _apply_result_to_job(job, result: dict) -> None:
    """Write the SageMaker result JSON into JobResult + finalize the Job."""
    from django.utils import timezone

    if result.get("status") == "failed":
        Job.objects.filter(pk=job.pk).update(
            status="failed",
            error_message=result.get("error_message", "Unknown SM failure"),
            completed_at=timezone.now(),
        )
        _notify_pipeline(job.pk)
        return

    JobResult.objects.update_or_create(
        job_id=job.pk,
        defaults={
            "events_csv_path": result.get("events_csv_path", ""),
            "tracking_csv_path": result.get("tracking_csv_path", ""),
            "foraging_trips_csv_path": result.get("foraging_trips_csv_path", ""),
            "interactions_csv_path": result.get("interactions_csv_path", ""),
            "crops_csv_path": result.get("crops_csv_path", ""),
            "annotated_video_path": result.get("annotated_video_path", ""),
            "total_events": result.get("total_events", 0),
            "entry_count": result.get("entry_count", 0),
            "exit_count": result.get("exit_count", 0),
            "unique_tracks": result.get("unique_tracks", 0),
            "nest_count": result.get("nest_count", 0),
            "foraging_trip_count": result.get("foraging_trip_count", 0),
            "avg_trip_duration_sec": result.get("avg_trip_duration_sec"),
            "interaction_count": result.get("interaction_count", 0),
            "summary_stats": result.get("summary_stats", {}),
        },
    )

    exec_secs = result.get("execution_seconds", 0) or 0
    credits_used = int(exec_secs)
    cost_rate = GPU_TIERS.get(job.gpu_tier, {}).get("cost_per_sec", 0.000306)
    cost_usd = round(exec_secs * cost_rate, 4)

    Job.objects.filter(pk=job.pk).update(
        status="completed", progress_pct=100,
        completed_at=timezone.now(),
        execution_seconds=exec_secs,
        compute_cost_usd=cost_usd,
    )

    try:
        from apps.accounts.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=job.user)
        profile.charge(credits_used, gpu_seconds=exec_secs)
    except Exception as e:
        logger.error("Failed to charge credits for job %s: %s", job.pk, e)

    logger.info("Poll: Job %s completed — %s events, %.1fs, $%.4f",
                job.pk, result.get("total_events", 0), exec_secs, cost_usd)

    # Flag the day's foraging summary as stale (cheap upsert — the background
    # sweep does the actual recompute). Never raises.
    from apps.analysis import foraging
    foraging.mark_stale_for_job(job)

    _notify_pipeline(job.pk)


def _notify_pipeline(job_pk) -> None:
    """If a finished Job belongs to a pipeline run, advance that run.

    Re-fetches the job fresh (callers use ``.update()``, leaving stale in-memory
    objects) and hands it to the pipelines engine. Never raises into the poller.
    """
    try:
        from apps.pipelines.engine import on_job_finished
        fresh = Job.objects.filter(pk=job_pk).first()
        if fresh and (fresh.config or {}).get("pipeline_run_id"):
            on_job_finished(fresh)
    except Exception:
        logger.exception("pipeline notify failed for job %s", job_pk)


def _video_job_config(base: dict, video, use_device_roi: bool) -> dict:
    """Per-video job config. When use_device_roi is on and the video's DEVICE has a
    configured hotel ROI / nest tubes, inject them so the run uses the human-set
    layout (the nest-detection model becomes a backup for videos/devices without
    one). No-op when the device has nothing set. The SageMaker worker honours
    ``hotel_roi`` / ``nest_layout`` when present."""
    cfg = dict(base)
    dev = getattr(video, "device", None)
    if use_device_roi and dev is not None:
        if dev.roi_override:
            cfg["hotel_roi"] = dev.roi_override     # normalized [x1,y1,x2,y2]
        if dev.nest_layout:
            cfg["nest_layout"] = dev.nest_layout    # [{id, box:[x1,y1,x2,y2]}, ...]
    return cfg


class BatchJobView(LoginRequiredMixin, View):
    """Submit analysis jobs for multiple videos via Modal parallel processing.

    Uses Modal's starmap() for true GPU parallelism — each video gets its own
    A10G container. No Django threads needed. Modal handles queueing and scaling.
    """

    def post(self, request):
        import threading
        from django.http import JsonResponse
        from django.utils import timezone as tz
        from .models import compute_config_hash

        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"

        def _fail(msg, status=400, level="error"):
            """Uniform error: JSON for AJAX (stay on the Processing page), else
            the classic message + redirect."""
            if is_ajax:
                return JsonResponse({"error": msg}, status=status)
            getattr(messages, level)(request, msg)
            return redirect("analysis:processing")

        video_ids = request.POST.getlist("video_ids")

        # Read config from form
        detection_mode = request.POST.get("detection_mode", "yolo")
        confidence = float(request.POST.get("confidence_threshold", 0.25))
        two_mode = request.POST.get("two_mode_tracking", "true") == "true"
        visualize = request.POST.get("visualize", "true") == "true"
        gpu_tier = request.POST.get("gpu_tier", "A10G")
        use_device_roi = request.POST.get("use_device_roi") in ("on", "true", "1")

        # Tracking runs the whole clip; the new Processing form marks itself
        # with analyses_form so an unchecked run_tracking = off.
        if request.POST.get("analyses_form"):
            run_tracking = request.POST.get("run_tracking") in ("on", "true", "1")
        else:
            run_tracking = True
        if not run_tracking:
            return _fail("Enable tracking to run analysis.", level="warning")

        config = {
            "detection_mode": detection_mode,
            "confidence_threshold": confidence,
            "two_mode_tracking": two_mode,
            "visualize": visualize,
            "run_tracking": run_tracking,
        }

        # Resolve custom model paths (separate for nest and bee)
        from apps.training.models import CustomModel
        custom_nest_model_path = ""
        custom_bee_model_path = ""
        for key, config_key in [("custom_nest_model", "custom_nest_model_path"), ("custom_bee_model", "custom_bee_model_path")]:
            model_id = request.POST.get(key, "")
            if model_id:
                try:
                    cm = CustomModel.objects.get(pk=model_id, user=request.user, is_active=True)
                    config[config_key] = cm.storage_key
                    if key == "custom_nest_model":
                        custom_nest_model_path = cm.storage_key
                    else:
                        custom_bee_model_path = cm.storage_key
                except CustomModel.DoesNotExist:
                    pass

        # Quota check
        from apps.accounts.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user=request.user)

        # Estimate credits (1 credit ≈ 1 GPU-second)
        est_credits_per_video = 349  # avg GPU-seconds per video

        # Build queryset — owner or device-manager (viewers excluded).
        if video_ids:
            videos = Video.manageable(request.user).filter(pk__in=video_ids, status=Video.Status.READY)
        else:
            return _fail("No videos selected.", level="warning")

        # Deduplication: check which videos already have completed jobs with same config
        skipped = 0
        videos_to_process = []
        video_configs = {}  # per-video config (may carry the device's ROI/nest tubes)
        for video in videos.select_related("source", "device"):
            vcfg = _video_job_config(config, video, use_device_roi)
            video_configs[video.pk] = vcfg
            config_hash = compute_config_hash(video.pk, vcfg)

            # Check for existing completed or processing job with same hash
            existing = Job.objects.filter(
                user=request.user,
                video=video,
                config_hash=config_hash,
                status__in=[Job.Status.COMPLETED, Job.Status.PROCESSING, Job.Status.QUEUED],
            ).exists()

            if existing:
                skipped += 1
            else:
                videos_to_process.append(video)

        if not videos_to_process:
            msg = f"All {skipped} video(s) already analyzed with this configuration."
            if is_ajax:
                return JsonResponse({"ok": True, "submitted": 0, "skipped": skipped,
                                     "message": msg, "video_ids": []})
            messages.info(request, msg)
            return redirect("analysis:processing")

        # Budget check
        total_est_credits = est_credits_per_video * len(videos_to_process)
        if not profile.has_budget(total_est_credits):
            return _fail(
                f"Insufficient credits. Need ~{total_est_credits:,} credits for "
                f"{len(videos_to_process)} videos but only {profile.remaining_credits:,} "
                "remaining this month.", status=402)

        # Concurrent job check — only block if already at GPU capacity
        active_count = Job.objects.filter(
            user=request.user, status=Job.Status.PROCESSING,
        ).count()
        if active_count >= profile.max_concurrent_jobs:
            return _fail(
                f"All {profile.max_concurrent_jobs} GPU slots are in use. Wait for "
                "some to complete before submitting more.", status=429)

        # Create Job records
        jobs_data = []
        for video in videos_to_process:
            modal_job_id = f"modal_{uuid.uuid4().hex[:12]}"
            job = Job.objects.create(
                user=request.user,
                video=video,
                config=video_configs[video.pk],
                config_hash=compute_config_hash(video.pk, video_configs[video.pk]),
                gpu_tier=gpu_tier,
                status=Job.Status.PROCESSING,
                started_at=tz.now(),
                modal_job_id=modal_job_id,
            )
            jobs_data.append({
                "job_pk": job.pk,
                "job_id": modal_job_id,
                "user_id": str(request.user.pk),
                "video": video,
                "config": video_configs[video.pk],  # carries hotel_roi/nest_layout
            })

        # Spawn all jobs on Modal
        thread = threading.Thread(
            target=_spawn_gpu_batch,
            args=(jobs_data, detection_mode, confidence, custom_nest_model_path, custom_bee_model_path),
            daemon=True,
        )
        thread.start()

        msg = f"Submitted {len(jobs_data)} video(s) on {gpu_tier} GPU."
        if skipped:
            msg += f" Skipped {skipped} already analyzed."
        est_total_credits = len(jobs_data) * est_credits_per_video
        msg += f" Estimated: ~{est_total_credits:,} credits"
        if is_ajax:
            return JsonResponse({
                "ok": True, "submitted": len(jobs_data), "skipped": skipped,
                "message": msg,
                "video_ids": [v.pk for v in videos_to_process],  # mark these processing
            })
        messages.success(request, msg)
        return redirect("analysis:processing")


class DownloadSpeciesCSVView(LoginRequiredMixin, View):
    """Export species/taxonomy observations (BioCLIP) as CSV — the second kind of
    processing output. One row per activity that has a best taxon. Optional
    ?device= filter (matches the Processing-hub filter)."""

    def get(self, request):
        import csv
        from django.http import HttpResponse
        from apps.devices.models import Device
        from apps.monitor.models import Activity

        qs = (Activity.objects
              .filter(device__in=Device.accessible(request.user), best_taxon__isnull=False)
              .select_related("device", "best_taxon", "video").order_by("-started_at"))
        device = request.GET.get("device", "")
        if device:
            qs = qs.filter(device_id=device)
        confirmed = request.GET.get("confirmed", "")
        if confirmed == "yes":
            qs = qs.filter(video__metadata__bee_confirmed=True)
        elif confirmed == "no":
            qs = qs.filter(video__metadata__bee_confirmed=False)

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="beemonitor_species.csv"'
        w = csv.writer(response)
        w.writerow(["bee_hotel", "started_at", "taxon", "common_name", "rank",
                    "confidence", "individuals", "video"])
        for a in qs[:10000]:
            t = a.best_taxon
            w.writerow([
                a.device.name if a.device else "",
                a.started_at.isoformat() if a.started_at else "",
                t.name, t.common_name, t.rank,
                a.best_confidence, a.individual_count,
                a.video.title if a.video_id else "",
            ])
        return response


class JobResultsView(LoginRequiredMixin, TemplateView):
    template_name = "analysis/results.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        job = get_object_or_404(Job, pk=self.kwargs["pk"], user=self.request.user)
        ctx["job"] = job

        try:
            result = job.result
            ctx["result"] = result
        except JobResult.DoesNotExist:
            ctx["result"] = None
            return ctx

        # Build paths — use DB values or construct from modal_job_id as fallback
        user_id = str(job.user.pk)
        modal_id = job.modal_job_id or ""
        prefix = f"{user_id}/{modal_id}"

        events_path = result.events_csv_path or (f"{prefix}/events.csv" if modal_id else "")
        tracking_path = result.tracking_csv_path or (f"{prefix}/tracking_results.csv" if modal_id else "")
        interactions_path = result.interactions_csv_path or (f"{prefix}/interactions.csv" if modal_id else "")
        annotated_path = result.annotated_video_path or (f"{prefix}/annotated_video.mp4" if modal_id else "")

        # Generate SAS URLs for viewing/downloading
        if events_path:
            ctx["events_csv_url"] = _generate_presigned_url(events_path)
        if tracking_path:
            ctx["tracking_csv_url"] = _generate_presigned_url(tracking_path)
        if interactions_path:
            ctx["interactions_csv_url"] = _generate_presigned_url(interactions_path)
        if annotated_path:
            ctx["annotated_video_url"] = _generate_presigned_url(annotated_path)

        # Original video SAS URL
        if job.video.storage_key:
            ctx["original_video_url"] = _generate_presigned_url(
                job.video.storage_key, container="raw-videos"
            )

        foraging_path = result.foraging_trips_csv_path or ""
        if foraging_path:
            ctx["foraging_csv_url"] = _generate_presigned_url(foraging_path)

        # Load CSV data for display in tables
        ctx["events_data"] = _load_csv_from_storage(events_path)
        ctx["tracking_data"] = _load_csv_from_storage(tracking_path)
        ctx["interactions_data"] = _load_csv_from_storage(interactions_path)

        # Data-driven stat tiles: core tracking counts always, plus derived
        # analyses only when they ran (so the page reflects the actual pipeline).
        stats = result.summary_stats or {}
        tiles = [
            {"label": "Unique Tracks", "value": result.unique_tracks, "color": "text-blue-600"},
            {"label": "Total Events", "value": result.total_events, "color": "text-gray-900"},
            {"label": "Entries", "value": result.entry_count, "color": "text-green-600"},
            {"label": "Exits", "value": result.exit_count, "color": "text-red-600"},
        ]
        if result.nest_count or stats.get("nest_bboxes"):
            n = result.nest_count or len(stats.get("nest_bboxes") or {})
            tiles.append({"label": "Nests", "value": n, "color": "text-amber-600"})
        if result.foraging_trip_count:
            tiles.append({"label": "Foraging Trips", "value": result.foraging_trip_count,
                          "color": "text-amber-700"})
        if result.interaction_count:
            tiles.append({"label": "Interactions", "value": result.interaction_count,
                          "color": "text-purple-600"})
        ctx["stat_tiles"] = tiles

        # Per-track crops (for later species ID): presign each track's crop keys.
        manifest = stats.get("crops_manifest") or {}
        track_crops = []
        total_crops = 0
        if manifest:
            for track_id in sorted(manifest, key=lambda t: int(t) if str(t).isdigit() else t):
                keys = manifest[track_id] or []
                thumbs = [u for k in keys if (u := _generate_presigned_url(k))]
                if thumbs:
                    track_crops.append({"track_id": track_id, "crops": thumbs})
                    total_crops += len(thumbs)
        ctx["track_crops"] = track_crops
        ctx["total_crops"] = total_crops
        if result.crops_csv_path:
            ctx["crops_csv_url"] = _generate_presigned_url(result.crops_csv_path)

        # Annotated-video generation state (post-analysis, on demand).
        ann = (job.config or {}).get("annotate") or {}
        ctx["annotate_status"] = ann.get("status", "")
        ctx["can_annotate"] = bool(tracking_path) and ann.get("status") != "processing"

        return ctx


class GenerateAnnotatedVideoView(LoginRequiredMixin, View):
    """Render the annotated overlay video for a finished job — a separate
    endpoint invocation streaming from the tracking CSV, so it can't OOM or
    stall the analysis itself (see inference._annotate_video)."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.utils import timezone

        job = get_object_or_404(Job, pk=pk, user=request.user)
        try:
            result = job.result
        except JobResult.DoesNotExist:
            result = None
        tracking_path = getattr(result, "tracking_csv_path", "") if result else ""
        if not tracking_path:
            messages.warning(request, "This job has no tracking data to annotate.")
            return redirect("analysis:results", pk=pk)

        stats = (getattr(result, "summary_stats", {}) or {})
        annotate_job_id = f"annot_{uuid.uuid4().hex[:12]}"
        payload = {
            "task": "annotate_video",
            "job_id": annotate_job_id,
            "user_id": str(job.user_id),
            "video_blob_path": job.video.storage_key,
            "tracking_csv_path": tracking_path,
            "nest_bboxes": stats.get("nest_bboxes") or {},
            "hotel_bbox": stats.get("hotel_bbox"),
        }
        try:
            input_uri = _put_inference_payload(annotate_job_id, payload)
            output_uri, _failure_uri = _invoke_endpoint_async(annotate_job_id, input_uri)
        except Exception as e:
            logger.exception("annotate spawn failed for job %s", pk)
            messages.error(request, f"Could not start annotation: {e}")
            return redirect("analysis:results", pk=pk)

        cfg = dict(job.config or {})
        cfg["annotate"] = {"status": "processing", "output_uri": output_uri,
                           "started_at": timezone.now().isoformat()}
        Job.objects.filter(pk=pk).update(config=cfg)
        messages.info(request, "Rendering the annotated video — this runs in the "
                               "background; refresh in a couple of minutes.")
        return redirect("analysis:results", pk=pk)


class _FilteredJobsMixin:
    """Shared logic for filtering completed jobs by site/year/month/day/hour."""

    def _get_filtered_results(self, request):
        site = request.GET.get("site", "")
        year = request.GET.get("year", "")
        month = request.GET.get("month", "")
        day = request.GET.get("day", "")
        hour = request.GET.get("hour", "")
        device = request.GET.get("device", "")
        confirmed = request.GET.get("confirmed", "")

        # Ecological data follows video access: results for any video the user
        # owns OR can see via a device share (mirrors Video.accessible).
        qs = JobResult.objects.filter(
            job__video__in=Video.accessible(request.user),
            job__status=Job.Status.COMPLETED,
        ).select_related("job__video")

        if device:
            qs = qs.filter(job__video__device_id=device)
        if confirmed == "yes":
            qs = qs.filter(job__video__metadata__bee_confirmed=True)
        elif confirmed == "no":
            qs = qs.filter(job__video__metadata__bee_confirmed=False)
        if site:
            qs = qs.filter(job__video__site_name=_unsanitize_site(site))
        if year:
            try:
                qs = qs.filter(job__video__year=int(year))
            except (ValueError, TypeError):
                pass
        if month:
            try:
                qs = qs.filter(job__video__month=int(month))
            except (ValueError, TypeError):
                pass
        if day:
            try:
                qs = qs.filter(job__video__day=int(day))
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                qs = qs.filter(job__video__hour=int(hour))
            except (ValueError, TypeError):
                pass

        label_parts = []
        if device:
            label_parts.append(f"device{device}")
        if site:
            label_parts.append(site)
        if year:
            label_parts.append(str(year))
        if month:
            label_parts.append(f"month{month}")
        if day:
            label_parts.append(f"day{day}")
        if hour:
            label_parts.append(f"hour{hour}")
        label = "_".join(label_parts) if label_parts else "all"

        return qs, label


class DownloadEventsCSVView(_FilteredJobsMixin, LoginRequiredMixin, View):
    """Download combined events CSV for all filtered completed jobs."""

    def get(self, request):
        import csv
        import io
        from django.http import HttpResponse

        results, label = self._get_filtered_results(request)

        if not results.exists():
            from django.contrib import messages as msg
            msg.warning(request, "No completed jobs matching this filter.")
            return redirect("analysis:processing")

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="beemonitor_events_{label}.csv"'

        writer = None
        s3 = get_s3_client()

        for result in results:
            path = result.events_csv_path
            if not path:
                # Try constructing from modal_job_id
                mid = result.job.modal_job_id
                uid = str(result.job.user_id)
                if mid:
                    path = f"{uid}/{mid}/events.csv"
                else:
                    continue

            try:
                buf = io.BytesIO()
                s3.download_to_stream("processed", path, buf)
                content = buf.getvalue().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                headers = next(reader, [])

                if writer is None:
                    all_headers = ["video_title", "site_name", "recorded_at"] + headers
                    writer = csv.writer(response)
                    writer.writerow(all_headers)

                video = result.job.video
                prefix = [
                    video.title,
                    video.site_name,
                    video.recorded_at.isoformat() if video.recorded_at else "",
                ]
                for row in reader:
                    writer.writerow(prefix + row)
            except Exception as e:
                logger.error("Failed to read events CSV %s: %s", path, e)

        if writer is None:
            writer = csv.writer(response)
            writer.writerow(["No event data found for this filter"])

        return response


class DownloadTrackingCSVView(_FilteredJobsMixin, LoginRequiredMixin, View):
    """Download combined tracking CSV for all filtered completed jobs."""

    def get(self, request):
        import csv
        import io
        from django.http import HttpResponse

        results, label = self._get_filtered_results(request)

        if not results.exists():
            from django.contrib import messages as msg
            msg.warning(request, "No completed jobs matching this filter.")
            return redirect("analysis:processing")

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="beemonitor_tracking_{label}.csv"'

        writer = None
        s3 = get_s3_client()

        for result in results:
            path = result.tracking_csv_path
            if not path:
                mid = result.job.modal_job_id
                uid = str(result.job.user_id)
                if mid:
                    path = f"{uid}/{mid}/tracking_results.csv"
                else:
                    continue

            try:
                buf = io.BytesIO()
                s3.download_to_stream("processed", path, buf)
                content = buf.getvalue().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                headers = next(reader, [])

                if writer is None:
                    all_headers = ["video_title", "site_name", "recorded_at"] + headers
                    writer = csv.writer(response)
                    writer.writerow(all_headers)

                video = result.job.video
                prefix = [
                    video.title,
                    video.site_name,
                    video.recorded_at.isoformat() if video.recorded_at else "",
                ]
                for row in reader:
                    writer.writerow(prefix + row)
            except Exception as e:
                logger.error("Failed to read tracking CSV %s: %s", path, e)

        if writer is None:
            writer = csv.writer(response)
            writer.writerow(["No tracking data found for this filter"])

        return response


class DownloadTripsCSVView(_FilteredJobsMixin, LoginRequiredMixin, View):
    """Download combined foraging trips CSV for all filtered completed jobs."""

    def get(self, request):
        import csv
        import io
        from django.http import HttpResponse

        results, label = self._get_filtered_results(request)

        if not results.exists():
            from django.contrib import messages as msg
            msg.warning(request, "No completed jobs matching this filter.")
            return redirect("analysis:processing")

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="beemonitor_foraging_trips_{label}.csv"'

        writer = None
        s3 = get_s3_client()

        for result in results:
            path = result.foraging_trips_csv_path
            if not path:
                mid = result.job.modal_job_id
                uid = str(result.job.user_id)
                if mid:
                    path = f"{uid}/{mid}/foraging_trips.csv"
                else:
                    continue

            try:
                buf = io.BytesIO()
                s3.download_to_stream("processed", path, buf)
                content = buf.getvalue().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                headers = next(reader, [])

                if writer is None:
                    all_headers = ["video_title", "site_name", "recorded_at"] + headers
                    writer = csv.writer(response)
                    writer.writerow(all_headers)

                video = result.job.video
                prefix = [
                    video.title,
                    video.site_name,
                    video.recorded_at.isoformat() if video.recorded_at else "",
                ]
                for row in reader:
                    writer.writerow(prefix + row)
            except Exception as e:
                logger.error("Failed to read foraging trips CSV %s: %s", path, e)

        if writer is None:
            writer = csv.writer(response)
            writer.writerow(["No foraging trip data found for this filter"])

        return response


class DownloadInteractionsCSVView(_FilteredJobsMixin, LoginRequiredMixin, View):
    """Download combined interactions CSV for all filtered completed jobs."""

    def get(self, request):
        import csv
        import io
        from django.http import HttpResponse

        results, label = self._get_filtered_results(request)

        if not results.exists():
            from django.contrib import messages as msg
            msg.warning(request, "No completed jobs matching this filter.")
            return redirect("analysis:processing")

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="beemonitor_interactions_{label}.csv"'

        writer = None
        s3 = get_s3_client()

        for result in results:
            path = result.interactions_csv_path
            if not path:
                mid = result.job.modal_job_id
                uid = str(result.job.user_id)
                if mid:
                    path = f"{uid}/{mid}/interactions.csv"
                else:
                    continue

            try:
                buf = io.BytesIO()
                s3.download_to_stream("processed", path, buf)
                content = buf.getvalue().decode("utf-8")
                reader = csv.reader(io.StringIO(content))
                headers = next(reader, [])

                if writer is None:
                    all_headers = ["video_title", "site_name", "recorded_at"] + headers
                    writer = csv.writer(response)
                    writer.writerow(all_headers)

                video = result.job.video
                prefix = [
                    video.title,
                    video.site_name,
                    video.recorded_at.isoformat() if video.recorded_at else "",
                ]
                for row in reader:
                    writer.writerow(prefix + row)
            except Exception as e:
                logger.error("Failed to read interactions CSV %s: %s", path, e)

        if writer is None:
            writer = csv.writer(response)
            writer.writerow(["No interaction data found for this filter"])

        return response


class DownloadNestDataCSVView(_FilteredJobsMixin, LoginRequiredMixin, View):
    """Download per-video nest bounding-box coordinates CSV for all filtered completed jobs."""

    def get(self, request):
        import csv

        from django.http import HttpResponse

        results, label = self._get_filtered_results(request)

        if not results.exists():
            from django.contrib import messages as msg

            msg.warning(request, "No completed jobs matching this filter.")
            return redirect("analysis:processing")

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = (
            f'attachment; filename="beemonitor_nest_bboxes_{label}.csv"'
        )

        writer = csv.writer(response)
        writer.writerow([
            "video_title", "site_name", "recorded_at",
            "nest_id", "x1", "y1", "x2", "y2",
        ])

        for result in results:
            video = result.job.video
            prefix = [
                video.title,
                video.site_name,
                video.recorded_at.isoformat() if video.recorded_at else "",
            ]
            stats = result.summary_stats or {}
            bboxes = stats.get("nest_bboxes", {})

            if isinstance(bboxes, dict) and bboxes:
                for nest_id in sorted(bboxes.keys(), key=lambda k: int(k) if k.isdigit() else k):
                    coords = bboxes[nest_id]
                    if isinstance(coords, (list, tuple)) and len(coords) == 4:
                        writer.writerow(prefix + [nest_id] + [c for c in coords])
                    else:
                        writer.writerow(prefix + [nest_id, "", "", "", ""])

        return response


class AnalyticsDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "analysis/analytics.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user

        # Get filter params
        site = self.request.GET.get("site", "")
        db_site = _unsanitize_site(site)  # reverse-map display name for DB queries
        year = self.request.GET.get("year", "")
        month = self.request.GET.get("month", "")
        day = self.request.GET.get("day", "")
        hour = self.request.GET.get("hour", "")
        device = self.request.GET.get("device", "")

        year_int = None
        month_int = None
        day_int = None
        hour_int = None
        device_int = None
        if year:
            try:
                year_int = int(year)
            except (ValueError, TypeError):
                pass
        if month:
            try:
                month_int = int(month)
            except (ValueError, TypeError):
                pass
        if day:
            try:
                day_int = int(day)
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                hour_int = int(hour)
            except (ValueError, TypeError):
                pass
        if device:
            try:
                device_int = int(device)
            except (ValueError, TypeError):
                pass

        # Filter options (use set() to guarantee uniqueness)
        user_videos = Video.accessible(user)
        ctx["devices"] = [
            {"id": r["device_id"], "name": r["device__name"]}
            for r in user_videos.exclude(device=None)
            .values("device_id", "device__name").distinct().order_by("device__name")
        ]
        ctx["site_names"] = sorted(set(
            _sanitize_site(s) for s in
            user_videos.exclude(site_name="")
            .values_list("site_name", flat=True)
        ))
        ctx["years"] = sorted(set(
            user_videos.exclude(year=None)
            .values_list("year", flat=True)
        ))
        ctx["months_list"] = list(range(1, 13))
        ctx["days_list"] = sorted(set(
            user_videos.exclude(day=None).values_list("day", flat=True)
        ))
        ctx["hours_list"] = list(range(24))

        ctx["current_site"] = site
        ctx["current_year"] = year
        ctx["current_month"] = month
        ctx["current_day"] = day
        ctx["current_hour"] = hour
        ctx["current_device"] = device

        # Job progress metrics
        user_jobs = Job.objects.filter(user=user)
        processing_jobs = user_jobs.filter(status=Job.Status.PROCESSING).select_related("video")
        failed_jobs = user_jobs.filter(status=Job.Status.FAILED).select_related("video").order_by("-created_at")[:5]
        completed_recent = user_jobs.filter(status=Job.Status.COMPLETED).order_by("-completed_at")[:5]

        ctx["job_stats"] = {
            "total": user_jobs.count(),
            "completed": user_jobs.filter(status=Job.Status.COMPLETED).count(),
            "processing": processing_jobs.count(),
            "failed": user_jobs.filter(status=Job.Status.FAILED).count(),
            "queued": user_jobs.filter(status=Job.Status.QUEUED).count(),
        }
        ctx["processing_jobs"] = processing_jobs[:20]
        ctx["failed_jobs"] = failed_jobs
        ctx["completed_recent"] = completed_recent

        # Analytics data — wrap in try/except so page never crashes
        try:
            ctx["summary"] = get_summary_stats(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
        except Exception as e:
            logger.error("Analytics summary failed: %s", e, exc_info=True)
            ctx["summary"] = {"total_videos": 0, "total_events": 0, "total_entries": 0,
                              "total_exits": 0, "avg_events_per_video": 0, "total_unique_tracks": 0, "completed_jobs": 0}

        try:
            activity = get_activity_over_time(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
            ctx["activity_json"] = json.dumps(activity, default=str)
        except Exception:
            ctx["activity_json"] = "[]"

        try:
            cumulative = get_cumulative_activity(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
            ctx["cumulative_json"] = json.dumps(cumulative, default=str)
        except Exception:
            ctx["cumulative_json"] = "[]"

        try:
            averages = get_period_averages(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
            # Convert int keys to string keys for JSON
            ctx["hourly_avg_json"] = json.dumps({str(k): v for k, v in averages["hourly"].items()})
            ctx["daily_avg_json"] = json.dumps({str(k): v for k, v in averages["daily"].items()})
            ctx["monthly_avg_json"] = json.dumps({str(k): v for k, v in averages["monthly"].items()})
        except Exception:
            ctx["hourly_avg_json"] = "{}"
            ctx["daily_avg_json"] = "{}"
            ctx["monthly_avg_json"] = "{}"

        try:
            vb = get_video_breakdown(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
            for row in vb:
                row["title"] = _sanitize_site(row["title"])
                row["site"] = _sanitize_site(row["site"])
            ctx["video_breakdown"] = vb
        except Exception:
            ctx["video_breakdown"] = []

        try:
            foraging_over_time = get_foraging_trips_over_time(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
            ctx["foraging_trips_json"] = json.dumps(foraging_over_time, default=str)
        except Exception:
            ctx["foraging_trips_json"] = "[]"

        try:
            trips_per_nest = get_trips_per_nest(user, site_name=db_site or None, year=year_int, month=month_int, day=day_int, hour=hour_int, device=device_int)
            ctx["trips_per_nest_json"] = json.dumps(trips_per_nest, default=str)
        except Exception:
            ctx["trips_per_nest_json"] = "[]"

        # Fetch weather data for the date range of foraging trips
        try:
            foraging_over_time = json.loads(ctx.get("foraging_trips_json", "[]"))
            if foraging_over_time:
                dates = [d["date"] for d in foraging_over_time]
                weather = _fetch_weather_data(dates[0], dates[-1])
                ctx["weather_hourly_json"] = json.dumps(weather.get("hourly", []), default=str)
                ctx["weather_daily_json"] = json.dumps(weather.get("daily", []), default=str)
            else:
                ctx["weather_hourly_json"] = "[]"
                ctx["weather_daily_json"] = "[]"
        except Exception:
            ctx["weather_hourly_json"] = "[]"
            ctx["weather_daily_json"] = "[]"

        return ctx


def _fetch_weather_data(start_date: str, end_date: str, lat: float = 40.79, lon: float = -77.86) -> dict:
    """Fetch historical hourly + daily weather from Open-Meteo (free, no API key).

    Returns dict with 'hourly' and 'daily' lists.
    """
    import urllib.request

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=auto"
    )

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        hourly = data.get("hourly", {})
        hourly_times = hourly.get("time", [])
        hourly_result = []
        for i, t in enumerate(hourly_times):
            hourly_result.append({
                "time": t,
                "temp_c": hourly.get("temperature_2m", [None])[i],
                "precipitation_mm": hourly.get("precipitation", [0])[i] or 0,
                "humidity_pct": hourly.get("relative_humidity_2m", [None])[i],
                "wind_kmh": hourly.get("wind_speed_10m", [None])[i],
            })

        daily = data.get("daily", {})
        daily_times = daily.get("time", [])
        daily_result = []
        for i, d in enumerate(daily_times):
            daily_result.append({
                "date": d,
                "temp_max": daily.get("temperature_2m_max", [None])[i],
                "temp_min": daily.get("temperature_2m_min", [None])[i],
                "precipitation_mm": daily.get("precipitation_sum", [0])[i] or 0,
            })

        return {"hourly": hourly_result, "daily": daily_result}
    except Exception as e:
        logger.warning("Weather fetch failed: %s", e)
        return {"hourly": [], "daily": []}


def _load_csv_from_storage(blob_path: str, container: str = "processed") -> dict:
    """Download a CSV from S3 and return headers + rows for template rendering."""
    if not blob_path:
        return {"headers": [], "rows": []}
    try:
        import csv
        import io

        buf = io.BytesIO()
        get_s3_client().download_to_stream(container, blob_path, buf)
        content = buf.getvalue().decode("utf-8")

        reader = csv.reader(io.StringIO(content))
        headers = next(reader, [])
        rows = list(reader)

        return {"headers": headers, "rows": rows, "total": len(rows)}
    except Exception as e:
        logger.error("Failed to load CSV %s/%s: %s", container, blob_path, e)
        return {"headers": [], "rows": []}
