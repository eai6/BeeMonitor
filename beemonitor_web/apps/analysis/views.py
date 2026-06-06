import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect
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
            "visualize": True,
        }
        if job.config.get("custom_nest_model_path"):
            payload["custom_nest_model_path"] = job.config["custom_nest_model_path"]
        if job.config.get("custom_bee_model_path"):
            payload["custom_bee_model_path"] = job.config["custom_bee_model_path"]

        input_uri = _put_inference_payload(job.modal_job_id, payload)
        output_uri = _invoke_endpoint_async(job.modal_job_id, input_uri)

        Job.objects.filter(pk=job_pk).update(modal_call_id=output_uri)
        logger.info("Job %s spawned on SageMaker: %s", job_pk, output_uri)

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
                    "visualize": True,
                }
                if custom_nest_model_path:
                    payload["custom_nest_model_path"] = custom_nest_model_path
                if custom_bee_model_path:
                    payload["custom_bee_model_path"] = custom_bee_model_path

                input_uri = _put_inference_payload(jd["job_id"], payload)
                output_uri = _invoke_endpoint_async(jd["job_id"], input_uri)
                Job.objects.filter(pk=jd["job_pk"]).update(modal_call_id=output_uri)
                logger.info("Job %s spawned: %s", jd["job_pk"], output_uri)
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


def _invoke_endpoint_async(job_id: str, input_uri: str) -> str:
    """Call invoke_endpoint_async. Returns the s3:// URI where the result will land."""
    endpoint = settings.SAGEMAKER_ENDPOINT_NAME
    if not endpoint:
        raise RuntimeError("SAGEMAKER_ENDPOINT_NAME is not configured")
    response = _sagemaker_runtime().invoke_endpoint_async(
        EndpointName=endpoint,
        InputLocation=input_uri,
        ContentType="application/json",
        InferenceId=job_id,
    )
    # OutputLocation = s3://<output-bucket>/<inference-id>.out  (typical layout)
    return response["OutputLocation"]


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
        from datetime import timedelta
        from django.http import JsonResponse
        from django.utils import timezone

        recent_cutoff = timezone.now() - timedelta(hours=4)
        processing_jobs = list(Job.objects.filter(
            user=request.user,
            status=Job.Status.PROCESSING,
            started_at__gte=recent_cutoff,
        ).exclude(modal_call_id="")[:200])

        if not processing_jobs:
            return JsonResponse({"checked": 0, "completed": 0})

        completed = _poll_sagemaker_results(processing_jobs)
        return JsonResponse({
            "checked": len(processing_jobs),
            "completed": completed,
        })


def _poll_sagemaker_results(jobs) -> int:
    """Check each job's expected output S3 URI. Update DB on completion.

    Returns the number of jobs newly marked completed/failed in this call.
    """
    import boto3
    import json as _json
    from urllib.parse import urlparse
    from django.utils import timezone
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"))
    n = 0

    for job in jobs:
        if not job.modal_call_id:
            continue
        try:
            parsed = urlparse(job.modal_call_id)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            failure_key = key.replace(".out", ".failure") if key.endswith(".out") else key + ".failure"

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

            # Then check for inference failure.
            try:
                body = s3.get_object(Bucket=bucket, Key=failure_key)["Body"].read()
                Job.objects.filter(pk=job.pk).update(
                    status="failed",
                    error_message=f"SageMaker inference failed: {body.decode('utf-8', errors='replace')[:500]}",
                    completed_at=timezone.now(),
                )
                n += 1
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in {"NoSuchKey", "404", "NotFound"}:
                    raise
                # Still running — neither output nor failure object exists yet.
                continue
        except Exception as e:
            logger.error("Poll error for job %s (%s): %s", job.pk, job.modal_call_id, e)

    return n


def _apply_result_to_job(job, result: dict) -> None:
    """Write the SageMaker result JSON into JobResult + finalize the Job."""
    from django.utils import timezone

    if result.get("status") == "failed":
        Job.objects.filter(pk=job.pk).update(
            status="failed",
            error_message=result.get("error_message", "Unknown SM failure"),
            completed_at=timezone.now(),
        )
        return

    JobResult.objects.update_or_create(
        job_id=job.pk,
        defaults={
            "events_csv_path": result.get("events_csv_path", ""),
            "tracking_csv_path": result.get("tracking_csv_path", ""),
            "foraging_trips_csv_path": result.get("foraging_trips_csv_path", ""),
            "interactions_csv_path": result.get("interactions_csv_path", ""),
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


class BatchJobView(LoginRequiredMixin, View):
    """Submit analysis jobs for multiple videos via Modal parallel processing.

    Uses Modal's starmap() for true GPU parallelism — each video gets its own
    A10G container. No Django threads needed. Modal handles queueing and scaling.
    """

    def post(self, request):
        import threading
        from django.utils import timezone as tz
        from .models import compute_config_hash

        video_ids = request.POST.getlist("video_ids")

        # Read config from form
        detection_mode = request.POST.get("detection_mode", "yolo")
        confidence = float(request.POST.get("confidence_threshold", 0.25))
        two_mode = request.POST.get("two_mode_tracking", "true") == "true"
        visualize = request.POST.get("visualize", "true") == "true"
        gpu_tier = request.POST.get("gpu_tier", "A10G")

        config = {
            "detection_mode": detection_mode,
            "confidence_threshold": confidence,
            "two_mode_tracking": two_mode,
            "visualize": visualize,
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

        # Build queryset
        if video_ids:
            videos = Video.objects.filter(user=request.user, pk__in=video_ids, status=Video.Status.READY)
        else:
            messages.warning(request, "No videos selected.")
            return redirect("videos:list")

        # Deduplication: check which videos already have completed jobs with same config
        skipped = 0
        videos_to_process = []
        for video in videos.select_related("source"):
            config_hash = compute_config_hash(video.pk, config)

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
            messages.info(request, msg)
            return redirect("analysis:list")

        # Budget check
        total_est_credits = est_credits_per_video * len(videos_to_process)
        if not profile.has_budget(total_est_credits):
            messages.error(
                request,
                f"Insufficient credits. Need ~{total_est_credits:,} credits for {len(videos_to_process)} videos "
                f"but only {profile.remaining_credits:,} remaining this month."
            )
            return redirect("videos:list")

        # Concurrent job check — only block if already at GPU capacity
        active_count = Job.objects.filter(
            user=request.user, status=Job.Status.PROCESSING,
        ).count()
        if active_count >= profile.max_concurrent_jobs:
            messages.error(
                request,
                f"All {profile.max_concurrent_jobs} GPU slots are in use. "
                f"Wait for some to complete before submitting more."
            )
            return redirect("videos:list")

        # Create Job records
        jobs_data = []
        for video in videos_to_process:
            modal_job_id = f"modal_{uuid.uuid4().hex[:12]}"
            job = Job.objects.create(
                user=request.user,
                video=video,
                config=config,
                config_hash=compute_config_hash(video.pk, config),
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
        messages.success(request, msg)
        return redirect("analysis:list")


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

        # Load CSV data for display in tables
        ctx["events_data"] = _load_csv_from_storage(events_path)
        ctx["tracking_data"] = _load_csv_from_storage(tracking_path)
        ctx["interactions_data"] = _load_csv_from_storage(interactions_path)

        return ctx


class _FilteredJobsMixin:
    """Shared logic for filtering completed jobs by site/year/month/day/hour."""

    def _get_filtered_results(self, request):
        site = request.GET.get("site", "")
        year = request.GET.get("year", "")
        month = request.GET.get("month", "")
        day = request.GET.get("day", "")
        hour = request.GET.get("hour", "")
        device = request.GET.get("device", "")

        qs = JobResult.objects.filter(
            job__user=request.user,
            job__status=Job.Status.COMPLETED,
        ).select_related("job__video")

        if device:
            qs = qs.filter(job__video__device_id=device)
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
            return redirect("analysis:analytics")

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
            return redirect("analysis:analytics")

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
            return redirect("analysis:analytics")

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
            return redirect("analysis:analytics")

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
            return redirect("analysis:analytics")

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
        user_videos = Video.objects.filter(user=user)
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
            nest_data = get_nest_activity_heatmap(user)
            ctx["nest_data"] = nest_data
            ctx["nest_data_json"] = json.dumps(nest_data, default=str)
        except Exception:
            ctx["nest_data"] = []
            ctx["nest_data_json"] = "[]"

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
