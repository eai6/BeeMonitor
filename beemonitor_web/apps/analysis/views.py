import json
import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.views import View
from django.views.generic import DetailView, FormView, ListView, TemplateView

from apps.videos.models import Video

from .analytics import (
    get_activity_over_time,
    get_cumulative_activity,
    get_nest_activity_heatmap,
    get_period_averages,
    get_summary_stats,
)
from .forms import JobCreateForm
from .models import Job, JobResult

logger = logging.getLogger(__name__)


def _generate_sas_url(blob_path: str, container: str = "processed") -> str:
    """Generate a time-limited SAS URL for a blob in Azure Storage."""
    try:
        from datetime import datetime, timedelta, timezone
        from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return ""

        service = BlobServiceClient.from_connection_string(conn_str)
        account_name = service.account_name

        # Extract account key from connection string
        account_key = ""
        for part in conn_str.split(";"):
            if part.startswith("AccountKey="):
                account_key = part.split("=", 1)[1]
                break

        token = generate_blob_sas(
            account_name=account_name,
            container_name=container,
            blob_name=blob_path,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=24),
        )
        return f"https://{account_name}.blob.core.windows.net/{container}/{blob_path}?{token}"
    except Exception as e:
        logger.error("Failed to generate SAS URL for %s: %s", blob_path, e)
        return ""


def _transfer_s3_to_azure(video) -> str:
    """If video is on S3, transfer it to Azure Blob Storage. Returns the Azure blob path."""
    blob_path = video.azure_blob_path
    if not blob_path.startswith("s3://"):
        return blob_path  # Already in Azure

    import boto3
    from azure.storage.blob import BlobServiceClient
    from django.conf import settings as django_settings
    import tempfile
    from pathlib import Path

    meta = video.metadata or {}
    remote_key = meta.get("remote_key", blob_path.replace("s3://", ""))
    bucket = meta.get("remote_bucket", "")

    if not bucket:
        raise ValueError("No S3 bucket in video metadata")

    # Get S3 credentials from the source
    source = video.source
    if not source:
        raise ValueError("Video has no linked data source for S3 credentials")

    from apps.sources.views import _decrypt_credentials
    creds = _decrypt_credentials(source)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=creds["access_key_id"],
        aws_secret_access_key=creds["secret_access_key"],
        region_name=creds.get("region", "us-east-1"),
    )

    conn_str = django_settings.AZURE_STORAGE_CONNECTION_STRING
    azure_service = BlobServiceClient.from_connection_string(conn_str)

    # Download from S3 to temp file, then upload to Azure
    filename = remote_key.split("/")[-1]
    azure_blob_path = f"{video.user_id}/{video.pk}/{filename}"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        logger.info("Transferring s3://%s/%s -> Azure raw-videos/%s", bucket, remote_key, azure_blob_path)
        s3.download_file(bucket, remote_key, tmp.name)
        file_size = Path(tmp.name).stat().st_size

        blob = azure_service.get_blob_client("raw-videos", azure_blob_path)
        with open(tmp.name, "rb") as fh:
            blob.upload_blob(fh, overwrite=True)

    # Update video record with Azure path and file size
    video.azure_blob_path = azure_blob_path
    video.file_size_bytes = file_size
    video.save(update_fields=["azure_blob_path", "file_size_bytes"])
    creds.clear()

    logger.info("Transfer complete: %s (%d MB)", azure_blob_path, file_size // (1024 * 1024))
    return azure_blob_path


def _run_modal_job(job_pk: int) -> None:
    """Background thread: call Modal and update job status on completion."""
    import django
    django.setup()
    from django.utils import timezone

    try:
        job = Job.objects.select_related("video", "video__source").get(pk=job_pk)
    except Job.DoesNotExist:
        return

    try:
        # If video is on S3, transfer to Azure first
        video_blob_path = _transfer_s3_to_azure(job.video)

        import modal
        process_video = modal.Function.from_name("beemonitor-cloud", "process_video")
        result_payload = process_video.remote(
            job_id=job.modal_job_id,
            user_id=str(job.user_id),
            video_blob_path=video_blob_path,
            detection_mode=job.config.get("detection_mode", "yolo"),
            confidence_threshold=job.config.get("confidence_threshold", 0.25),
            visualize=True,
        )

        JobResult.objects.update_or_create(
            job=job,
            defaults={
                "events_csv_path": result_payload.get("events_csv_path", ""),
                "tracking_csv_path": result_payload.get("tracking_csv_path", ""),
                "annotated_video_path": result_payload.get("annotated_video_path", ""),
                "total_events": result_payload.get("total_events", 0),
                "entry_count": result_payload.get("entry_count", 0),
                "exit_count": result_payload.get("exit_count", 0),
                "unique_tracks": result_payload.get("unique_tracks", 0),
                "nest_count": result_payload.get("nest_count", 0),
                "summary_stats": result_payload.get("summary_stats", {}),
            },
        )
        job.status = Job.Status.COMPLETED
        job.progress_pct = 100
        job.completed_at = timezone.now()
        job.save(update_fields=["status", "progress_pct", "completed_at"])
        logger.info("Job %s completed — %s events", job_pk, result_payload.get("total_events", 0))

    except Exception as e:
        job.status = Job.Status.FAILED
        job.error_message = str(e)
        job.save(update_fields=["status", "error_message"])
        logger.exception("Job %s failed", job_pk)


def _run_modal_batch(jobs_data: list, detection_mode: str, confidence: float) -> None:
    """Background thread: submit batch to Modal for parallel GPU processing."""
    import django
    django.setup()
    from django.utils import timezone

    try:
        import modal
        batch_process = modal.Function.from_name("beemonitor-cloud", "batch_process")

        # Build Modal job configs
        modal_jobs = []
        s3_creds_cache = {}  # source_id -> decrypted creds

        for jd in jobs_data:
            video = jd["video"]
            blob_path = video.azure_blob_path

            if blob_path.startswith("s3://"):
                # S3 video — need credentials for transfer
                meta = video.metadata or {}
                source = video.source
                source_id = source.pk if source else None

                if source_id and source_id not in s3_creds_cache:
                    try:
                        from apps.sources.views import _decrypt_credentials
                        s3_creds_cache[source_id] = _decrypt_credentials(source)
                    except Exception as e:
                        logger.error("Failed to decrypt creds for source %s: %s", source_id, e)
                        Job.objects.filter(pk=jd["job_pk"]).update(
                            status=Job.Status.FAILED,
                            error_message=f"Credential error: {e}",
                        )
                        continue

                creds = s3_creds_cache.get(source_id, {})
                modal_jobs.append({
                    "job_id": jd["job_id"],
                    "user_id": jd["user_id"],
                    "s3_bucket": meta.get("remote_bucket", creds.get("bucket", "")),
                    "s3_key": meta.get("remote_key", ""),
                    "s3_access_key_id": creds.get("access_key_id", ""),
                    "s3_secret_access_key": creds.get("secret_access_key", ""),
                    "s3_region": creds.get("region", "us-east-1"),
                    "detection_mode": detection_mode,
                    "confidence_threshold": confidence,
                    "visualize": True,
                })
            else:
                # Azure video — already in blob storage
                modal_jobs.append({
                    "job_id": jd["job_id"],
                    "user_id": jd["user_id"],
                    "video_blob_path": blob_path,
                    "detection_mode": detection_mode,
                    "confidence_threshold": confidence,
                    "visualize": True,
                })

        if not modal_jobs:
            return

        # Call Modal batch_process — this uses starmap() for parallel GPU containers
        logger.info("Submitting %d videos to Modal batch_process", len(modal_jobs))
        results = batch_process.remote(modal_jobs)

        # Update Django Job records with results
        result_by_job_id = {r.get("job_id"): r for r in results if isinstance(r, dict)}

        for jd in jobs_data:
            result = result_by_job_id.get(jd["job_id"])
            if result:
                try:
                    job = Job.objects.get(pk=jd["job_pk"])

                    # Update video blob path if transferred from S3
                    if result.get("azure_blob_path"):
                        video = jd["video"]
                        video.azure_blob_path = result["azure_blob_path"]
                        video.file_size_bytes = result.get("file_size", 0) or video.file_size_bytes
                        video.save(update_fields=["azure_blob_path", "file_size_bytes"])

                    JobResult.objects.update_or_create(
                        job=job,
                        defaults={
                            "events_csv_path": result.get("events_csv_path", ""),
                            "tracking_csv_path": result.get("tracking_csv_path", ""),
                            "annotated_video_path": result.get("annotated_video_path", ""),
                            "total_events": result.get("total_events", 0),
                            "entry_count": result.get("entry_count", 0),
                            "exit_count": result.get("exit_count", 0),
                            "unique_tracks": result.get("unique_tracks", 0),
                            "nest_count": result.get("nest_count", 0),
                            "summary_stats": result.get("summary_stats", {}),
                        },
                    )
                    job.status = Job.Status.COMPLETED
                    job.progress_pct = 100
                    job.completed_at = timezone.now()
                    job.save(update_fields=["status", "progress_pct", "completed_at"])
                except Exception as e:
                    logger.error("Failed to save result for job %s: %s", jd["job_pk"], e)
            else:
                Job.objects.filter(pk=jd["job_pk"]).update(
                    status=Job.Status.FAILED,
                    error_message="No result returned from Modal",
                )

        # Clear cached credentials
        for creds in s3_creds_cache.values():
            creds.clear()

        logger.info("Batch complete: %d/%d results", len(result_by_job_id), len(jobs_data))

    except Exception as e:
        logger.exception("Batch processing failed: %s", e)
        job_pks = [jd["job_pk"] for jd in jobs_data]
        Job.objects.filter(pk__in=job_pks, status=Job.Status.PROCESSING).update(
            status=Job.Status.FAILED,
            error_message=f"Batch error: {e}",
        )


class JobListView(LoginRequiredMixin, ListView):
    template_name = "analysis/list.html"
    context_object_name = "jobs"
    paginate_by = 20

    def get_queryset(self):
        return Job.objects.filter(user=self.request.user).select_related("video")


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

        job = Job.objects.create(
            user=self.request.user,
            video=form.cleaned_data["video"],
            config={
                "detection_mode": form.cleaned_data["detection_mode"],
                "confidence_threshold": form.cleaned_data["confidence_threshold"],
            },
            status=Job.Status.PROCESSING,
            started_at=timezone.now(),
            modal_job_id=f"modal_{uuid.uuid4().hex[:12]}",
        )

        # Run Modal processing in background thread — don't block the request
        thread = threading.Thread(
            target=_run_modal_job,
            args=(job.pk,),
            daemon=True,
        )
        thread.start()

        messages.info(self.request, f"Job #{job.pk} submitted — processing on GPU. This page auto-refreshes.")
        self._job_pk = job.pk
        return super().form_valid(form)

    def get_success_url(self):
        return reverse("analysis:detail", kwargs={"pk": self._job_pk})


class BatchJobView(LoginRequiredMixin, View):
    """Submit analysis jobs for multiple videos via Modal parallel processing.

    Uses Modal's starmap() for true GPU parallelism — each video gets its own
    A10G container. No Django threads needed. Modal handles queueing and scaling.
    """

    def post(self, request):
        import threading
        from django.utils import timezone as tz

        video_ids = request.POST.getlist("video_ids")
        site = request.POST.get("site", "")
        year = request.POST.get("year", "")
        month = request.POST.get("month", "")
        day = request.POST.get("day", "")
        detection_mode = request.POST.get("detection_mode", "yolo")
        confidence = float(request.POST.get("confidence_threshold", 0.25))

        # Build queryset from filters or explicit IDs
        if video_ids:
            videos = Video.objects.filter(user=request.user, pk__in=video_ids, status=Video.Status.READY)
        else:
            videos = Video.objects.filter(user=request.user, status=Video.Status.READY)
            if site:
                videos = videos.filter(site_name=site)
            if year:
                try:
                    videos = videos.filter(year=int(year))
                except (ValueError, TypeError):
                    pass
            if month:
                try:
                    videos = videos.filter(month=int(month))
                except (ValueError, TypeError):
                    pass
            if day:
                try:
                    videos = videos.filter(day=int(day))
                except (ValueError, TypeError):
                    pass

        # Skip videos that already have a running or completed job
        existing_video_ids = set(
            Job.objects.filter(
                user=request.user,
                status__in=[Job.Status.QUEUED, Job.Status.PROCESSING, Job.Status.COMPLETED],
            ).values_list("video_id", flat=True)
        )

        videos_to_process = [v for v in videos.select_related("source") if v.pk not in existing_video_ids]

        if not videos_to_process:
            messages.warning(request, "No new videos to analyze (all already have jobs).")
            return redirect("analysis:list")

        # Create Job records for all videos
        jobs_data = []
        for video in videos_to_process:
            modal_job_id = f"modal_{uuid.uuid4().hex[:12]}"
            job = Job.objects.create(
                user=request.user,
                video=video,
                config={
                    "detection_mode": detection_mode,
                    "confidence_threshold": confidence,
                },
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

        # Launch batch processing in one background thread
        # This thread calls Modal batch_process which uses starmap() for parallelism
        thread = threading.Thread(
            target=_run_modal_batch,
            args=(jobs_data, detection_mode, confidence),
            daemon=True,
        )
        thread.start()

        messages.success(
            request,
            f"Submitted {len(jobs_data)} video(s) for parallel GPU processing. "
            f"Modal will spin up a separate A10G container for each video.",
        )
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
        annotated_path = result.annotated_video_path or (f"{prefix}/annotated_video.mp4" if modal_id else "")

        # Generate SAS URLs for viewing/downloading
        if events_path:
            ctx["events_csv_url"] = _generate_sas_url(events_path)
        if tracking_path:
            ctx["tracking_csv_url"] = _generate_sas_url(tracking_path)
        if annotated_path:
            ctx["annotated_video_url"] = _generate_sas_url(annotated_path)

        # Original video SAS URL
        if job.video.azure_blob_path:
            ctx["original_video_url"] = _generate_sas_url(
                job.video.azure_blob_path, container="raw-videos"
            )

        # Load CSV data for display in tables
        ctx["events_data"] = _load_csv_from_azure(events_path)
        ctx["tracking_data"] = _load_csv_from_azure(tracking_path)

        return ctx


class AnalyticsDashboardView(LoginRequiredMixin, TemplateView):
    template_name = "analysis/analytics.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user

        # Get filter params
        site = self.request.GET.get("site", "")
        year = self.request.GET.get("year", "")
        month = self.request.GET.get("month", "")

        year_int = None
        month_int = None
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

        # Filter options
        user_videos = Video.objects.filter(user=user)
        ctx["site_names"] = sorted(
            user_videos.exclude(site_name="")
            .values_list("site_name", flat=True)
            .distinct()
        )
        ctx["years"] = sorted(
            user_videos.exclude(year=None)
            .values_list("year", flat=True)
            .distinct()
        )
        ctx["months_list"] = list(range(1, 13))

        ctx["current_site"] = site
        ctx["current_year"] = year
        ctx["current_month"] = month

        # Analytics data — wrap in try/except so page never crashes
        try:
            ctx["summary"] = get_summary_stats(user, site_name=site or None, year=year_int, month=month_int)
        except Exception:
            ctx["summary"] = {"total_videos": 0, "total_events": 0, "total_entries": 0,
                              "total_exits": 0, "avg_events_per_video": 0, "total_unique_tracks": 0, "completed_jobs": 0}

        try:
            activity = get_activity_over_time(user, site_name=site or None, year=year_int, month=month_int)
            ctx["activity_json"] = json.dumps(activity, default=str)
        except Exception:
            ctx["activity_json"] = "[]"

        try:
            cumulative = get_cumulative_activity(user, site_name=site or None)
            ctx["cumulative_json"] = json.dumps(cumulative, default=str)
        except Exception:
            ctx["cumulative_json"] = "[]"

        try:
            averages = get_period_averages(user, site_name=site or None)
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

        return ctx


def _load_csv_from_azure(blob_path: str, container: str = "processed", max_rows: int = 500) -> dict:
    """Download a CSV from Azure and return headers + rows for template rendering."""
    if not blob_path:
        return {"headers": [], "rows": []}
    try:
        import csv
        import io
        from azure.storage.blob import BlobServiceClient

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            return {"headers": [], "rows": []}

        service = BlobServiceClient.from_connection_string(conn_str)
        blob = service.get_blob_client(container, blob_path)
        content = blob.download_blob().readall().decode("utf-8")

        reader = csv.reader(io.StringIO(content))
        headers = next(reader, [])
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

        return {"headers": headers, "rows": rows, "total": len(rows)}
    except Exception as e:
        logger.error("Failed to load CSV %s: %s", blob_path, e)
        return {"headers": [], "rows": []}
