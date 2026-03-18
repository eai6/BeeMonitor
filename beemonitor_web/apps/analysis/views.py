import json
import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404
from django.urls import reverse, reverse_lazy
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


def _run_modal_job(job_pk: int) -> None:
    """Background thread: call Modal and update job status on completion."""
    import django
    django.setup()
    from django.utils import timezone

    try:
        job = Job.objects.select_related("video").get(pk=job_pk)
    except Job.DoesNotExist:
        return

    try:
        import modal

        process_video = modal.Function.from_name("beemonitor-cloud", "process_video")
        result_payload = process_video.remote(
            job_id=job.modal_job_id,
            user_id=str(job.user_id),
            video_blob_path=job.video.azure_blob_path,
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

        # Analytics data
        ctx["summary"] = get_summary_stats(user, site_name=site or None, year=year_int, month=month_int)

        activity = get_activity_over_time(user, site_name=site or None, year=year_int, month=month_int)
        ctx["activity_json"] = json.dumps(activity)

        cumulative = get_cumulative_activity(user, site_name=site or None)
        ctx["cumulative_json"] = json.dumps(cumulative)

        averages = get_period_averages(user, site_name=site or None)
        ctx["hourly_avg_json"] = json.dumps(averages["hourly"])
        ctx["daily_avg_json"] = json.dumps(averages["daily"])
        ctx["monthly_avg_json"] = json.dumps(averages["monthly"])

        nest_data = get_nest_activity_heatmap(user)
        ctx["nest_data"] = nest_data
        ctx["nest_data_json"] = json.dumps(nest_data)

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
