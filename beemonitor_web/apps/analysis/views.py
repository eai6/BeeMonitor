import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404
from django.urls import reverse, reverse_lazy
from django.views.generic import DetailView, FormView, ListView, TemplateView

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
        from django.utils import timezone

        job = Job.objects.create(
            user=self.request.user,
            video=form.cleaned_data["video"],
            config={
                "detection_mode": form.cleaned_data["detection_mode"],
                "confidence_threshold": form.cleaned_data["confidence_threshold"],
            },
            status=Job.Status.QUEUED,
        )

        try:
            import modal

            process_video = modal.Function.from_name("beemonitor-cloud", "process_video")
            job.status = Job.Status.PROCESSING
            job.started_at = timezone.now()
            job.modal_job_id = f"modal_{uuid.uuid4().hex[:12]}"
            job.save(update_fields=["status", "started_at", "modal_job_id"])

            result_payload = process_video.remote(
                job_id=job.modal_job_id,
                user_id=str(self.request.user.pk),
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
            messages.success(self.request, f"Job #{job.pk} completed — {result_payload.get('total_events', 0)} events detected.")
        except ImportError:
            messages.info(self.request, f"Job #{job.pk} queued (Modal SDK not installed).")
        except Exception as e:
            job.status = Job.Status.FAILED
            job.error_message = str(e)
            job.save(update_fields=["status", "error_message"])
            messages.error(self.request, f"Job #{job.pk} failed: {e}")

        return super().form_valid(form)

    def get_success_url(self):
        return reverse("analysis:list")


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

        # Generate SAS URLs for viewing/downloading
        if result.events_csv_path:
            ctx["events_csv_url"] = _generate_sas_url(result.events_csv_path)
        if result.tracking_csv_path:
            ctx["tracking_csv_url"] = _generate_sas_url(result.tracking_csv_path)
        if result.annotated_video_path:
            ctx["annotated_video_url"] = _generate_sas_url(result.annotated_video_path)

        # Original video SAS URL
        if job.video.azure_blob_path:
            ctx["original_video_url"] = _generate_sas_url(
                job.video.azure_blob_path, container="raw-videos"
            )

        return ctx
