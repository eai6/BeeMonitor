import csv
import logging
import re
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import DetailView, FormView, ListView

from .forms import VideoBatchUploadForm, VideoUploadForm
from .models import Video

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


class VideoListView(LoginRequiredMixin, ListView):
    template_name = "videos/list.html"
    context_object_name = "videos"
    paginate_by = 20

    def get_queryset(self):
        qs = Video.objects.filter(user=self.request.user)

        # Apply filters from query params
        search = self.request.GET.get("q", "").strip()
        site = self.request.GET.get("site")
        year = self.request.GET.get("year")
        month = self.request.GET.get("month")
        day = self.request.GET.get("day")
        hour = self.request.GET.get("hour")

        if search:
            qs = qs.filter(title__icontains=search)
        if site:
            qs = qs.filter(site_name=_unsanitize_site(site))
        if year:
            try:
                qs = qs.filter(year=int(year))
            except (ValueError, TypeError):
                pass
        if month:
            try:
                qs = qs.filter(month=int(month))
            except (ValueError, TypeError):
                pass
        if day:
            try:
                qs = qs.filter(day=int(day))
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                qs = qs.filter(hour=int(hour))
            except (ValueError, TypeError):
                pass

        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user_videos = Video.objects.filter(user=self.request.user)

        # Build filter options from existing data (use set() to guarantee uniqueness)
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

        # Preserve current filter selections
        ctx["current_search"] = self.request.GET.get("q", "")
        ctx["current_site"] = self.request.GET.get("site", "")
        ctx["current_year"] = self.request.GET.get("year", "")
        ctx["current_month"] = self.request.GET.get("month", "")
        ctx["current_day"] = self.request.GET.get("day", "")
        ctx["current_hour"] = self.request.GET.get("hour", "")

        # All filtered video IDs (for "Select All Filtered" across pages)
        filtered_qs = self.get_queryset()
        ctx["all_video_ids"] = list(filtered_qs.values_list("pk", flat=True))
        ctx["video_count"] = len(ctx["all_video_ids"])

        # Custom models for config panel (split by type for nest/bee dropdowns)
        try:
            from apps.training.models import CustomModel
            all_models = CustomModel.objects.filter(user=self.request.user, is_active=True)
            ctx["custom_models"] = all_models  # all models for generic dropdown
            ctx["custom_nest_models"] = all_models.filter(model_type__in=["nest_detection", "custom"])
            ctx["custom_bee_models"] = all_models.filter(model_type__in=["bee_tracking", "custom"])
        except Exception:
            ctx["custom_models"] = []
            ctx["custom_nest_models"] = []
            ctx["custom_bee_models"] = []

        return ctx


def _upload_to_azure(blob_path, video_file):
    """Upload a file to Azure Blob Storage. Returns True on success."""
    try:
        from azure.storage.blob import BlobServiceClient

        conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
        if conn_str:
            service = BlobServiceClient.from_connection_string(conn_str)
            blob = service.get_blob_client("raw-videos", blob_path)
            blob.upload_blob(video_file, overwrite=True)
            logger.info("Uploaded %s to Azure Blob Storage", blob_path)
            return True
        else:
            logger.warning("No Azure connection string — file metadata saved but not uploaded")
            return False
    except Exception as e:
        logger.error("Azure upload failed: %s", e)
        return False


class VideoUploadView(LoginRequiredMixin, FormView):
    template_name = "videos/upload.html"
    form_class = VideoUploadForm
    success_url = reverse_lazy("videos:list")

    def form_valid(self, form):
        video_file = self.request.FILES["video_file"]
        upload_id = uuid.uuid4().hex[:12]
        blob_path = f"{self.request.user.pk}/{upload_id}/{video_file.name}"

        # Upload to Azure Blob Storage
        if not _upload_to_azure(blob_path, video_file):
            messages.warning(self.request, "Video metadata saved but upload may have failed.")

        # Parse timestamp from filename
        site_name, recorded_at = Video.parse_timestamp_from_filename(video_file.name)

        Video.objects.create(
            user=self.request.user,
            title=form.cleaned_data["title"],
            azure_blob_path=blob_path,
            file_size_bytes=video_file.size,
            status=Video.Status.READY,
            recorded_at=recorded_at,
            site_name=site_name,
        )
        messages.success(self.request, "Video uploaded successfully.")
        return super().form_valid(form)


class VideoBatchUploadView(LoginRequiredMixin, FormView):
    template_name = "videos/batch_upload.html"
    form_class = VideoBatchUploadForm
    success_url = reverse_lazy("videos:list")

    def form_valid(self, form):
        files = self.request.FILES.getlist("video_files")
        site_name_override = form.cleaned_data.get("site_name", "")
        uploaded_count = 0
        failed_count = 0

        for video_file in files:
            upload_id = uuid.uuid4().hex[:12]
            blob_path = f"{self.request.user.pk}/{upload_id}/{video_file.name}"

            # Upload to Azure
            upload_ok = _upload_to_azure(blob_path, video_file)
            if not upload_ok:
                failed_count += 1

            # Parse timestamp from filename
            parsed_site, recorded_at = Video.parse_timestamp_from_filename(video_file.name)
            final_site = site_name_override or parsed_site

            # Use filename as title (strip extension)
            title = video_file.name
            if "." in title:
                title = title.rsplit(".", 1)[0]

            Video.objects.create(
                user=self.request.user,
                title=title,
                azure_blob_path=blob_path,
                file_size_bytes=video_file.size,
                status=Video.Status.READY,
                recorded_at=recorded_at,
                site_name=final_site,
            )
            uploaded_count += 1

        if failed_count:
            messages.warning(
                self.request,
                f"{uploaded_count} video(s) created. {failed_count} upload(s) may have failed.",
            )
        else:
            messages.success(self.request, f"{uploaded_count} video(s) uploaded successfully.")
        return super().form_valid(form)


class VideoDetailView(LoginRequiredMixin, DetailView):
    template_name = "videos/detail.html"
    context_object_name = "video"

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        video = self.object
        blob_path = video.azure_blob_path

        # Generate SAS URL for video playback
        if blob_path and not blob_path.startswith("s3://"):
            try:
                from datetime import datetime, timedelta, timezone
                from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

                conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
                if conn_str:
                    service = BlobServiceClient.from_connection_string(conn_str)
                    account_name = service.account_name
                    account_key = ""
                    for part in conn_str.split(";"):
                        if part.startswith("AccountKey="):
                            account_key = part.split("=", 1)[1]
                            break
                    token = generate_blob_sas(
                        account_name=account_name,
                        container_name="raw-videos",
                        blob_name=blob_path,
                        account_key=account_key,
                        permission=BlobSasPermissions(read=True),
                        expiry=datetime.now(timezone.utc) + timedelta(hours=24),
                    )
                    ctx["video_url"] = f"https://{account_name}.blob.core.windows.net/raw-videos/{blob_path}?{token}"
            except Exception as e:
                logger.error("Failed to generate video SAS URL: %s", e)

        return ctx


def _delete_azure_blobs_for_video(video):
    """Delete all Azure blobs associated with a video and its analysis results."""
    conn_str = getattr(settings, "AZURE_STORAGE_CONNECTION_STRING", "")
    if not conn_str:
        return

    try:
        from azure.storage.blob import BlobServiceClient
        service = BlobServiceClient.from_connection_string(conn_str)

        # Delete raw video blob
        blob_path = video.azure_blob_path
        if blob_path and not blob_path.startswith("s3://"):
            try:
                blob = service.get_blob_client("raw-videos", blob_path)
                blob.delete_blob()
                logger.info("Deleted raw video blob: %s", blob_path)
            except Exception as e:
                logger.warning("Could not delete raw blob %s: %s", blob_path, e)

        # Delete processed blobs (events CSV, tracking CSV, trips CSV, annotated video)
        from apps.analysis.models import JobResult
        for result in JobResult.objects.filter(job__video=video):
            for path in [
                result.events_csv_path,
                result.tracking_csv_path,
                result.foraging_trips_csv_path,
                result.annotated_video_path,
            ]:
                if path:
                    try:
                        blob = service.get_blob_client("processed", path)
                        blob.delete_blob()
                    except Exception:
                        pass
            logger.info("Deleted processed blobs for job %s", result.job_id)

    except Exception as e:
        logger.error("Azure blob cleanup failed for video %s: %s", video.pk, e)


class VideoExportCSVView(LoginRequiredMixin, View):
    """Export filtered video filenames as CSV."""

    def get(self, request):
        qs = Video.objects.filter(user=request.user)

        search = request.GET.get("q", "").strip()
        site = request.GET.get("site")
        year = request.GET.get("year")
        month = request.GET.get("month")
        day = request.GET.get("day")
        hour = request.GET.get("hour")

        if search:
            qs = qs.filter(title__icontains=search)
        if site:
            qs = qs.filter(site_name=_unsanitize_site(site))
        if year:
            try:
                qs = qs.filter(year=int(year))
            except (ValueError, TypeError):
                pass
        if month:
            try:
                qs = qs.filter(month=int(month))
            except (ValueError, TypeError):
                pass
        if day:
            try:
                qs = qs.filter(day=int(day))
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                qs = qs.filter(hour=int(hour))
            except (ValueError, TypeError):
                pass

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="video_filenames.csv"'

        writer = csv.writer(response)
        writer.writerow(["title", "site", "recorded_at", "status", "size_bytes"])
        for v in qs.order_by("-recorded_at"):
            writer.writerow([
                _sanitize_site(v.title),
                _sanitize_site(v.site_name or ""),
                v.recorded_at.isoformat() if v.recorded_at else "",
                v.get_status_display(),
                v.file_size_bytes,
            ])

        return response


class VideoDeleteView(LoginRequiredMixin, View):
    """Delete a single video and its Azure blobs."""

    def post(self, request, pk):
        video = get_object_or_404(Video, pk=pk, user=request.user)
        title = video.title

        _delete_azure_blobs_for_video(video)
        video.delete()  # CASCADE handles Jobs, JobResults, Annotations

        messages.success(request, f"Deleted video: {title}")
        return redirect("videos:list")


class VideoBatchDeleteView(LoginRequiredMixin, View):
    """Delete multiple videos and their Azure blobs."""

    def post(self, request):
        video_ids = request.POST.getlist("video_ids")
        if not video_ids:
            messages.warning(request, "No videos selected.")
            return redirect("videos:list")

        videos = Video.objects.filter(user=request.user, pk__in=video_ids)
        count = videos.count()

        for video in videos:
            _delete_azure_blobs_for_video(video)

        videos.delete()

        messages.success(request, f"Deleted {count} video(s) and their analysis data.")
        return redirect("videos:list")
