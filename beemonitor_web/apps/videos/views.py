import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import DetailView, FormView, ListView

from .forms import VideoBatchUploadForm, VideoUploadForm
from .models import Video

logger = logging.getLogger(__name__)


class VideoListView(LoginRequiredMixin, ListView):
    template_name = "videos/list.html"
    context_object_name = "videos"
    paginate_by = 20

    def get_queryset(self):
        qs = Video.objects.filter(user=self.request.user)

        # Apply filters from query params
        site = self.request.GET.get("site")
        year = self.request.GET.get("year")
        month = self.request.GET.get("month")
        day = self.request.GET.get("day")

        if site:
            qs = qs.filter(site_name=site)
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

        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user_videos = Video.objects.filter(user=self.request.user)

        # Build filter options from existing data (use set() to guarantee uniqueness)
        ctx["site_names"] = sorted(set(
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

        # Preserve current filter selections
        ctx["current_site"] = self.request.GET.get("site", "")
        ctx["current_year"] = self.request.GET.get("year", "")
        ctx["current_month"] = self.request.GET.get("month", "")
        ctx["current_day"] = self.request.GET.get("day", "")

        # Custom models for config panel
        try:
            from apps.training.models import CustomModel
            ctx["custom_nest_models"] = CustomModel.objects.filter(
                user=self.request.user, model_type="nest_detection", is_active=True
            )
            ctx["custom_bee_models"] = CustomModel.objects.filter(
                user=self.request.user, model_type="bee_tracking", is_active=True
            )
        except Exception:
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
