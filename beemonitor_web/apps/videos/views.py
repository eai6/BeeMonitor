import logging
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import DetailView, FormView, ListView

from .forms import VideoUploadForm
from .models import Video

logger = logging.getLogger(__name__)


class VideoListView(LoginRequiredMixin, ListView):
    template_name = "videos/list.html"
    context_object_name = "videos"
    paginate_by = 20

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)


class VideoUploadView(LoginRequiredMixin, FormView):
    template_name = "videos/upload.html"
    form_class = VideoUploadForm
    success_url = reverse_lazy("videos:list")

    def form_valid(self, form):
        video_file = self.request.FILES["video_file"]
        upload_id = uuid.uuid4().hex[:12]
        blob_path = f"{self.request.user.pk}/{upload_id}/{video_file.name}"

        # Upload to Azure Blob Storage
        try:
            from azure.storage.blob import BlobServiceClient

            conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
            if conn_str:
                service = BlobServiceClient.from_connection_string(conn_str)
                blob = service.get_blob_client("raw-videos", blob_path)
                blob.upload_blob(video_file, overwrite=True)
                logger.info("Uploaded %s to Azure Blob Storage", blob_path)
            else:
                logger.warning("No Azure connection string — file metadata saved but not uploaded")
        except Exception as e:
            logger.error("Azure upload failed: %s", e)
            messages.warning(self.request, f"Video metadata saved but upload failed: {e}")

        # Get video duration/resolution if possible
        Video.objects.create(
            user=self.request.user,
            title=form.cleaned_data["title"],
            azure_blob_path=blob_path,
            file_size_bytes=video_file.size,
            status=Video.Status.READY,
        )
        messages.success(self.request, "Video uploaded successfully.")
        return super().form_valid(form)


class VideoDetailView(LoginRequiredMixin, DetailView):
    template_name = "videos/detail.html"
    context_object_name = "video"

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)
