from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import DetailView, FormView, ListView

from .forms import VideoUploadForm
from .models import Video


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
        Video.objects.create(
            user=self.request.user,
            title=form.cleaned_data["title"],
            azure_blob_path=f"uploads/{self.request.user.pk}/{video_file.name}",
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
