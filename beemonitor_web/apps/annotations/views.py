import io
import json
import zipfile

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView, DetailView, ListView, TemplateView

from .forms import ProjectCreateForm
from .models import Annotation, AnnotationProject


class ProjectListView(LoginRequiredMixin, ListView):
    model = AnnotationProject
    template_name = "annotations/list.html"
    context_object_name = "projects"
    paginate_by = 20

    def get_queryset(self):
        return AnnotationProject.objects.filter(user=self.request.user)


class ProjectCreateView(LoginRequiredMixin, CreateView):
    model = AnnotationProject
    form_class = ProjectCreateForm
    template_name = "annotations/create.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse_lazy("annotations:detail", kwargs={"pk": self.object.pk})


class ProjectDetailView(LoginRequiredMixin, DetailView):
    model = AnnotationProject
    template_name = "annotations/detail.html"
    context_object_name = "project"

    def get_queryset(self):
        return AnnotationProject.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        project = self.object
        videos = project.videos.all()
        video_data = []
        for video in videos:
            ann_count = Annotation.objects.filter(project=project, video=video).count()
            video_data.append({"video": video, "annotation_count": ann_count})
        ctx["video_data"] = video_data
        ctx["total_annotations"] = project.annotations.count()
        return ctx


class AnnotationEditorView(LoginRequiredMixin, TemplateView):
    template_name = "annotations/editor.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        project = get_object_or_404(
            AnnotationProject, pk=self.kwargs["pk"], user=self.request.user
        )
        video_id = self.request.GET.get("video")
        frame_number = int(self.request.GET.get("frame", 0))

        video = None
        boxes = []
        if video_id:
            video = get_object_or_404(project.videos, pk=video_id)
            try:
                annotation = Annotation.objects.get(
                    project=project, video=video, frame_number=frame_number
                )
                boxes = annotation.boxes
            except Annotation.DoesNotExist:
                boxes = []

        ctx["project"] = project
        ctx["video"] = video
        ctx["frame_number"] = frame_number
        ctx["boxes"] = json.dumps(boxes)
        ctx["classes"] = json.dumps(project.classes)
        ctx["videos"] = project.videos.all()
        return ctx


class SaveAnnotationView(LoginRequiredMixin, View):
    def post(self, request, pk):
        project = get_object_or_404(
            AnnotationProject, pk=pk, user=request.user
        )
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        video_id = data.get("video_id")
        frame_number = data.get("frame_number")
        boxes = data.get("boxes", [])

        if video_id is None or frame_number is None:
            return JsonResponse({"error": "video_id and frame_number are required"}, status=400)

        video = get_object_or_404(project.videos, pk=video_id)

        annotation, created = Annotation.objects.update_or_create(
            project=project,
            video=video,
            frame_number=frame_number,
            defaults={"boxes": boxes},
        )

        return JsonResponse({
            "success": True,
            "created": created,
            "annotation_id": annotation.pk,
        })


class ExportProjectView(LoginRequiredMixin, View):
    def get(self, request, pk):
        project = get_object_or_404(
            AnnotationProject, pk=pk, user=request.user
        )
        annotations = project.annotations.select_related("video").all()

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write data.yaml
            class_lines = "\n".join(
                f"  {i}: {cls}" for i, cls in enumerate(project.classes)
            )
            data_yaml = (
                f"train: ./images/train\n"
                f"val: ./images/val\n"
                f"nc: {len(project.classes)}\n"
                f"names:\n{class_lines}\n"
            )
            zf.writestr("data.yaml", data_yaml)

            # Write label files
            for ann in annotations:
                yolo_txt = ann.to_yolo_format()
                filename = f"labels/{ann.video.title}_frame{ann.frame_number:06d}.txt"
                zf.writestr(filename, yolo_txt)

        buf.seek(0)
        response = HttpResponse(buf.read(), content_type="application/zip")
        response["Content-Disposition"] = (
            f'attachment; filename="{project.name}_yolo_dataset.zip"'
        )
        return response
