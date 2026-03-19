import io
import json
import zipfile

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView, DetailView, ListView, TemplateView

import logging

from .forms import ProjectCreateForm
from .models import Annotation, AnnotationProject

logger = logging.getLogger(__name__)


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

        # Available videos to add (not already in project)
        from apps.videos.models import Video
        existing_ids = set(videos.values_list("pk", flat=True))
        available = Video.objects.filter(user=self.request.user).exclude(pk__in=existing_ids)
        ctx["available_videos"] = available[:500]  # Cap for performance
        ctx["available_sites"] = sorted(set(
            available.exclude(site_name="").values_list("site_name", flat=True)
        ))
        return ctx


class AddVideosView(LoginRequiredMixin, View):
    """Add selected videos to an annotation project."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        video_ids = request.POST.getlist("video_ids")

        if not video_ids:
            messages.warning(request, "No videos selected.")
            return redirect("annotations:detail", pk=pk)

        from apps.videos.models import Video
        videos = Video.objects.filter(user=request.user, pk__in=video_ids)
        added = 0
        for video in videos:
            if not project.videos.filter(pk=video.pk).exists():
                project.videos.add(video)
                added += 1

        messages.success(request, f"Added {added} video(s) to project.")
        return redirect("annotations:detail", pk=pk)


class AnnotationEditorView(LoginRequiredMixin, TemplateView):
    template_name = "annotations/editor.html"

    def get(self, request, *args, **kwargs):
        """Return JSON for AJAX frame navigation, HTML for normal page load."""
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.GET.get("format") == "json":
            project = get_object_or_404(
                AnnotationProject, pk=self.kwargs["pk"], user=request.user
            )
            video_id = request.GET.get("video")
            frame_number = int(request.GET.get("frame", 0))
            boxes = []
            if video_id:
                try:
                    video = project.videos.get(pk=video_id)
                    ann = Annotation.objects.get(project=project, video=video, frame_number=frame_number)
                    boxes = ann.boxes
                except (Annotation.DoesNotExist, Exception):
                    boxes = []
            return JsonResponse({"boxes": boxes, "frame": frame_number})
        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        logger = logging.getLogger(__name__)

        ctx = super().get_context_data(**kwargs)
        try:
            project = get_object_or_404(
                AnnotationProject, pk=self.kwargs["pk"], user=self.request.user
            )
        except Exception as e:
            logger.error("Editor: project lookup failed: %s", e)
            raise

        video_id = self.request.GET.get("video")
        frame_number = int(self.request.GET.get("frame", 0))

        video = None
        boxes = []
        if video_id:
            try:
                video = project.videos.get(pk=video_id)
            except Exception as e:
                logger.error("Editor: video %s not in project %s: %s", video_id, project.pk, e)
                video = None
            try:
                annotation = Annotation.objects.get(
                    project=project, video=video, frame_number=frame_number
                )
                boxes = annotation.boxes
            except Annotation.DoesNotExist:
                boxes = []

        # Auto-transfer S3 video to Azure if needed
        ctx["transferring"] = False
        if video and video.azure_blob_path.startswith("s3://"):
            try:
                import threading
                from apps.analysis.views import _transfer_s3_to_azure

                # Check if already transferring (avoid duplicate)
                if not getattr(video, '_transfer_started', False):
                    ctx["transferring"] = True
                    # Transfer in background — page will show spinner and auto-reload
                    thread = threading.Thread(
                        target=self._do_transfer,
                        args=(video.pk,),
                        daemon=True,
                    )
                    thread.start()
            except Exception as e:
                logger.error("Auto-transfer failed for video %s: %s", video.pk, e)

        ctx["project"] = project
        ctx["video"] = video
        ctx["frame_number"] = frame_number
        ctx["boxes"] = json.dumps(boxes)
        ctx["classes"] = json.dumps(project.classes)
        ctx["videos"] = project.videos.all()
        ctx["video_url"] = ""

        # Generate video SAS URL for frame display
        if video and video.azure_blob_path and not video.azure_blob_path.startswith("s3://"):
            try:
                from datetime import datetime, timedelta, timezone as dt_tz
                from django.conf import settings
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
                        account_name=account_name, container_name="raw-videos",
                        blob_name=video.azure_blob_path, account_key=account_key,
                        permission=BlobSasPermissions(read=True),
                        expiry=datetime.now(dt_tz.utc) + timedelta(hours=24),
                    )
                    ctx["video_url"] = f"https://{account_name}.blob.core.windows.net/raw-videos/{video.azure_blob_path}?{token}"
            except Exception:
                pass

        return ctx

    @staticmethod
    def _do_transfer(video_pk):
        """Background thread: transfer S3 video to Azure."""
        import django
        django.setup()
        from django.db import connection
        try:
            from apps.videos.models import Video
            from apps.analysis.views import _transfer_s3_to_azure
            video = Video.objects.select_related("source").get(pk=video_pk)
            _transfer_s3_to_azure(video)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Background transfer failed for %s: %s", video_pk, e)
        finally:
            connection.close()


class TransferVideoView(LoginRequiredMixin, View):
    """Transfer a video from S3 to Azure so frames are available for annotation."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        video_id = request.POST.get("video_id")
        frame = request.POST.get("frame", 0)

        from apps.videos.models import Video
        video = get_object_or_404(Video, pk=video_id, user=request.user)

        if not video.azure_blob_path.startswith("s3://"):
            messages.info(request, "Video is already in Azure.")
            return redirect(f"/annotations/{pk}/edit/?video={video_id}&frame={frame}")

        try:
            from apps.analysis.views import _transfer_s3_to_azure
            new_path = _transfer_s3_to_azure(video)
            messages.success(request, f"Video transferred to Azure. Frames now available.")
        except Exception as e:
            logger.error("Transfer failed for video %s: %s", video_id, e)
            messages.error(request, f"Transfer failed: {e}")

        return redirect(f"/annotations/{pk}/edit/?video={video_id}&frame={frame}")


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
