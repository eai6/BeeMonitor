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


class PreAnnotateView(LoginRequiredMixin, View):
    """Run YOLO detection on a video's frames and save as initial annotations."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages
        import threading

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        video_id = request.POST.get("video_id")

        from apps.videos.models import Video
        video = get_object_or_404(Video, pk=video_id, user=request.user)

        blob_path = video.azure_blob_path
        if blob_path.startswith("s3://"):
            messages.error(request, "Video must be transferred to Azure first.")
            return redirect("annotations:detail", pk=pk)

        # Spawn Modal pre-annotation in background
        thread = threading.Thread(
            target=self._run_pre_annotate,
            args=(project.pk, video.pk, blob_path),
            daemon=True,
        )
        thread.start()

        messages.info(request, f"Pre-annotating '{video.title}' with AI. This takes 1-2 minutes. Refresh to see results.")
        return redirect("annotations:detail", pk=pk)

    @staticmethod
    def _run_pre_annotate(project_pk, video_pk, blob_path):
        import django
        django.setup()
        from django.db import connection

        try:
            import modal
            fn = modal.Function.from_name("beemonitor-cloud", "pre_annotate_video")
            result = fn.remote(video_blob_path=blob_path, sample_interval=30, max_frames=200)

            if not result or not result.get("frames"):
                return

            from apps.annotations.models import Annotation, AnnotationProject
            from apps.videos.models import Video

            project = AnnotationProject.objects.get(pk=project_pk)
            video = Video.objects.get(pk=video_pk)
            width = result.get("video_width", 1280)
            height = result.get("video_height", 720)

            created = 0
            for frame_data in result["frames"]:
                _, was_created = Annotation.objects.update_or_create(
                    project=project,
                    video=video,
                    frame_number=frame_data["frame_number"],
                    defaults={
                        "boxes": frame_data["boxes"],
                        "image_width": width,
                        "image_height": height,
                    },
                )
                if was_created:
                    created += 1

            logger.info("Pre-annotated video %s: %d frames, %d new annotations",
                        video_pk, len(result["frames"]), created)
        except Exception as e:
            logger.error("Pre-annotate failed for video %s: %s", video_pk, e, exc_info=True)
        finally:
            connection.close()


class ExportProjectView(LoginRequiredMixin, View):
    """Export YOLO dataset with images extracted from videos."""

    def get(self, request, pk):
        project = get_object_or_404(
            AnnotationProject, pk=pk, user=request.user
        )
        annotations = project.annotations.select_related("video").order_by("video", "frame_number")

        if not annotations.exists():
            from django.contrib import messages
            messages.warning(request, "No annotations to export.")
            return HttpResponse(status=302, headers={"Location": f"/annotations/{pk}/"})

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # Write data.yaml
            class_lines = "\n".join(
                f"  {i}: {cls}" for i, cls in enumerate(project.classes)
            )
            data_yaml = (
                f"path: .\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: {len(project.classes)}\n"
                f"names:\n{class_lines}\n"
            )
            zf.writestr("data.yaml", data_yaml)

            # Group annotations and extract frames from videos
            from django.conf import settings
            conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
            video_cache = {}  # video_pk -> cv2.VideoCapture

            ann_list = list(annotations)
            split_idx = int(len(ann_list) * 0.8)

            for i, ann in enumerate(ann_list):
                split = "train" if i < split_idx else "val"
                base_name = f"{ann.video.title}_f{ann.frame_number:06d}"
                label_name = f"labels/{split}/{base_name}.txt"

                # Write label
                yolo_txt = ann.to_yolo_format()
                zf.writestr(label_name, yolo_txt)

                # Extract frame image from video
                try:
                    if ann.video.pk not in video_cache:
                        blob_path = ann.video.azure_blob_path
                        if blob_path and not blob_path.startswith("s3://") and conn_str:
                            import tempfile, cv2
                            from azure.storage.blob import BlobServiceClient

                            service = BlobServiceClient.from_connection_string(conn_str)
                            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                            blob = service.get_blob_client("raw-videos", blob_path)
                            with open(tmp.name, "wb") as fh:
                                stream = blob.download_blob()
                                stream.readinto(fh)
                            cap = cv2.VideoCapture(tmp.name)
                            video_cache[ann.video.pk] = (cap, tmp.name)

                    if ann.video.pk in video_cache:
                        import cv2
                        cap, _ = video_cache[ann.video.pk]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame_number)
                        ret, frame = cap.read()
                        if ret:
                            _, img_buf = cv2.imencode(".jpg", frame)
                            zf.writestr(f"images/{split}/{base_name}.jpg", img_buf.tobytes())
                except Exception as e:
                    logger.error("Failed to extract frame %d from video %s: %s",
                                ann.frame_number, ann.video.pk, e)

            # Cleanup video captures
            for cap, tmp_path in video_cache.values():
                cap.release()
                import os
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        buf.seek(0)
        response = HttpResponse(buf.read(), content_type="application/zip")
        response["Content-Disposition"] = (
            f'attachment; filename="{project.name}_yolo_dataset.zip"'
        )
        return response
