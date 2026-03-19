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
        ctx["project_videos"] = videos
        ctx["video_count"] = videos.count()

        # Video data for "no annotations" fallback links
        video_data = []
        for video in videos:
            ann_count = Annotation.objects.filter(project=project, video=video).count()
            video_data.append({"video": video, "annotation_count": ann_count})
        ctx["video_data"] = video_data

        # Available videos to add
        from apps.videos.models import Video
        existing_ids = set(videos.values_list("pk", flat=True))
        available = Video.objects.filter(user=self.request.user).exclude(pk__in=existing_ids)
        ctx["available_videos"] = available[:500]
        ctx["available_sites"] = sorted(set(
            available.exclude(site_name="").values_list("site_name", flat=True)
        ))

        # Build combined frame grid with filters
        filter_video = self.request.GET.get("video", "")
        filter_class = self.request.GET.get("class", "")
        ctx["filter_video"] = filter_video
        ctx["filter_class"] = filter_class

        anns_qs = project.annotations.select_related("video").order_by("video__title", "frame_number")
        if filter_video:
            try:
                anns_qs = anns_qs.filter(video_id=int(filter_video))
            except (ValueError, TypeError):
                pass

        total_boxes = 0
        class_counts = {}
        frame_cards = []

        for ann in anns_qs[:500]:
            boxes = ann.boxes or []
            box_classes = sorted(set(b.get("class", "unknown") for b in boxes)) if boxes else []

            # Class filter
            if filter_class and filter_class not in box_classes:
                continue

            total_boxes += len(boxes)
            for b in boxes:
                cls = b.get("class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1

            frame_cards.append({
                "video_pk": ann.video_id,
                "video_title": ann.video.title,
                "frame_number": ann.frame_number,
                "box_count": len(boxes),
                "classes": box_classes,
            })

        ctx["total_annotations"] = len(frame_cards)
        ctx["total_boxes"] = total_boxes
        ctx["class_counts"] = class_counts
        ctx["frame_cards"] = frame_cards

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

        # Build ordered frame list for prev/next navigation across all project frames
        all_frames = list(
            Annotation.objects.filter(project=project)
            .order_by("video__title", "frame_number")
            .values_list("video_id", "frame_number")
        )
        current_key = (video.pk if video else None, frame_number)
        ctx["total_project_frames"] = len(all_frames)
        ctx["current_frame_index"] = 0
        ctx["prev_frame_url"] = ""
        ctx["next_frame_url"] = ""

        if all_frames and current_key in all_frames:
            idx = all_frames.index(current_key)
            ctx["current_frame_index"] = idx + 1
            base_url = f"/annotations/{project.pk}/edit/"
            if idx > 0:
                pv, pf = all_frames[idx - 1]
                ctx["prev_frame_url"] = f"{base_url}?video={pv}&frame={pf}"
            if idx < len(all_frames) - 1:
                nv, nf = all_frames[idx + 1]
                ctx["next_frame_url"] = f"{base_url}?video={nv}&frame={nf}"

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

        # Auto-transfer from S3 if needed, then pre-annotate
        # The background thread handles both transfer and annotation
        thread = threading.Thread(
            target=self._run_pre_annotate,
            args=(project.pk, video.pk, blob_path),
            daemon=True,
        )
        thread.start()

        is_s3 = blob_path.startswith("s3://")
        if is_s3:
            messages.info(request, f"Transferring '{video.title}' from S3 then running AI detection. This takes 2-4 minutes. Refresh to see results.")
        else:
            messages.info(request, f"Pre-annotating '{video.title}' with AI. This takes 1-2 minutes. Refresh to see results.")
        return redirect("annotations:detail", pk=pk)

    @staticmethod
    def _run_pre_annotate(project_pk, video_pk, blob_path):
        import django
        django.setup()
        from django.db import connection

        try:
            from apps.videos.models import Video
            video = Video.objects.select_related("source").get(pk=video_pk)

            # Re-read blob_path from DB (might have been updated by analysis job)
            blob_path = video.azure_blob_path

            # Transfer from S3 if needed
            if blob_path.startswith("s3://"):
                logger.info("Pre-annotate: transferring video %s from S3", video_pk)
                try:
                    from apps.analysis.views import _transfer_s3_to_azure
                    blob_path = _transfer_s3_to_azure(video)
                    logger.info("Pre-annotate: transfer done -> %s", blob_path)
                except Exception as e:
                    logger.error("Pre-annotate: S3 transfer failed for %s: %s", video_pk, e, exc_info=True)
                    return

            connection.close()

            logger.info("Pre-annotate: calling Modal for video %s, path=%s", video_pk, blob_path)
            import modal
            fn = modal.Function.from_name("beemonitor-cloud", "pre_annotate_video")
            result = fn.remote(video_blob_path=blob_path, sample_interval=10, max_frames=300)

            logger.info("Pre-annotate: Modal returned %s frames, %s detections",
                        len(result.get("frames", [])) if result else 0,
                        result.get("total_detections", 0) if result else 0)

            if not result or not result.get("frames"):
                logger.info("Pre-annotate: no frames returned for video %s", video_pk)
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


class PreAnnotateAllView(LoginRequiredMixin, View):
    """Run AI pre-annotation on ALL videos in a project."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages
        import threading

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        videos = project.videos.all()

        if not videos.exists():
            messages.warning(request, "No videos in this project.")
            return redirect("annotations:detail", pk=pk)

        count = 0
        for video in videos:
            blob_path = video.azure_blob_path
            if not blob_path:
                continue
            thread = threading.Thread(
                target=PreAnnotateView._run_pre_annotate,
                args=(project.pk, video.pk, blob_path),
                daemon=True,
            )
            thread.start()
            count += 1

        messages.info(
            request,
            f"AI pre-annotation started for {count} video(s). This takes 1-4 minutes per video. Refresh to see results.",
        )
        return redirect("annotations:detail", pk=pk)


class FrameImageView(LoginRequiredMixin, View):
    """Return a JPG image of a specific video frame with optional bounding boxes drawn.

    NOTE: This downloads the entire video from Azure to extract a single frame,
    which is slow for large videos. Consider adding a frame cache (e.g. Redis or
    disk-based LRU cache keyed on video_pk + frame_number) to avoid repeated
    downloads of the same video.
    """

    def get(self, request, pk):
        video_id = request.GET.get("video")
        frame_number = int(request.GET.get("frame", 0))
        draw_boxes = request.GET.get("boxes", "true") == "true"
        project_pk = request.GET.get("project")

        # Get video and generate SAS URL or download frame
        from apps.videos.models import Video
        video = get_object_or_404(Video, pk=video_id, user=request.user)

        blob_path = video.azure_blob_path
        if blob_path.startswith("s3://") or not blob_path:
            # Return placeholder image
            return HttpResponse(status=404)

        try:
            import cv2
            import tempfile
            import numpy as np
            from django.conf import settings
            from azure.storage.blob import BlobServiceClient

            conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
            service = BlobServiceClient.from_connection_string(conn_str)

            # Download video to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            blob = service.get_blob_client("raw-videos", blob_path)
            with open(tmp.name, "wb") as fh:
                stream = blob.download_blob()
                stream.readinto(fh)

            cap = cv2.VideoCapture(tmp.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()

            import os
            os.unlink(tmp.name)

            if not ret:
                return HttpResponse(status=404)

            # Draw boxes if requested
            if draw_boxes and project_pk:
                try:
                    project = AnnotationProject.objects.get(pk=project_pk)
                    ann = Annotation.objects.get(project=project, video=video, frame_number=frame_number)

                    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
                    for box in ann.boxes:
                        x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                        cls_id = box.get("class_id", 0)
                        color = colors[cls_id % len(colors)]
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        label = box.get("class", "")
                        conf = box.get("confidence", "")
                        text = f"{label}" + (f" {conf:.0%}" if isinstance(conf, float) else "")
                        cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                except (Annotation.DoesNotExist, AnnotationProject.DoesNotExist):
                    pass

            # Encode as JPEG
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return HttpResponse(buf.tobytes(), content_type="image/jpeg")

        except Exception as e:
            logger.error("FrameImageView error: %s", e, exc_info=True)
            return HttpResponse(status=500)


class ReviewView(LoginRequiredMixin, TemplateView):
    """Visual annotation review page (Roboflow-style grid of annotated frames)."""
    template_name = "annotations/review.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        project = get_object_or_404(
            AnnotationProject, pk=self.kwargs["pk"], user=self.request.user
        )

        # Filters from query params
        video_filter = self.request.GET.get("video", "")
        class_filter = self.request.GET.get("cls", "")
        page_num = int(self.request.GET.get("page", 1))

        annotations_qs = project.annotations.select_related("video").order_by(
            "video__title", "frame_number"
        )

        if video_filter:
            annotations_qs = annotations_qs.filter(video__pk=video_filter)

        # Materialise and apply class filter (boxes is JSON, so filter in Python)
        all_anns = list(annotations_qs[:2000])  # Cap for safety

        if class_filter:
            filtered = []
            for ann in all_anns:
                classes_in_ann = {b.get("class", "") for b in (ann.boxes or [])}
                if class_filter in classes_in_ann:
                    filtered.append(ann)
            all_anns = filtered

        # Build annotation card data
        ann_data = []
        for ann in all_anns:
            boxes = ann.boxes or []
            class_names = sorted(set(b.get("class", "unknown") for b in boxes)) if boxes else []
            ann_data.append({
                "video": ann.video,
                "video_pk": ann.video.pk,
                "frame_number": ann.frame_number,
                "box_count": len(boxes),
                "class_names": class_names,
            })

        # Pagination (50 per page)
        per_page = 50
        total = len(ann_data)
        total_pages = max(1, (total + per_page - 1) // per_page)
        page_num = max(1, min(page_num, total_pages))
        start = (page_num - 1) * per_page
        end = start + per_page
        page_anns = ann_data[start:end]

        ctx["project"] = project
        ctx["annotations"] = page_anns
        ctx["total_count"] = total
        ctx["page"] = page_num
        ctx["total_pages"] = total_pages
        ctx["has_prev"] = page_num > 1
        ctx["has_next"] = page_num < total_pages
        ctx["prev_page"] = page_num - 1
        ctx["next_page"] = page_num + 1

        # Filter options
        ctx["videos"] = project.videos.all().order_by("title")
        ctx["classes"] = project.classes
        ctx["current_video"] = video_filter
        ctx["current_class"] = class_filter

        return ctx


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
