import io
import json
import zipfile
from concurrent.futures import ThreadPoolExecutor

from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from django.urls import reverse_lazy
from django.views import View
from django.views.generic import CreateView, DetailView, ListView, TemplateView, UpdateView

import logging

from .forms import ProjectCreateForm
from .models import Annotation, AnnotationProject

logger = logging.getLogger(__name__)


def _preannotate_opts(request):
    """(sample_interval, max_frames, confidence) from POST, clamped, with
    settings-based defaults. Lets power users tune sampling per request."""
    from django.conf import settings

    def _clamp(name, default, lo, hi, cast=int):
        try:
            v = cast(request.POST.get(name) or default)
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    return (
        _clamp("sample_interval", settings.PREANNOTATE_SAMPLE_INTERVAL, 1, 120),
        _clamp("max_frames", settings.PREANNOTATE_MAX_FRAMES, 10, 2000),
        _clamp("confidence", settings.PREANNOTATE_CONFIDENCE, 0.01, 0.9, cast=float),
    )


def _labeler_opts(request):
    """(labeler, custom_model_key) from the Labeler select. "custom:<pk>"
    picks one of the user's fine-tuned models — it runs on the YOLO endpoint
    with custom_bee_model_path; unknown/foreign pks fall back to the default."""
    raw = request.POST.get("labeler") or "yolo"
    if raw == "sam3":
        return "sam3", ""
    if raw.startswith("custom:"):
        from apps.training.models import CustomModel
        try:
            cm = CustomModel.objects.get(
                pk=int(raw[7:]), user=request.user, is_active=True,
                status=CustomModel.Status.READY,
            )
            if cm.storage_key:
                return "yolo", cm.storage_key
        except (CustomModel.DoesNotExist, TypeError, ValueError):
            pass
    return "yolo", ""


def _sam3_opts(request):
    """(nms_iou, max_detections) from POST, clamped. SAM 3-specific quality knobs:
    nms_iou dedupes overlapping boxes across the per-class prompt passes (0 = off);
    max_detections caps boxes per class per frame."""
    def _num(name, default, lo, hi, cast):
        try:
            v = cast(request.POST.get(name) or default)
        except (TypeError, ValueError):
            v = default
        return max(lo, min(hi, v))

    return (
        _num("nms_iou", 0.5, 0.0, 1.0, float),
        _num("max_detections", 100, 1, 500, int),
    )


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


class ProjectUpdateView(LoginRequiredMixin, UpdateView):
    """Edit a project's name, description, and classes. Reuses the create form."""
    model = AnnotationProject
    form_class = ProjectCreateForm
    template_name = "annotations/settings.html"

    def get_queryset(self):
        return AnnotationProject.objects.filter(user=self.request.user)

    def get_initial(self):
        initial = super().get_initial()
        initial["classes_text"] = ", ".join(self.object.classes or [])
        return initial

    def get_success_url(self):
        return reverse_lazy("annotations:detail", kwargs={"pk": self.object.pk})


class ProjectDeleteView(LoginRequiredMixin, View):
    """Delete a project (cascades its annotations + pre-annotation tasks)."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        name = project.name
        project.delete()
        messages.info(request, f"Deleted project '{name}' and its annotations.")
        return redirect("annotations:list")


class ProjectDetailView(LoginRequiredMixin, DetailView):
    model = AnnotationProject
    template_name = "annotations/detail.html"
    context_object_name = "project"

    def get_queryset(self):
        return AnnotationProject.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        project = self.object

        # Advance any in-flight pre-annotation tasks on page load (the
        # background reconciler also does this, so it works with no open tab).
        poll_preannotation_tasks(self.request.user)
        from .models import PreAnnotationTask
        active = list(
            PreAnnotationTask.objects.filter(
                project=project,
                status__in=[PreAnnotationTask.Status.QUEUED, PreAnnotationTask.Status.PROCESSING],
            ).select_related("video")
        )
        recent_failed = list(
            PreAnnotationTask.objects.filter(
                project=project, status=PreAnnotationTask.Status.FAILED,
            ).select_related("video")[:5]
        )
        ctx["preannot_active"] = active
        ctx["preannot_failed"] = recent_failed

        videos = project.videos.all()
        ctx["project_videos"] = videos
        ctx["video_count"] = videos.count()

        # Video data for "no annotations" fallback links
        video_data = []
        for video in videos:
            ann_count = Annotation.objects.filter(project=project, video=video).count()
            video_data.append({"video": video, "annotation_count": ann_count})
        ctx["video_data"] = video_data

        # Available videos to add — the SAME comprehensive filter as the Processing
        # page (device · confirmation · site · year · month · day · hour · search),
        # over all accessible videos (own + shared), so every hotel/location/time is
        # reachable — not just the first page.
        from apps.analysis.views import _sanitize_site, _unsanitize_site
        from apps.devices.models import Device
        from apps.videos.models import Video
        existing_ids = set(videos.values_list("pk", flat=True))
        all_user = Video.accessible(self.request.user)
        available = all_user.exclude(pk__in=existing_ids)

        af = {k: self.request.GET.get("av_" + k, "").strip()
              for k in ("q", "device", "site", "year", "month", "day", "hour", "confirmed")}
        if af["q"]:
            available = available.filter(title__icontains=af["q"])
        if af["device"]:
            available = available.filter(device_id=af["device"])
        if af["site"]:
            available = available.filter(site_name=_unsanitize_site(af["site"]))
        for field in ("year", "month", "day", "hour"):
            if af[field]:
                try:
                    available = available.filter(**{field: int(af[field])})
                except (ValueError, TypeError):
                    pass
        if af["confirmed"] == "yes":
            available = available.filter(metadata__bee_confirmed=True)
        elif af["confirmed"] == "no":
            available = available.filter(metadata__bee_confirmed=False)
        available = available.select_related("device").order_by("-recorded_at", "-id")

        ctx["available_videos"] = available[:500]
        ctx["available_count"] = available.count()
        ctx["available_filter"] = af
        ctx["available_filter_on"] = any(af.values())
        ctx["available_devices"] = Device.accessible(self.request.user).order_by("name")
        # Options from ALL accessible videos so nothing is hidden.
        ctx["available_opts"] = {
            "sites": sorted({_sanitize_site(s) for s in
                             all_user.exclude(site_name="").values_list("site_name", flat=True)}),
            "years": sorted(set(all_user.exclude(year=None).values_list("year", flat=True))),
            "months": sorted(set(all_user.exclude(month=None).values_list("month", flat=True))),
            "days": sorted(set(all_user.exclude(day=None).values_list("day", flat=True))),
            "hours": sorted(set(all_user.exclude(hour=None).values_list("hour", flat=True))),
        }

        # Build combined frame grid with filters
        filter_video = self.request.GET.get("video", "")
        filter_class = self.request.GET.get("class", "")
        filter_review = self.request.GET.get("review", "")  # ""|reviewed|unreviewed|human|llm
        ctx["filter_video"] = filter_video
        ctx["filter_class"] = filter_class
        ctx["filter_review"] = filter_review

        anns_qs = project.annotations.select_related("video").order_by("video__title", "frame_number")
        if filter_video:
            try:
                anns_qs = anns_qs.filter(video_id=int(filter_video))
            except (ValueError, TypeError):
                pass
        if filter_review == "reviewed":
            anns_qs = anns_qs.filter(reviewed=True)
        elif filter_review == "unreviewed":
            anns_qs = anns_qs.filter(reviewed=False)
        elif filter_review in ("human", "llm"):
            anns_qs = anns_qs.filter(review_source=filter_review)

        # Review progress across the whole project (not just the filtered page).
        proj_anns = project.annotations
        ctx["reviewed_count"] = proj_anns.filter(reviewed=True).count()
        ctx["reviewed_human"] = proj_anns.filter(review_source="human").count()
        ctx["reviewed_llm"] = proj_anns.filter(review_source="llm").count()

        # Stats run over ALL matching annotations; the thumbnail grid is
        # paginated (presigning thousands of URLs per page-load is too slow).
        PAGE_SIZE = 500
        try:
            page = max(1, int(self.request.GET.get("page", 1)))
        except (TypeError, ValueError):
            page = 1
        win_start = (page - 1) * PAGE_SIZE
        win_end = win_start + PAGE_SIZE

        total_boxes = 0
        total_matching = 0  # index into the (class-)filtered set
        class_counts = {}
        frame_cards = []

        for ann in anns_qs:
            boxes = ann.boxes or []
            box_classes = sorted(set(b.get("class", "unknown") for b in boxes)) if boxes else []

            # Class filter
            if filter_class and filter_class not in box_classes:
                continue

            for b in boxes:
                cls = b.get("class", "unknown")
                class_counts[cls] = class_counts.get(cls, 0) + 1
            total_boxes += len(boxes)

            # Only build cards for the current page window.
            if win_start <= total_matching < win_end:
                frame_cards.append({
                    "video_pk": ann.video_id,
                    "video_title": ann.video.title,
                    "frame_number": ann.frame_number,
                    "box_count": len(boxes),
                    "classes": box_classes,
                    "frame_image_path": ann.frame_image_path or "",
                    "reviewed": ann.reviewed,
                    "review_source": ann.review_source,
                })
            total_matching += 1

        # Presigned URLs for frame thumbnails
        try:
            from config.storage import get_s3_client
            s3 = get_s3_client()
            for card in frame_cards:
                if card["frame_image_path"]:
                    card["thumbnail_url"] = s3.generate_presigned_url(
                        "processed", card["frame_image_path"], expiry_hours=2,
                    )
        except Exception as e:
            logger.warning("Failed to presign thumbnails: %s", e)

        ctx["total_annotations"] = total_matching
        ctx["total_boxes"] = total_boxes
        ctx["class_counts"] = class_counts
        ctx["frame_cards"] = frame_cards
        # Pagination for the grid.
        import math
        total_pages = max(1, math.ceil(total_matching / PAGE_SIZE)) if total_matching else 1
        page = min(page, total_pages)
        ctx["page"] = page
        ctx["total_pages"] = total_pages
        ctx["page_size"] = PAGE_SIZE
        ctx["page_start"] = win_start + 1 if frame_cards else 0
        ctx["page_end"] = win_start + len(frame_cards)
        ctx["has_prev_page"] = page > 1
        ctx["has_next_page"] = page < total_pages
        # Query string carrying the current filters (minus page) for page links.
        from urllib.parse import urlencode
        _pg = {k: v for k, v in (("video", filter_video), ("class", filter_class),
                                 ("review", filter_review)) if v}
        ctx["page_qs_prefix"] = ("?" + urlencode(_pg) + "&") if _pg else "?"

        # Defaults for the AI pre-annotate sampling controls.
        from django.conf import settings
        ctx["preannotate_defaults"] = {
            "sample_interval": settings.PREANNOTATE_SAMPLE_INTERVAL,
            "max_frames": settings.PREANNOTATE_MAX_FRAMES,
            "confidence": settings.PREANNOTATE_CONFIDENCE,
        }
        # Fine-tuned models selectable as the pre-annotation labeler.
        from apps.training.models import CustomModel
        ctx["custom_bee_models"] = CustomModel.objects.filter(
            user=self.request.user, is_active=True, status=CustomModel.Status.READY,
        ).exclude(storage_key="")

        return ctx


class RemoveVideoView(LoginRequiredMixin, View):
    """Remove one video (and its annotations) from an annotation project."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        video_id = request.POST.get("video_id")
        if video_id:
            project.videos.remove(video_id)
            Annotation.objects.filter(project=project, video_id=video_id).delete()
            messages.info(request, "Video removed from the project.")
        return redirect("annotations:detail", pk=pk)


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

        # Auto-ingest external-S3 video into our raw-videos bucket if needed
        ctx["transferring"] = False
        if video and video.storage_key.startswith("s3://"):
            try:
                import threading
                from apps.analysis.views import _ingest_external_s3_to_storage

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

        # Presigned URL for video playback
        if video and video.storage_key and not video.storage_key.startswith("s3://"):
            try:
                from config.storage import get_s3_client
                ctx["video_url"] = get_s3_client().generate_presigned_url(
                    "raw-videos", video.storage_key,
                )
            except Exception:
                pass

        return ctx

    @staticmethod
    def _do_transfer(video_pk):
        """Background thread: ingest external-S3 video into our raw-videos bucket."""
        import django
        django.setup()
        from django.db import connection
        try:
            from apps.videos.models import Video
            from apps.analysis.views import _ingest_external_s3_to_storage
            video = Video.objects.select_related("source").get(pk=video_pk)
            _ingest_external_s3_to_storage(video)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Background transfer failed for %s: %s", video_pk, e)
        finally:
            connection.close()


class TransferVideoView(LoginRequiredMixin, View):
    """Ingest an external-S3 video into our raw-videos bucket so frames are available."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        video_id = request.POST.get("video_id")
        frame = request.POST.get("frame", 0)

        from apps.videos.models import Video
        video = get_object_or_404(Video, pk=video_id, user=request.user)

        if not video.storage_key.startswith("s3://"):
            messages.info(request, "Video is already in storage.")
            return redirect(f"/annotations/{pk}/edit/?video={video_id}&frame={frame}")

        try:
            from apps.analysis.views import _ingest_external_s3_to_storage
            _ingest_external_s3_to_storage(video)
            messages.success(request, "Video transferred. Frames now available.")
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

        # A human saving in the editor = a human review.
        from django.utils import timezone
        annotation, created = Annotation.objects.update_or_create(
            project=project,
            video=video,
            frame_number=frame_number,
            defaults={
                "boxes": boxes,
                "reviewed": True,
                "review_source": Annotation.ReviewSource.HUMAN,
                "reviewed_at": timezone.now(),
            },
        )

        return JsonResponse({
            "success": True,
            "created": created,
            "annotation_id": annotation.pk,
            "reviewed": True,
        })


# Bounded pool for the (short) pre-annotation SPAWN — transfer video if needed,
# then invoke_endpoint_async. Finalization is NOT here; the reconciler does it.
_PREANNOT_POOL = ThreadPoolExecutor(max_workers=3, thread_name_prefix="preannot-spawn")


def spawn_preannotation_async(task_pk: int) -> None:
    _PREANNOT_POOL.submit(_spawn_preannotation, task_pk)


def _spawn_preannotation(task_pk: int) -> None:
    """Fire the SageMaker async invocation for one PreAnnotationTask and store
    its output/failure S3 URIs. Durable: after this returns the reconciler can
    finalize the task even across a deploy. Runs in a bounded thread."""
    import django
    django.setup()
    from urllib.parse import urlparse
    import boto3
    from botocore.config import Config
    from django.conf import settings
    from django.db import connection
    from django.utils import timezone

    from .models import PreAnnotationTask

    try:
        task = PreAnnotationTask.objects.select_related("project", "video", "video__source").get(pk=task_pk)
    except PreAnnotationTask.DoesNotExist:
        connection.close()
        return

    try:
        video = task.video
        blob_path = video.storage_key
        if blob_path.startswith("s3://"):
            from apps.analysis.views import _ingest_external_s3_to_storage
            blob_path = _ingest_external_s3_to_storage(video)

        labeler = task.labeler
        if labeler == "sam3":
            endpoint = settings.SAGEMAKER_SAM3_ENDPOINT_NAME
        else:
            endpoint = settings.SAGEMAKER_ENDPOINT_NAME
        if not endpoint:
            raise RuntimeError(f"{'SAM3' if labeler=='sam3' else 'YOLO'} endpoint not configured")

        in_bucket = settings.SAGEMAKER_INPUT_BUCKET
        region = settings.AWS_REGION
        p = task.params or {}
        payload = {
            "task": "pre_annotate",
            "job_id": f"preannot-{task_pk}",
            "user_id": str(task.user_id),
            "video_blob_path": blob_path,
            "classes": task.project.classes or ["bee", "wasp", "nest"],
            "sample_interval": p.get("sample_interval", 10),
            "max_frames": p.get("max_frames", 300),
            "confidence_threshold": p.get("confidence", 0.15),
            "selection": p.get("selection", "uniform"),
            "nms_iou": p.get("nms_iou", 0.5),
            "max_detections": p.get("max_detections", 100),
        }
        if task.custom_model_key:
            payload["custom_bee_model_path"] = task.custom_model_key

        cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2})
        s3 = boto3.client("s3", region_name=region, config=cfg)
        smrt = boto3.client("sagemaker-runtime", region_name=region, config=cfg)
        key = f"preannotate/{task.project_id}/{task_pk}-{labeler}.json"
        s3.put_object(Bucket=in_bucket, Key=key,
                      Body=json.dumps(payload).encode("utf-8"),
                      ContentType="application/json")
        resp = smrt.invoke_endpoint_async(
            EndpointName=endpoint,
            InputLocation=f"s3://{in_bucket}/{key}",
            ContentType="application/json",
            InferenceId=f"preannot-{task_pk}",
        )
        out = urlparse(resp["OutputLocation"])
        fail_loc = resp.get("FailureLocation", "") or ""
        if not fail_loc:
            fail_loc = resp["OutputLocation"].replace(".out", ".failure")
        PreAnnotationTask.objects.filter(pk=task_pk).update(
            status=PreAnnotationTask.Status.PROCESSING,
            output_uri=resp["OutputLocation"], failure_uri=fail_loc,
            started_at=timezone.now(),
        )
        logger.info("pre-annotate task %s invoked -> %s", task_pk, resp["OutputLocation"])
    except Exception as e:
        logger.exception("pre-annotate spawn failed for task %s", task_pk)
        PreAnnotationTask.objects.filter(pk=task_pk).update(
            status=PreAnnotationTask.Status.FAILED,
            error_message=str(e)[:1000], completed_at=timezone.now(),
        )
    finally:
        connection.close()


# Past this the async request has expired (SM async: 1h processing + 6h TTL).
_PREANNOT_TIMEOUT_HOURS = 8


def finalize_preannotation_task(task, s3) -> bool:
    """Check one PROCESSING task's output/failure S3 URI. On the result, write
    Annotation rows (protecting human-reviewed frames) and complete; on failure
    or timeout, mark failed. Returns True if the task reached a terminal state.
    Idempotent; safe to call repeatedly from the reconciler."""
    import json as _json
    from datetime import timedelta
    from urllib.parse import urlparse
    from django.utils import timezone
    from botocore.exceptions import ClientError

    from .models import Annotation, PreAnnotationTask

    if not task.output_uri:
        return False
    out = urlparse(task.output_uri)
    try:
        body = s3.get_object(Bucket=out.netloc, Key=out.path.lstrip("/"))["Body"].read()
        result = _json.loads(body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code", "") not in ("NoSuchKey", "404", "NotFound"):
            logger.warning("pre-annotate poll error task %s: %s", task.pk, e)
            return False
        # No output yet — check for a platform failure, else timeout.
        if task.failure_uri:
            f = urlparse(task.failure_uri)
            try:
                fb = s3.get_object(Bucket=f.netloc, Key=f.path.lstrip("/"))["Body"].read()
                PreAnnotationTask.objects.filter(pk=task.pk).update(
                    status=PreAnnotationTask.Status.FAILED,
                    error_message=f"endpoint failure: {fb.decode('utf-8', 'replace')[:500]}",
                    completed_at=timezone.now())
                return True
            except ClientError:
                pass
        age = timezone.now() - (task.started_at or task.created_at)
        if age > timedelta(hours=_PREANNOT_TIMEOUT_HOURS):
            PreAnnotationTask.objects.filter(pk=task.pk).update(
                status=PreAnnotationTask.Status.FAILED,
                error_message="Timed out — no result. Re-run pre-annotation.",
                completed_at=timezone.now())
            return True
        return False

    frames = result.get("frames") or []
    width = result.get("video_width", 1280)
    height = result.get("video_height", 720)
    human_frames = set(
        Annotation.objects.filter(
            project_id=task.project_id, video_id=task.video_id,
            review_source=Annotation.ReviewSource.HUMAN,
        ).values_list("frame_number", flat=True)
    )
    created = 0
    for fd in frames:
        if fd["frame_number"] in human_frames:
            continue
        Annotation.objects.update_or_create(
            project_id=task.project_id, video_id=task.video_id,
            frame_number=fd["frame_number"],
            defaults={
                "boxes": fd["boxes"],
                "image_width": width, "image_height": height,
                "frame_image_path": fd.get("frame_image_path", ""),
                "reviewed": False,
                "review_source": Annotation.ReviewSource.NONE,
                "reviewed_at": None,
            },
        )
        created += 1

    PreAnnotationTask.objects.filter(pk=task.pk).update(
        status=PreAnnotationTask.Status.COMPLETED,
        frames_written=created, completed_at=timezone.now())

    exec_secs = result.get("execution_seconds", 0) or 0
    try:
        from apps.accounts.models import UserProfile
        profile, _ = UserProfile.objects.get_or_create(user_id=task.user_id)
        profile.charge(int(exec_secs) if exec_secs else 30, gpu_seconds=exec_secs or 30)
    except Exception as e:
        logger.error("pre-annotate credit charge failed task %s: %s", task.pk, e)
    logger.info("pre-annotate task %s complete: %d frames written", task.pk, created)
    return True


def poll_preannotation_tasks(user=None) -> int:
    """Finalize in-flight pre-annotation tasks. Reusable by the reconciler and a
    page poll. ``user=None`` polls all users. Idempotent; never raises."""
    import boto3
    from botocore.config import Config
    from django.conf import settings

    from .models import PreAnnotationTask
    try:
        qs = PreAnnotationTask.objects.filter(status=PreAnnotationTask.Status.PROCESSING)
        if user is not None:
            qs = qs.filter(user=user)
        tasks = list(qs.select_related("project", "video")[:100])
        if not tasks:
            return 0
        s3 = boto3.client("s3", region_name=getattr(settings, "AWS_REGION", "us-east-1"),
                          config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}))
        return sum(1 for t in tasks if finalize_preannotation_task(t, s3))
    except Exception:
        logger.exception("poll_preannotation_tasks failed")
        return 0


class PreAnnotateView(LoginRequiredMixin, View):
    """Create a durable pre-annotation task for one video and fire it."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        from apps.videos.models import Video
        video = get_object_or_404(Video, pk=request.POST.get("video_id"), user=request.user)

        task = _create_preannotation_task(request, project, video)
        spawn_preannotation_async(task.pk)
        engine = "SAM 3" if task.labeler == "sam3" else "AI"
        messages.info(request, f"Pre-annotating '{video.title}' with {engine}. "
                               "This runs in the background — refresh to see progress.")
        return redirect("annotations:detail", pk=pk)


def _create_preannotation_task(request, project, video):
    """Build a PreAnnotationTask row from the request's pre-annotate options."""
    from .models import PreAnnotationTask

    sample_interval, max_frames, confidence = _preannotate_opts(request)
    labeler, custom_model_key = _labeler_opts(request)
    selection = "diverse" if request.POST.get("selection", "diverse") == "diverse" else "uniform"
    if labeler != "sam3":
        selection = "uniform"
    nms_iou, max_detections = _sam3_opts(request)
    return PreAnnotationTask.objects.create(
        user=request.user, project=project, video=video,
        labeler=labeler, custom_model_key=custom_model_key,
        params={
            "sample_interval": sample_interval, "max_frames": max_frames,
            "confidence": confidence, "selection": selection,
            "nms_iou": nms_iou, "max_detections": max_detections,
        },
    )


class PreAnnotateAllView(LoginRequiredMixin, View):
    """Run AI pre-annotation on all videos in a project, or on the subset
    checked in the video list (video_ids)."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        videos = project.videos.all()
        video_ids = request.POST.getlist("video_ids")
        if video_ids:
            videos = videos.filter(pk__in=video_ids)

        if not videos.exists():
            messages.warning(request, "No videos in this project.")
            return redirect("annotations:detail", pk=pk)

        # One durable task per video; the bounded spawn pool + reconciler handle
        # them (no more one-unbounded-thread-per-video).
        count = 0
        labeler = "yolo"
        for video in videos:
            if not video.storage_key:
                continue
            task = _create_preannotation_task(request, project, video)
            labeler = task.labeler
            spawn_preannotation_async(task.pk)
            count += 1

        engine = "SAM 3" if labeler == "sam3" else "AI"
        messages.info(
            request,
            f"{engine} pre-annotation started for {count} video(s). Runs in the "
            "background — refresh to see progress.",
        )
        return redirect("annotations:detail", pk=pk)


class FrameImageView(LoginRequiredMixin, View):
    """Return a JPG image of a specific video frame.

    First tries to serve a pre-saved frame from S3 (uploaded during
    pre-annotation). Falls back to extracting from the video if needed.
    """

    def get(self, request, pk):
        video_id = request.GET.get("video")
        frame_number = int(request.GET.get("frame", 0))
        draw_boxes = request.GET.get("boxes", "false") == "true"

        from apps.videos.models import Video
        video = get_object_or_404(Video, pk=video_id, user=request.user)

        ann = None
        try:
            project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
            ann = Annotation.objects.get(project=project, video=video, frame_number=frame_number)
        except Annotation.DoesNotExist:
            ann = None

        # Prefer the pre-saved JPEG, but FALL BACK to extracting from the video
        # when that object is missing/broken — otherwise a stale frame_image_path
        # renders permanently blank even though the video is available.
        if ann and ann.frame_image_path:
            resp = self._serve_from_storage(ann.frame_image_path, ann if draw_boxes else None)
            if resp is not None:
                return resp
            logger.warning("FrameImageView: saved frame %s missing — extracting from video",
                           ann.frame_image_path)

        return self._extract_from_video(video, frame_number)

    def _serve_from_storage(self, blob_path, ann_for_boxes=None):
        """Serve a pre-saved JPEG frame from the processed S3 bucket.
        Returns None (not a 404) on failure so the caller can fall back."""
        try:
            import io
            from config.storage import get_s3_client

            buf = io.BytesIO()
            get_s3_client().download_to_stream("processed", blob_path, buf)
            data = buf.getvalue()
            if not data:
                return None

            if ann_for_boxes and ann_for_boxes.boxes:
                import cv2
                import numpy as np
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 0, 255)]
                for box in ann_for_boxes.boxes:
                    x, y, w, h = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
                    color = colors[box.get("class_id", 0) % len(colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    label = box.get("class", "")
                    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                data = encoded.tobytes()

            response = HttpResponse(data, content_type="image/jpeg")
            response["Cache-Control"] = "public, max-age=3600"
            return response
        except Exception as e:
            logger.warning("FrameImageView S3 miss for %s: %s", blob_path, e)
            return None  # let the caller fall back to video extraction

    def _extract_from_video(self, video, frame_number):
        """Fallback: download video and extract frame (slow)."""
        blob_path = video.storage_key
        if not blob_path or blob_path.startswith("s3://"):
            return HttpResponse(status=404)

        try:
            import cv2
            import tempfile
            import os
            from config.storage import get_s3_client

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            get_s3_client().download_file("raw-videos", blob_path, tmp.name)

            cap = cv2.VideoCapture(tmp.name)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            os.unlink(tmp.name)

            if not ret:
                return HttpResponse(status=404)

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return HttpResponse(buf.tobytes(), content_type="image/jpeg")
        except Exception as e:
            logger.error("FrameImageView extract error: %s", e, exc_info=True)
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

            from config.storage import get_s3_client
            s3 = get_s3_client()
            video_cache = {}  # video_pk -> (cv2.VideoCapture, tmp_path)

            # Deterministic split matching the training container's (sorted tail
            # → val). val_percent from ?val_percent= (default 20, = the training
            # job default) so "what you export" == "what training splits".
            try:
                val_percent = int(request.GET.get("val_percent", 20))
            except (TypeError, ValueError):
                val_percent = 20
            val_percent = max(0, min(90, val_percent))
            ann_list = list(annotations)
            split_idx = len(ann_list) - max(1, len(ann_list) * val_percent // 100) if ann_list else 0

            for i, ann in enumerate(ann_list):
                split = "val" if i >= split_idx else "train"
                base_name = f"{ann.video.title}_f{ann.frame_number:06d}"
                zf.writestr(f"labels/{split}/{base_name}.txt", ann.to_yolo_format())

                # Prefer the pre-saved frame JPEG (processed bucket) — the
                # tracker/pre-annotator already extracted it. Re-extract from
                # video only as a fallback.
                img_bytes = None
                if ann.frame_image_path:
                    try:
                        import io
                        b = io.BytesIO()
                        s3.download_to_stream("processed", ann.frame_image_path, b)
                        img_bytes = b.getvalue()
                    except Exception as e:
                        logger.warning("export: saved frame %s missing (%s) — re-extracting",
                                       ann.frame_image_path, e)
                if img_bytes is None:
                    try:
                        if ann.video.pk not in video_cache:
                            blob_path = ann.video.storage_key
                            if blob_path and not blob_path.startswith("s3://"):
                                import tempfile, cv2
                                tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                                tmp.close()
                                s3.download_file("raw-videos", blob_path, tmp.name)
                                video_cache[ann.video.pk] = (cv2.VideoCapture(tmp.name), tmp.name)
                        if ann.video.pk in video_cache:
                            import cv2
                            cap, _ = video_cache[ann.video.pk]
                            cap.set(cv2.CAP_PROP_POS_FRAMES, ann.frame_number)
                            ret, frame = cap.read()
                            if ret:
                                _, img_buf = cv2.imencode(".jpg", frame)
                                img_bytes = img_buf.tobytes()
                    except Exception as e:
                        logger.error("Failed to extract frame %d from video %s: %s",
                                     ann.frame_number, ann.video.pk, e)
                if img_bytes is not None:
                    zf.writestr(f"images/{split}/{base_name}.jpg", img_bytes)

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


# --- LLM vision review (memory/26) ------------------------------------------

class LLMReviewView(LoginRequiredMixin, View):
    """Per-frame AI review: filter the current boxes with the accurate model, return
    the kept set for the editor to show (the human saves = human-reviewed)."""

    def post(self, request, pk):
        from django.conf import settings
        from . import llm_review

        if not llm_review.available():
            return JsonResponse({"error": "LLM review isn't configured on this server "
                                          "(no ANTHROPIC_API_KEY)."}, status=400)
        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        try:
            data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)
        video_id = data.get("video_id")
        frame_number = data.get("frame_number")
        boxes = data.get("boxes", [])

        ann = Annotation.objects.filter(
            project=project, video_id=video_id, frame_number=frame_number).first()
        if not ann or not ann.frame_image_path:
            return JsonResponse({"error": "No saved frame image to review — Save first, "
                                          "then AI Review."}, status=400)
        kept, notes = llm_review.review_boxes(ann, boxes, model=settings.ASSISTANT_MODEL)
        return JsonResponse({
            "success": True,
            "boxes": kept,
            "removed": len(boxes) - len(kept),
            "notes": notes,
        })


class LLMReviewAllView(LoginRequiredMixin, View):
    """Batch AI review: filter every UNREVIEWED frame in the project with the cheap
    model (capped), persisting each as llm-reviewed. Never touches human-reviewed."""

    CAP = 200

    def post(self, request, pk):
        import threading

        from django.conf import settings
        from django.shortcuts import redirect
        from django.contrib import messages

        from . import llm_review

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        if not llm_review.available():
            messages.warning(request, "LLM review isn't configured (no ANTHROPIC_API_KEY).")
            return redirect("annotations:detail", pk=pk)

        ann_ids = list(project.annotations.filter(reviewed=False)
                       .values_list("pk", flat=True)[:self.CAP])
        if not ann_ids:
            messages.info(request, "No unreviewed frames to review.")
            return redirect("annotations:detail", pk=pk)

        model = settings.ASSISTANT_FAST_MODEL

        def _run():
            from django.db import connection
            for aid in ann_ids:
                llm_review.review_annotation(aid, model=model)
            connection.close()

        threading.Thread(target=_run, daemon=True).start()
        messages.info(request, f"LLM is reviewing {len(ann_ids)} unreviewed frame(s) with "
                               f"{model}. Refresh to see the blue ✓ badges.")
        return redirect("annotations:detail", pk=pk)
