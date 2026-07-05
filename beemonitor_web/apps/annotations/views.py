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
                "frame_image_path": ann.frame_image_path or "",
                "reviewed": ann.reviewed,
                "review_source": ann.review_source,
            })

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

        ctx["total_annotations"] = len(frame_cards)
        ctx["total_boxes"] = total_boxes
        ctx["class_counts"] = class_counts
        ctx["frame_cards"] = frame_cards

        # Defaults for the AI pre-annotate sampling controls.
        from django.conf import settings
        ctx["preannotate_defaults"] = {
            "sample_interval": settings.PREANNOTATE_SAMPLE_INTERVAL,
            "max_frames": settings.PREANNOTATE_MAX_FRAMES,
            "confidence": settings.PREANNOTATE_CONFIDENCE,
        }

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

        blob_path = video.storage_key
        sample_interval, max_frames, confidence = _preannotate_opts(request)
        # labeler=sam3 routes to the SAM 3 endpoint (domain-robust, open-vocabulary) —
        # the fix for new-domain footage the current YOLO misses. Default: yolo.
        labeler = "sam3" if request.POST.get("labeler") == "sam3" else "yolo"
        # DINOv3 diverse selection (SAM 3 only). Default diverse for SAM 3 (its point),
        # uniform every-Nth for YOLO.
        selection = "diverse" if request.POST.get("selection", "diverse") == "diverse" else "uniform"
        if labeler != "sam3":
            selection = "uniform"
        nms_iou, max_detections = _sam3_opts(request)

        # Auto-transfer from S3 if needed, then pre-annotate
        # The background thread handles both transfer and annotation
        thread = threading.Thread(
            target=self._run_pre_annotate,
            args=(project.pk, video.pk, blob_path, sample_interval, max_frames, confidence,
                  labeler, selection, nms_iou, max_detections),
            daemon=True,
        )
        thread.start()

        engine = "SAM 3" if labeler == "sam3" else "AI"
        is_s3 = blob_path.startswith("s3://")
        if is_s3:
            messages.info(request, f"Transferring '{video.title}' from S3 then running {engine} detection. This takes 2-4 minutes. Refresh to see results.")
        else:
            messages.info(request, f"Pre-annotating '{video.title}' with {engine}. This takes 1-2 minutes. Refresh to see results.")
        return redirect("annotations:detail", pk=pk)

    @staticmethod
    def _run_pre_annotate(project_pk, video_pk, blob_path,
                          sample_interval=10, max_frames=300, confidence=0.15,
                          labeler="yolo", selection="uniform",
                          nms_iou=0.5, max_detections=100):
        """Invoke the SageMaker async endpoint (task=pre_annotate), poll the
        result from S3, then create Annotation rows. Runs in a daemon thread."""
        import json
        import time
        from urllib.parse import urlparse

        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
        from django.conf import settings
        from django.db import connection

        try:
            from apps.annotations.models import Annotation, AnnotationProject
            from apps.videos.models import Video

            video = Video.objects.select_related("source").get(pk=video_pk)
            blob_path = video.storage_key
            if blob_path.startswith("s3://"):
                logger.info("pre-annotate: transferring video %s from S3", video_pk)
                from apps.analysis.views import _ingest_external_s3_to_storage
                blob_path = _ingest_external_s3_to_storage(video)

            project = AnnotationProject.objects.get(pk=project_pk)
            classes = project.classes or ["bee", "wasp", "nest"]
            user_id = project.user_id

            if labeler == "sam3":
                endpoint = settings.SAGEMAKER_SAM3_ENDPOINT_NAME
                if not endpoint:
                    logger.error("pre-annotate: SAGEMAKER_SAM3_ENDPOINT_NAME unset — SAM 3 endpoint not deployed?")
                    return
            else:
                endpoint = settings.SAGEMAKER_ENDPOINT_NAME
                if not endpoint:
                    logger.error("pre-annotate: SAGEMAKER_ENDPOINT_NAME unset — endpoint not deployed?")
                    return
            in_bucket = settings.SAGEMAKER_INPUT_BUCKET
            region = settings.AWS_REGION
            connection.close()  # don't hold a DB conn across the long poll

            cfg = Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2})
            s3 = boto3.client("s3", region_name=region, config=cfg)
            smrt = boto3.client("sagemaker-runtime", region_name=region, config=cfg)

            payload = {
                "task": "pre_annotate",
                "job_id": f"preannot-{labeler}-{project_pk}-{video_pk}",
                "user_id": str(user_id),
                "video_blob_path": blob_path,
                "classes": classes,
                "sample_interval": sample_interval,
                "max_frames": max_frames,
                "confidence_threshold": confidence,
                "selection": selection,  # SAM 3: "diverse" (DINOv2) or "uniform"
                "nms_iou": nms_iou,          # SAM 3: dedupe overlapping boxes (<=0 off)
                "max_detections": max_detections,  # SAM 3: cap per class per frame
            }
            key = f"preannotate/{project_pk}/{video_pk}-{labeler}.json"
            s3.put_object(Bucket=in_bucket, Key=key,
                          Body=json.dumps(payload).encode("utf-8"),
                          ContentType="application/json")
            resp = smrt.invoke_endpoint_async(
                EndpointName=endpoint,
                InputLocation=f"s3://{in_bucket}/{key}",
                ContentType="application/json",
                InferenceId=f"preannot-{labeler}-{project_pk}-{video_pk}",
            )
            out = urlparse(resp["OutputLocation"])
            out_bucket, out_key = out.netloc, out.path.lstrip("/")
            fail_key = out_key.replace(".out", ".failure")

            # Poll the async output (cold start can take a few minutes).
            result = None
            deadline = time.time() + 15 * 60
            while time.time() < deadline:
                time.sleep(10)
                try:
                    body = s3.get_object(Bucket=out_bucket, Key=out_key)["Body"].read()
                    result = json.loads(body)
                    break
                except ClientError as e:
                    if e.response["Error"]["Code"] not in ("NoSuchKey", "404", "NotFound"):
                        raise
                    try:
                        fail = s3.get_object(Bucket=out_bucket, Key=fail_key)["Body"].read()
                        logger.error("pre-annotate: endpoint failure for video %s: %s",
                                     video_pk, fail[:500])
                        return
                    except ClientError:
                        continue  # still running

            if not result or not result.get("frames"):
                logger.info("pre-annotate: no frames for video %s (timeout or empty)", video_pk)
                return

            project = AnnotationProject.objects.get(pk=project_pk)
            video = Video.objects.get(pk=video_pk)
            width = result.get("video_width", 1280)
            height = result.get("video_height", 720)
            # Never overwrite frames a human already reviewed; re-detected
            # boxes invalidate any prior LLM review, so reset review state.
            human_frames = set(
                Annotation.objects.filter(
                    project=project, video=video,
                    review_source=Annotation.ReviewSource.HUMAN,
                ).values_list("frame_number", flat=True)
            )
            created = 0
            skipped = 0
            for frame_data in result["frames"]:
                if frame_data["frame_number"] in human_frames:
                    skipped += 1
                    continue
                _, was_created = Annotation.objects.update_or_create(
                    project=project, video=video,
                    frame_number=frame_data["frame_number"],
                    defaults={
                        "boxes": frame_data["boxes"],
                        "image_width": width, "image_height": height,
                        "frame_image_path": frame_data.get("frame_image_path", ""),
                        "reviewed": False,
                        "review_source": Annotation.ReviewSource.NONE,
                        "reviewed_at": None,
                    },
                )
                created += int(was_created)
            logger.info("pre-annotated video %s: %d frames, %d new, %d human-reviewed kept",
                        video_pk, len(result["frames"]), created, skipped)

            exec_secs = result.get("execution_seconds", 0) or 0
            credits_used = int(exec_secs) if exec_secs else 30
            try:
                from apps.accounts.models import UserProfile
                profile, _ = UserProfile.objects.get_or_create(user=project.user)
                profile.charge(credits_used, gpu_seconds=exec_secs or credits_used)
            except Exception as charge_err:
                logger.error("pre-annotate: credit charge failed for %s: %s", video_pk, charge_err)

        except Exception as e:
            logger.error("pre-annotate failed for video %s: %s", video_pk, e, exc_info=True)
        finally:
            connection.close()


class PreAnnotateAllView(LoginRequiredMixin, View):
    """Run AI pre-annotation on all videos in a project, or on the subset
    checked in the video list (video_ids)."""

    def post(self, request, pk):
        from django.shortcuts import redirect
        from django.contrib import messages
        import threading

        project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
        videos = project.videos.all()
        video_ids = request.POST.getlist("video_ids")
        if video_ids:
            videos = videos.filter(pk__in=video_ids)

        if not videos.exists():
            messages.warning(request, "No videos in this project.")
            return redirect("annotations:detail", pk=pk)

        sample_interval, max_frames, confidence = _preannotate_opts(request)
        labeler = "sam3" if request.POST.get("labeler") == "sam3" else "yolo"
        selection = "diverse" if request.POST.get("selection", "diverse") == "diverse" else "uniform"
        if labeler != "sam3":
            selection = "uniform"
        nms_iou, max_detections = _sam3_opts(request)
        count = 0
        for video in videos:
            blob_path = video.storage_key
            if not blob_path:
                continue
            thread = threading.Thread(
                target=PreAnnotateView._run_pre_annotate,
                args=(project.pk, video.pk, blob_path, sample_interval, max_frames, confidence,
                      labeler, selection, nms_iou, max_detections),
                daemon=True,
            )
            thread.start()
            count += 1

        engine = "SAM 3" if labeler == "sam3" else "AI"
        messages.info(
            request,
            f"{engine} pre-annotation started for {count} video(s). This takes 1-4 minutes per video. Refresh to see results.",
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

        try:
            project = get_object_or_404(AnnotationProject, pk=pk, user=request.user)
            ann = Annotation.objects.get(project=project, video=video, frame_number=frame_number)

            if ann.frame_image_path:
                return self._serve_from_storage(ann.frame_image_path, ann if draw_boxes else None)
        except Annotation.DoesNotExist:
            pass

        return self._extract_from_video(video, frame_number)

    def _serve_from_storage(self, blob_path, ann_for_boxes=None):
        """Serve a pre-saved JPEG frame from the processed S3 bucket."""
        try:
            import io
            from config.storage import get_s3_client

            buf = io.BytesIO()
            get_s3_client().download_to_stream("processed", blob_path, buf)
            data = buf.getvalue()

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
            logger.error("FrameImageView S3 error: %s", e)
            return HttpResponse(status=404)

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

            # Group annotations and extract frames from videos
            from config.storage import get_s3_client
            s3 = get_s3_client()
            video_cache = {}  # video_pk -> (cv2.VideoCapture, tmp_path)

            ann_list = list(annotations)
            split_idx = int(len(ann_list) * 0.8)

            for i, ann in enumerate(ann_list):
                split = "train" if i < split_idx else "val"
                base_name = f"{ann.video.title}_f{ann.frame_number:06d}"
                label_name = f"labels/{split}/{base_name}.txt"

                yolo_txt = ann.to_yolo_format()
                zf.writestr(label_name, yolo_txt)

                try:
                    if ann.video.pk not in video_cache:
                        blob_path = ann.video.storage_key
                        if blob_path and not blob_path.startswith("s3://"):
                            import tempfile, cv2

                            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                            tmp.close()
                            s3.download_file("raw-videos", blob_path, tmp.name)
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
