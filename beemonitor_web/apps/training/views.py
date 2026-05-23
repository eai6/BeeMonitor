import logging
import threading
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import JsonResponse
from django.shortcuts import redirect
from django.urls import reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import CreateView, DetailView, FormView, ListView

from .forms import TrainingCreateForm, ModelUploadForm
from .models import CustomModel, TrainingJob

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def _build_training_payload(job: TrainingJob) -> tuple[str, list[dict]]:
    """Gather annotations from the project and build Modal payload.

    Returns (dataset_yaml_content, video_annotations).
    """
    from apps.annotations.models import Annotation

    project = job.project
    logger.info(
        "[train:%s] Building payload — project=%s (pk=%s), user=%s",
        job.pk, project.name, project.pk, job.user_id,
    )

    annotations = (
        Annotation.objects.filter(project=project)
        .select_related("video")
        .order_by("video__title", "frame_number")
    )

    ann_count = annotations.count()
    logger.info("[train:%s] Found %d annotations in project '%s'", job.pk, ann_count, project.name)

    if ann_count == 0:
        logger.error("[train:%s] No annotations found — aborting", job.pk)
        raise ValueError("No annotations found in this project")

    # Build dataset YAML
    classes = project.classes or ["bee", "wasp", "nest"]
    logger.info("[train:%s] Classes (%d): %s", job.pk, len(classes), classes)
    class_lines = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    dataset_yaml = (
        f"path: .\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"nc: {len(classes)}\n"
        f"names:\n{class_lines}\n"
    )
    logger.debug("[train:%s] Dataset YAML:\n%s", job.pk, dataset_yaml)

    # Group annotations by video
    video_groups: dict[int, dict] = {}
    total_frames = 0
    for ann in annotations:
        vid_pk = ann.video_id
        if vid_pk not in video_groups:
            video_groups[vid_pk] = {
                "video_blob_path": ann.video.storage_key,
                "frames": [],
            }
            logger.info(
                "[train:%s] Video pk=%s title='%s' blob_path='%s'",
                job.pk, vid_pk, ann.video.title, ann.video.storage_key,
            )
        base_name = f"{ann.video.title}_f{ann.frame_number:06d}"
        yolo_label = ann.to_yolo_format()
        box_count = len(ann.boxes) if ann.boxes else 0
        video_groups[vid_pk]["frames"].append({
            "frame_number": ann.frame_number,
            "filename": f"{base_name}.jpg",
            "label": yolo_label,
        })
        total_frames += 1
        logger.debug(
            "[train:%s] Annotation: video=%s frame=%d boxes=%d label_lines=%d",
            job.pk, vid_pk, ann.frame_number, box_count, len(yolo_label.strip().split("\n")) if yolo_label.strip() else 0,
        )

    video_annotations = list(video_groups.values())
    logger.info(
        "[train:%s] Payload ready — %d videos, %d total frames, 80/20 split => ~%d train / ~%d val",
        job.pk, len(video_annotations), total_frames,
        int(total_frames * 0.8), total_frames - int(total_frames * 0.8),
    )

    return dataset_yaml, video_annotations


def _spawn_training_job(job_pk: int) -> None:
    """Spawn the Modal training function in a background thread."""
    import django
    django.setup()
    from django.db import connection

    logger.info("[train:%s] Background thread started — loading job from DB", job_pk)

    try:
        job = TrainingJob.objects.select_related("project").get(pk=job_pk)
    except TrainingJob.DoesNotExist:
        logger.error("[train:%s] Job not found in DB — aborting", job_pk)
        return

    logger.info(
        "[train:%s] Job loaded — name='%s' base_model=%s epochs=%d imgsz=%d batch=%d gpu=%s project='%s'",
        job_pk, job.name, job.base_model, job.epochs, job.image_size,
        job.batch_size, job.gpu_tier, job.project.name,
    )

    try:
        logger.info("[train:%s] Building training payload...", job_pk)
        dataset_yaml, video_annotations = _build_training_payload(job)

        logger.info("[train:%s] Importing modal SDK...", job_pk)
        import modal

        logger.info("[train:%s] Looking up Modal function 'beemonitor-cloud/train_yolo_model'...", job_pk)
        fn = modal.Function.from_name("beemonitor-cloud", "train_yolo_model")
        logger.info("[train:%s] Modal function resolved — spawning async call...", job_pk)

        call = fn.spawn(
            job_id=str(job.pk),
            user_id=str(job.user_id),
            base_model=job.base_model,
            dataset_yaml_content=dataset_yaml,
            video_annotations=video_annotations,
            epochs=job.epochs,
            imgsz=job.image_size,
            batch_size=job.batch_size,
        )

        modal_call_id = call.object_id
        logger.info("[train:%s] Modal call spawned — call_id=%s", job_pk, modal_call_id)

        TrainingJob.objects.filter(pk=job_pk).update(
            status=TrainingJob.Status.PREPARING,
            modal_call_id=modal_call_id,
            started_at=timezone.now(),
        )
        logger.info(
            "[train:%s] DB updated — status=preparing, modal_call_id=%s, started_at=%s",
            job_pk, modal_call_id, timezone.now().isoformat(),
        )

    except Exception as e:
        logger.exception("[train:%s] SPAWN FAILED — %s: %s", job_pk, type(e).__name__, e)
        TrainingJob.objects.filter(pk=job_pk).update(
            status=TrainingJob.Status.FAILED,
            error_message=str(e),
        )
        logger.info("[train:%s] DB updated — status=failed, error_message='%s'", job_pk, str(e)[:200])
    finally:
        connection.close()
        logger.info("[train:%s] Background thread finished — DB connection closed", job_pk)


# ── Views ────────────────────────────────────────────────────────────


class TrainingListView(LoginRequiredMixin, ListView):
    model = TrainingJob
    template_name = "training/list.html"
    context_object_name = "jobs"
    paginate_by = 20

    def get_queryset(self):
        return TrainingJob.objects.filter(user=self.request.user).select_related("project")


class TrainingCreateView(LoginRequiredMixin, CreateView):
    model = TrainingJob
    form_class = TrainingCreateForm
    template_name = "training/new.html"

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs["user"] = self.request.user
        return kwargs

    def form_valid(self, form):
        form.instance.user = self.request.user
        logger.info(
            "[train:NEW] form_valid called — user=%s project=%s name='%s' base_model=%s epochs=%d imgsz=%d batch=%d gpu=%s",
            self.request.user.pk,
            form.cleaned_data.get("project"),
            form.cleaned_data.get("name"),
            form.cleaned_data.get("base_model"),
            form.cleaned_data.get("epochs", 50),
            form.cleaned_data.get("image_size", 640),
            form.cleaned_data.get("batch_size", 16),
            form.cleaned_data.get("gpu_tier", "A10G"),
        )

        response = super().form_valid(form)
        logger.info("[train:%s] TrainingJob saved to DB — pk=%s status=%s", self.object.pk, self.object.pk, self.object.status)

        # Dispatch to Modal in a background thread
        logger.info("[train:%s] Launching background thread for Modal dispatch...", self.object.pk)
        thread = threading.Thread(
            target=_spawn_training_job,
            args=(self.object.pk,),
            daemon=True,
        )
        thread.start()
        logger.info("[train:%s] Background thread started (thread=%s)", self.object.pk, thread.name)

        messages.info(
            self.request,
            f"Training job '{self.object.name}' submitted to GPU. This page auto-refreshes.",
        )
        return response

    def get_success_url(self):
        return reverse_lazy("training:detail", kwargs={"pk": self.object.pk})


class TrainingDetailView(LoginRequiredMixin, DetailView):
    model = TrainingJob
    template_name = "training/detail.html"
    context_object_name = "job"

    def get_queryset(self):
        return TrainingJob.objects.filter(user=self.request.user).select_related("project")

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        try:
            ctx["custom_model"] = self.object.custom_model
        except CustomModel.DoesNotExist:
            ctx["custom_model"] = None
        return ctx


class PollTrainingJobsView(LoginRequiredMixin, View):
    """HTMX/JSON endpoint: check Modal for completed training jobs."""

    def get(self, request):
        active_jobs = TrainingJob.objects.filter(
            user=request.user,
            status__in=[
                TrainingJob.Status.QUEUED,
                TrainingJob.Status.PREPARING,
                TrainingJob.Status.TRAINING,
            ],
        ).exclude(modal_call_id="")

        active_count = active_jobs.count()
        logger.info(
            "[poll] user=%s — %d active training job(s) with modal_call_id",
            request.user.pk, active_count,
        )

        if active_count == 0:
            # Also check for stuck queued jobs without modal_call_id
            stuck_jobs = TrainingJob.objects.filter(
                user=request.user,
                status=TrainingJob.Status.QUEUED,
                modal_call_id="",
            )
            stuck_count = stuck_jobs.count()
            if stuck_count:
                logger.warning(
                    "[poll] user=%s has %d queued job(s) with NO modal_call_id (stuck?): pks=%s",
                    request.user.pk, stuck_count,
                    list(stuck_jobs.values_list("pk", flat=True)[:10]),
                )
            return JsonResponse({"checked": 0, "completed": 0})

        completed = 0
        try:
            import modal

            for job in active_jobs[:10]:
                logger.info(
                    "[poll] Checking job pk=%s — status=%s, modal_call_id=%s, created=%s",
                    job.pk, job.status, job.modal_call_id, job.created_at.isoformat(),
                )
                try:
                    fc = modal.functions.FunctionCall.from_id(job.modal_call_id)
                    logger.debug("[poll] job=%s — FunctionCall handle obtained", job.pk)

                    try:
                        result = fc.get(timeout=0)
                    except TimeoutError:
                        # Still running — update status to training if needed
                        if job.status != TrainingJob.Status.TRAINING:
                            TrainingJob.objects.filter(pk=job.pk).update(
                                status=TrainingJob.Status.TRAINING,
                            )
                            logger.info("[poll] job=%s — still running, updated status queued/preparing -> training", job.pk)
                        else:
                            logger.info("[poll] job=%s — still running (status=training)", job.pk)
                        continue
                    except modal.exception.ExecutionError as e:
                        logger.error("[poll] job=%s — Modal execution error: %s", job.pk, e)
                        TrainingJob.objects.filter(pk=job.pk).update(
                            status=TrainingJob.Status.FAILED,
                            error_message=str(e),
                        )
                        continue

                    logger.info("[poll] job=%s — got result from Modal: type=%s", job.pk, type(result).__name__)

                    if not isinstance(result, dict):
                        logger.error("[poll] job=%s — unexpected result type: %s, value=%s", job.pk, type(result).__name__, str(result)[:500])
                        continue

                    logger.info("[poll] job=%s — result keys: %s", job.pk, list(result.keys()))

                    # Check for errors from the Modal function
                    if result.get("error"):
                        logger.error("[poll] job=%s — Modal function returned error: %s", job.pk, result["error"])
                        TrainingJob.objects.filter(pk=job.pk).update(
                            status=TrainingJob.Status.FAILED,
                            error_message=result["error"],
                        )
                        continue

                    metrics = result.get("metrics", {})
                    exec_secs = result.get("execution_seconds", 0) or 0
                    storage_key = result.get("storage_key", "")
                    epochs_completed = result.get("epochs_completed", 0)

                    logger.info(
                        "[poll] job=%s — COMPLETED: mAP50=%.4f, mAP50-95=%.4f, precision=%.4f, recall=%.4f, "
                        "exec_secs=%.1f, epochs=%d, model_path='%s'",
                        job.pk,
                        metrics.get("mAP50", 0), metrics.get("mAP50_95", 0),
                        metrics.get("precision", 0), metrics.get("recall", 0),
                        exec_secs, epochs_completed, storage_key,
                    )

                    TrainingJob.objects.filter(pk=job.pk).update(
                        status=TrainingJob.Status.COMPLETED,
                        completed_at=timezone.now(),
                        execution_seconds=exec_secs,
                        metrics=metrics,
                    )
                    logger.info("[poll] job=%s — DB updated to completed", job.pk)

                    # Create CustomModel record if we got a model path
                    if storage_key:
                        cm = CustomModel.objects.create(
                            user=job.user,
                            training_job=job,
                            name=f"{job.name} (trained)",
                            model_type=CustomModel.ModelType.CUSTOM,
                            base_model=job.base_model,
                            storage_key=storage_key,
                            classes=job.project.classes or [],
                            metrics=metrics,
                            is_active=True,
                        )
                        logger.info(
                            "[poll] job=%s — CustomModel created: pk=%s name='%s' path='%s'",
                            job.pk, cm.pk, cm.name, storage_key,
                        )
                    else:
                        logger.warning("[poll] job=%s — no model path returned, skipping CustomModel creation", job.pk)

                    # Charge credits (1 credit ≈ 1 GPU-second)
                    credits_used = int(exec_secs)
                    if credits_used > 0:
                        try:
                            from apps.accounts.models import UserProfile
                            profile, _ = UserProfile.objects.get_or_create(user=job.user)
                            profile.charge(credits_used, gpu_seconds=exec_secs)
                            logger.info("[poll] job=%s — charged %d credits (%.1fs GPU)", job.pk, credits_used, exec_secs)
                        except Exception as e:
                            logger.error("[poll] job=%s — failed to charge credits: %s", job.pk, e)

                    completed += 1

                except Exception as e:
                    logger.exception("[poll] job=%s — unexpected error: %s: %s", job.pk, type(e).__name__, e)

        except ImportError as e:
            logger.error("[poll] Failed to import modal SDK: %s", e)

        logger.info("[poll] Poll complete — checked=%d, completed=%d", active_count, completed)
        return JsonResponse({
            "checked": active_count,
            "completed": completed,
        })


class CustomModelListView(LoginRequiredMixin, ListView):
    model = CustomModel
    template_name = "training/models.html"
    context_object_name = "models_list"
    paginate_by = 20

    def get_queryset(self):
        return CustomModel.objects.filter(user=self.request.user).select_related("training_job")


class CustomModelDetailView(LoginRequiredMixin, DetailView):
    model = CustomModel
    template_name = "training/model_detail.html"
    context_object_name = "model"

    def get_queryset(self):
        return CustomModel.objects.filter(user=self.request.user).select_related("training_job")


class UploadModelView(LoginRequiredMixin, FormView):
    """Upload a custom .pt model file."""
    template_name = "training/upload_model.html"
    form_class = ModelUploadForm

    def form_valid(self, form):
        model_file = self.request.FILES["model_file"]
        name = form.cleaned_data["name"]
        model_type = form.cleaned_data["model_type"]
        classes_text = form.cleaned_data.get("classes", "")
        classes = [c.strip() for c in classes_text.split(",") if c.strip()] if classes_text else []

        logger.info(
            "[upload] user=%s uploading model '%s' type=%s file=%s size=%d",
            self.request.user.pk, name, model_type, model_file.name, model_file.size,
        )

        # Upload to S3 models bucket
        upload_id = uuid.uuid4().hex[:12]
        blob_path = f"custom/{self.request.user.pk}/{upload_id}/{model_file.name}"

        try:
            from config.storage import get_s3_client
            get_s3_client().upload_stream(
                "models", blob_path, model_file,
                content_type="application/octet-stream",
            )
            logger.info("[upload] Uploaded to S3 models: %s", blob_path)
        except Exception as e:
            logger.exception("[upload] Upload failed: %s", e)
            messages.error(self.request, f"Upload failed: {e}")
            return self.form_invalid(form)

        CustomModel.objects.create(
            user=self.request.user,
            name=name,
            model_type=model_type,
            base_model="uploaded",
            storage_key=blob_path,
            classes=classes,
            metrics={"source": "uploaded", "file_size": model_file.size},
            is_active=True,
        )

        logger.info("[upload] CustomModel created for user=%s name='%s'", self.request.user.pk, name)
        messages.success(self.request, f"Model '{name}' uploaded successfully.")
        return redirect("training:models")
