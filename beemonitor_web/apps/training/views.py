import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

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

# Cap concurrent SageMaker training-job spawns (each holds a DB conn briefly).
_TRAIN_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="train-spawn")

# Map the user-facing GPU tier to a SageMaker *training* instance type.
_INSTANCE_BY_TIER = {
    "T4": "ml.g4dn.xlarge",
    "L4": "ml.g6.xlarge",
    "A10G": "ml.g5.xlarge",
    "L40S": "ml.g6e.xlarge",
    "A100": "ml.p4d.24xlarge",
}
_DEFAULT_INSTANCE = "ml.g5.xlarge"
_MAX_TRAIN_SECONDS = 4 * 60 * 60  # SageMaker StoppingCondition cap


def _boto3(service: str):
    import boto3
    from botocore.config import Config
    return boto3.client(
        service, region_name=getattr(settings, "AWS_REGION", "us-east-1"),
        config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 2}),
    )


# ── Helpers ──────────────────────────────────────────────────────────


def _yolo_label_subset(ann, name_to_new_id: dict) -> str:
    """YOLO label lines for only the boxes whose class is in the subset,
    renumbered to the compact 0..k ids in ``name_to_new_id`` (keyed by class
    name). Boxes of other classes are dropped; an all-dropped frame yields ""
    (a valid negative)."""
    lines = []
    for box in (ann.boxes or []):
        name = box.get("class")
        if name not in name_to_new_id:
            continue
        cx = (box["x"] + box["w"] / 2) / ann.image_width
        cy = (box["y"] + box["h"] / 2) / ann.image_height
        w = box["w"] / ann.image_width
        h = box["h"] / ann.image_height
        lines.append(f"{name_to_new_id[name]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines)


def _build_training_payload(job: TrainingJob) -> tuple[list, list[dict]]:
    """Gather annotations from the project and build the training payload.

    Returns (effective_classes, video_annotations). effective_classes is the
    project's classes, or the job's class subset (with labels remapped) when set.
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
    if job.frame_filter == TrainingJob.FrameFilter.REVIEWED:
        annotations = annotations.filter(reviewed=True)
    elif job.frame_filter == TrainingJob.FrameFilter.HUMAN:
        annotations = annotations.filter(review_source=Annotation.ReviewSource.HUMAN)

    ann_count = annotations.count()
    logger.info("[train:%s] Found %d annotations in project '%s' (filter=%s)",
                job.pk, ann_count, project.name, job.frame_filter)

    if ann_count == 0:
        logger.error("[train:%s] No annotations found — aborting", job.pk)
        raise ValueError(
            f"No annotations match the '{job.get_frame_filter_display()}' filter in this project")

    # Build dataset YAML. A class subset trains a model on only those classes:
    # drop out-of-subset boxes and renumber the kept classes to a compact 0..k.
    all_classes = project.classes or ["bee", "wasp", "nest"]
    subset = [c for c in all_classes if c in (job.class_subset or [])]
    if subset:
        classes = subset
        name_to_new_id = {c: i for i, c in enumerate(subset)}
        logger.info("[train:%s] Class subset (%d of %d): %s",
                    job.pk, len(subset), len(all_classes), subset)
    else:
        classes = all_classes
        name_to_new_id = None
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
        yolo_label = _yolo_label_subset(ann, name_to_new_id) if name_to_new_id is not None else ann.to_yolo_format()
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

    # Return the EFFECTIVE classes (subset-aware) so the manifest matches the
    # remapped labels; dataset_yaml is built by the container from these.
    return classes, video_annotations


def _spawn_training_job(job_pk: int) -> None:
    """Create a SageMaker training job for this TrainingJob (background thread).

    Writes the manifest to the SageMaker input bucket and calls
    create_training_job; the training container (Dockerfile.training) reads the
    manifest, trains, and writes best.pt + result.json. The SageMaker
    TrainingJobName is stored in ``modal_call_id`` (legacy field name) for polling.
    """
    from django.db import connection

    try:
        job = TrainingJob.objects.select_related("project").get(pk=job_pk)
    except TrainingJob.DoesNotExist:
        logger.error("[train:%s] Job not found in DB — aborting", job_pk)
        return

    if not settings.SAGEMAKER_TRAINING_ROLE_ARN or not settings.SAGEMAKER_TRAINING_IMAGE:
        TrainingJob.objects.filter(pk=job_pk).update(
            status=TrainingJob.Status.FAILED,
            error_message="SageMaker training is not configured (role/image env unset).",
        )
        connection.close()
        return

    try:
        classes, video_annotations = _build_training_payload(job)
        model_key = f"custom/{job.user_id}/{job.pk}/best.pt"
        result_key = f"training/{job.pk}/result.json"
        manifest = {
            "job_id": str(job.pk),
            "user_id": str(job.user_id),
            "base_model": job.base_model,
            "init_weights_key": job.init_weights_key,  # fine-tune source (models bucket), "" = scratch
            "epochs": job.epochs,
            "imgsz": job.image_size,
            "batch_size": job.batch_size,
            "val_percent": job.val_percent,
            "classes": classes,
            "video_annotations": video_annotations,
            "model_key": model_key,
            "result_key": result_key,
        }

        input_bucket = settings.SAGEMAKER_INPUT_BUCKET
        manifest_prefix = f"training/{job.pk}/"
        _boto3("s3").put_object(
            Bucket=input_bucket, Key=f"{manifest_prefix}manifest.json",
            Body=json.dumps(manifest).encode("utf-8"), ContentType="application/json",
        )

        job_name = f"bm-train-{job.pk}-{uuid.uuid4().hex[:8]}"  # <=63 chars, [A-Za-z0-9-]
        instance = _INSTANCE_BY_TIER.get(job.gpu_tier, _DEFAULT_INSTANCE)
        _boto3("sagemaker").create_training_job(
            TrainingJobName=job_name,
            RoleArn=settings.SAGEMAKER_TRAINING_ROLE_ARN,
            AlgorithmSpecification={
                "TrainingImage": settings.SAGEMAKER_TRAINING_IMAGE,
                "TrainingInputMode": "File",
            },
            InputDataConfig=[{
                "ChannelName": "manifest",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{input_bucket}/{manifest_prefix}",
                    "S3DataDistributionType": "FullyReplicated",
                }},
            }],
            OutputDataConfig={"S3OutputPath": f"s3://{settings.SAGEMAKER_OUTPUT_BUCKET}/training-artifacts/"},
            ResourceConfig={"InstanceType": instance, "InstanceCount": 1, "VolumeSizeInGB": 100},
            StoppingCondition={"MaxRuntimeInSeconds": _MAX_TRAIN_SECONDS},
            Environment={
                "AWS_S3_BUCKET_RAW_VIDEOS": settings.AWS_S3_BUCKET_RAW_VIDEOS,
                "AWS_S3_BUCKET_MODELS": settings.AWS_S3_BUCKET_MODELS,
                "SAGEMAKER_OUTPUT_BUCKET": settings.SAGEMAKER_OUTPUT_BUCKET,
                "AWS_REGION": settings.AWS_REGION,
            },
        )

        TrainingJob.objects.filter(pk=job_pk).update(
            status=TrainingJob.Status.PREPARING,
            modal_call_id=job_name,
            started_at=timezone.now(),
        )
        logger.info("[train:%s] SageMaker training job created: %s (%s)", job_pk, job_name, instance)

    except Exception as e:
        logger.exception("[train:%s] SPAWN FAILED — %s: %s", job_pk, type(e).__name__, e)
        TrainingJob.objects.filter(pk=job_pk).update(
            status=TrainingJob.Status.FAILED,
            error_message=str(e)[:1000],
        )
    finally:
        connection.close()


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

    def get_context_data(self, **kwargs):
        import json as _json
        from apps.annotations.models import AnnotationProject
        ctx = super().get_context_data(**kwargs)
        # {project_pk: [class names]} so the Classes checkboxes can react to the
        # chosen project without a round-trip.
        ctx["project_classes_json"] = _json.dumps({
            str(p.pk): (p.classes or [])
            for p in AnnotationProject.objects.filter(user=self.request.user)
        })
        return ctx

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
        logger.info("[train:%s] TrainingJob saved (status=%s) — dispatching to SageMaker",
                    self.object.pk, self.object.status)

        # Create the SageMaker training job off-request (bounded pool).
        _TRAIN_POOL.submit(_spawn_training_job, self.object.pk)

        messages.info(
            self.request,
            f"Training job '{self.object.name}' submitted to GPU. This page auto-refreshes.",
        )
        return response

    def get_success_url(self):
        return reverse_lazy("training:detail", kwargs={"pk": self.object.pk})


_W, _H, _PAD = 560, 180, 28  # epoch-chart SVG geometry


def _line_series(em, keys_colors, y_max):
    """Build inline-SVG polyline points for each metric series over epochs.
    keys_colors: list of (key, label, color). Returns series with a `points`
    string sized to the (_W, _H) viewbox and the last value."""
    n = len(em)
    plot_w, plot_h = _W - 2 * _PAD, _H - 2 * _PAD
    series = []
    for key, label, color in keys_colors:
        pts = []
        last = None
        for i, row in enumerate(em):
            v = row.get(key)
            if v is None:
                continue
            x = _PAD + (plot_w * i / (n - 1) if n > 1 else plot_w / 2)
            y = _PAD + plot_h * (1 - (v / y_max if y_max else 0))
            pts.append(f"{x:.1f},{y:.1f}")
            last = v
        if pts:
            series.append({"label": label, "color": color,
                           "points": " ".join(pts), "last": last,
                           "single": len(pts) == 1})
    return series


def _build_epoch_charts(em):
    """Two charts (detection metrics 0–1, and losses) from epoch_metrics."""
    loss_vals = [row[k] for row in em for k in ("box_loss", "cls_loss")
                 if row.get(k) is not None]
    loss_max = max(loss_vals) * 1.1 if loss_vals else 1.0
    return {
        "w": _W, "h": _H, "pad": _PAD,
        "epochs": [row["epoch"] for row in em],
        "detection": _line_series(em, [
            ("mAP50", "mAP50", "#d97706"),
            ("mAP50_95", "mAP50-95", "#ea580c"),
            ("precision", "Precision", "#2563eb"),
            ("recall", "Recall", "#16a34a"),
            ("f1", "F1", "#9333ea"),
        ], 1.0),
        "detection_ymax": 1.0,
        "loss": _line_series(em, [
            ("box_loss", "Box loss", "#dc2626"),
            ("cls_loss", "Cls loss", "#64748b"),
        ], loss_max),
        "loss_ymax": round(loss_max, 3),
    }


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

        # Rendered best.pt predictions on the held-out val split (uploaded by
        # the training container); presign so the gallery can show them.
        pred_keys = (self.object.metrics or {}).get("val_predictions") or []
        if pred_keys:
            try:
                s3 = _boto3("s3")
                ctx["val_predictions"] = [
                    {
                        "name": key.rsplit("/", 1)[-1],
                        "url": s3.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": settings.SAGEMAKER_OUTPUT_BUCKET, "Key": key},
                            ExpiresIn=3600,
                        ),
                    }
                    for key in pred_keys
                ]
            except Exception as e:
                logger.warning("[train:%s] presigning val predictions failed: %s",
                               self.object.pk, e)

        # Per-epoch metric curves (present on jobs trained after the container
        # update that emits epoch_metrics).
        em = (self.object.metrics or {}).get("epoch_metrics") or []
        if em:
            ctx["epoch_charts"] = _build_epoch_charts(em)
        ctx["best_epoch"] = (self.object.metrics or {}).get("best_epoch")
        return ctx


_ACTIVE_TRAINING_STATUSES = [
    TrainingJob.Status.QUEUED,
    TrainingJob.Status.PREPARING,
    TrainingJob.Status.TRAINING,
]


def poll_training_jobs(user=None) -> dict:
    """Poll SageMaker for active training jobs and finalize completed ones.

    Reusable by the page-load view and the background reconciler (so training
    completes without an open browser). ``user=None`` polls all users.
    Idempotent; never raises.
    """
    qs = TrainingJob.objects.filter(status__in=_ACTIVE_TRAINING_STATUSES).exclude(modal_call_id="")
    if user is not None:
        qs = qs.filter(user=user)
    active_jobs = list(qs.order_by("-created_at")[:20])
    if not active_jobs:
        return {"checked": 0, "completed": 0}

    sm = _boto3("sagemaker")
    s3 = _boto3("s3")
    completed = 0

    for job in active_jobs:
        try:
            d = sm.describe_training_job(TrainingJobName=job.modal_call_id)
        except Exception as e:
            logger.warning("[poll] job=%s describe failed: %s", job.pk, e)
            continue

        st = d.get("TrainingJobStatus", "")
        if st == "InProgress":
            secondary = d.get("SecondaryStatus", "")
            new_status = (
                TrainingJob.Status.TRAINING
                if secondary in ("Training", "Uploading")
                else TrainingJob.Status.PREPARING
            )
            if job.status != new_status:
                TrainingJob.objects.filter(pk=job.pk).update(status=new_status)
            continue

        if st in ("Failed", "Stopped", "Stopping"):
            TrainingJob.objects.filter(pk=job.pk).update(
                status=TrainingJob.Status.FAILED,
                error_message=d.get("FailureReason", f"training {st.lower()}")[:1000],
                completed_at=timezone.now(),
            )
            completed += 1
            continue

        if st != "Completed":
            continue

        # Completed — read the result.json the container wrote to the output bucket.
        result = {}
        try:
            body = s3.get_object(
                Bucket=settings.SAGEMAKER_OUTPUT_BUCKET,
                Key=f"training/{job.pk}/result.json",
            )["Body"].read()
            result = json.loads(body)
        except Exception as e:
            logger.warning("[poll] job=%s no result.json yet: %s", job.pk, e)

        if result.get("status") == "failed":
            TrainingJob.objects.filter(pk=job.pk).update(
                status=TrainingJob.Status.FAILED,
                error_message=str(result.get("error", "training failed"))[:1000],
                completed_at=timezone.now(),
            )
            completed += 1
            continue

        metrics = result.get("metrics", {}) or {}
        for extra in ("train_count", "val_count", "val_predictions",
                      "epoch_metrics", "best_epoch"):
            if result.get(extra) is not None:
                metrics[extra] = result[extra]
        storage_key = result.get("storage_key", "")
        exec_secs = float(result.get("execution_seconds") or d.get("TrainingTimeInSeconds") or 0)

        TrainingJob.objects.filter(pk=job.pk).update(
            status=TrainingJob.Status.COMPLETED,
            completed_at=timezone.now(),
            execution_seconds=exec_secs,
            metrics=metrics,
        )

        if storage_key:
            CustomModel.objects.get_or_create(
                training_job=job,
                defaults={
                    "user": job.user,
                    "name": f"{job.name} (trained)",
                    "model_type": CustomModel.ModelType.CUSTOM,
                    # Fine-tunes inherit the source's architecture, not the
                    # placeholder arch stored on the job — label the lineage.
                    "base_model": (f"fine-tuned: {job.init_from_label}"[:50]
                                   if job.init_weights_key else job.base_model),
                    "storage_key": storage_key,
                    # Subset jobs produce a model with only the subset's classes.
                    "classes": [c for c in (job.project.classes or []) if c in job.class_subset]
                               or (job.project.classes or []),
                    "metrics": metrics,
                    "status": CustomModel.Status.READY,
                    "is_active": True,
                },
            )
            logger.info("[poll] job=%s COMPLETED -> CustomModel (%s)", job.pk, storage_key)
        else:
            logger.warning("[poll] job=%s completed but no storage_key in result", job.pk)

        credits_used = int(exec_secs)
        if credits_used > 0:
            try:
                from apps.accounts.models import UserProfile
                profile, _ = UserProfile.objects.get_or_create(user=job.user)
                profile.charge(credits_used, gpu_seconds=exec_secs)
            except Exception as e:
                logger.error("[poll] job=%s credit charge failed: %s", job.pk, e)

        completed += 1

    return {"checked": len(active_jobs), "completed": completed}


class PollTrainingJobsView(LoginRequiredMixin, View):
    """JSON endpoint hit by the training pages to advance active jobs."""

    def get(self, request):
        return JsonResponse(poll_training_jobs(request.user))


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


# --- Domain-drift detection (memory/25, P2c) --------------------------------

class DriftDashboardView(LoginRequiredMixin, View):
    """Baseline + recent drift checks; pick known-good videos as the baseline and
    score other videos against it to flag domain shift (the fine-tuning trigger)."""

    def get(self, request):
        from django.shortcuts import render

        from apps.videos.models import Video
        from .models import DriftCheck, DriftReference

        ref = DriftReference.objects.filter(user=request.user, scope="default").first()
        checks = DriftCheck.objects.filter(user=request.user).select_related("video")[:20]
        videos = Video.objects.filter(user=request.user).order_by("-id")[:100]
        return render(request, "training/drift.html",
                      {"ref": ref, "checks": checks, "videos": videos})


class ToggleAutoAdaptView(LoginRequiredMixin, View):
    """Flip the baseline's auto-adapt flag (drift → auto-start adaptation)."""

    def post(self, request):
        from django.shortcuts import redirect
        from .models import DriftReference

        ref = DriftReference.objects.filter(user=request.user, scope="default").first()
        if ref:
            ref.auto_adapt = request.POST.get("auto_adapt") in ("on", "true", "1")
            ref.save(update_fields=["auto_adapt"])
            state = "on" if ref.auto_adapt else "off"
            messages.info(request, f"Auto-adaptation turned {state}. Promotion stays user-approved.")
        else:
            messages.warning(request, "Set a baseline first.")
        return redirect("training:drift")


class SetDriftBaselineView(LoginRequiredMixin, View):
    def post(self, request):
        import threading

        from apps.videos.models import Video
        from . import drift

        ids = request.POST.getlist("video_ids")
        videos = Video.objects.filter(user=request.user, pk__in=ids)
        paths = [v.storage_key for v in videos if v.storage_key]
        if not paths:
            messages.warning(request, "Select at least one known-good video for the baseline.")
            return redirect("training:drift")
        threading.Thread(
            target=drift.build_reference, args=(request.user.id, paths),
            kwargs={"note": f"{len(paths)} videos"}, daemon=True,
        ).start()
        messages.info(request, f"Building drift baseline from {len(paths)} video(s) with DINOv2. This takes a few minutes; refresh to see it.")
        return redirect("training:drift")


class CheckDriftView(LoginRequiredMixin, View):
    def post(self, request):
        import threading

        from django.shortcuts import get_object_or_404

        from apps.videos.models import Video
        from . import drift
        from .models import DriftCheck, DriftReference

        video = get_object_or_404(Video, pk=request.POST.get("video_id"), user=request.user)
        ref = DriftReference.objects.filter(user=request.user, scope="default").first()
        if not ref:
            messages.warning(request, "Set a baseline first, then check videos against it.")
            return redirect("training:drift")
        check = DriftCheck.objects.create(user=request.user, reference=ref, video=video)
        threading.Thread(target=drift.run_check, args=(check.id,), daemon=True).start()
        messages.info(request, f"Checking '{video.title}' for domain drift. Refresh in a minute.")
        return redirect("training:drift")


# --- Closed auto-fine-tuning loop (memory/25, P3) ---------------------------

class AdaptationDashboardView(LoginRequiredMixin, View):
    """List adaptation runs, advance in-flight ones, and approve promotions."""

    def get(self, request):
        from django.shortcuts import render

        from . import orchestrator
        from .models import AdaptationRun

        orchestrator.advance_user_runs(request.user)
        runs = AdaptationRun.objects.filter(user=request.user).select_related(
            "project", "training_job", "candidate_model")[:30]
        return render(request, "training/adaptation.html", {"runs": runs})


class StartAdaptationView(LoginRequiredMixin, View):
    def post(self, request):
        from . import orchestrator

        ids = request.POST.getlist("video_ids") or (
            [request.POST["video_id"]] if request.POST.get("video_id") else [])
        scope = request.POST.get("scope") or "default"
        if not ids:
            messages.warning(request, "Select at least one video to adapt to.")
            return redirect(request.META.get("HTTP_REFERER", "training:adaptation"))
        run = orchestrator.start_run(request.user, ids, scope=scope)
        if run:
            messages.info(request, "Adaptation started: SAM 3 is relabeling, then it will fine-tune. Refresh to follow progress.")
        else:
            messages.warning(request, "No usable videos (missing storage) for adaptation.")
        return redirect("training:adaptation")


class PromoteAdaptationView(LoginRequiredMixin, View):
    def post(self, request):
        from django.shortcuts import get_object_or_404

        from . import orchestrator
        from .models import AdaptationRun

        run = get_object_or_404(AdaptationRun, pk=request.POST.get("run_id"), user=request.user)
        orchestrator.promote_run(run)
        if run.status == AdaptationRun.Status.PROMOTED:
            messages.success(request, f"Promoted — '{run.candidate_model.name}' is now the active model.")
        else:
            messages.warning(request, "Run is not awaiting approval; nothing promoted.")
        return redirect("training:adaptation")
