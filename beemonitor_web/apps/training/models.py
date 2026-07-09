from django.conf import settings
from django.db import models

from apps.analysis.models import Job


class TrainingJob(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        PREPARING = "preparing", "Preparing"
        TRAINING = "training", "Training"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    class BaseModel(models.TextChoices):
        YOLOV8N = "yolov8n", "YOLOv8n (Nano)"
        YOLOV8S = "yolov8s", "YOLOv8s (Small)"
        YOLOV8M = "yolov8m", "YOLOv8m (Medium)"
        YOLOV11N = "yolov11n", "YOLOv11n (Nano)"
        YOLOV11S = "yolov11s", "YOLOv11s (Small)"

    class FrameFilter(models.TextChoices):
        ALL = "all", "All annotated frames"
        REVIEWED = "reviewed", "Reviewed frames only (human + LLM)"
        HUMAN = "human", "Human-reviewed frames only"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="training_jobs",
    )
    project = models.ForeignKey(
        "annotations.AnnotationProject",
        on_delete=models.CASCADE,
        related_name="training_jobs",
    )
    name = models.CharField(max_length=200)
    base_model = models.CharField(max_length=50, choices=BaseModel.choices)
    frame_filter = models.CharField(
        max_length=10, choices=FrameFilter.choices, default=FrameFilter.ALL,
        help_text="Which of the project's annotated frames go into the dataset.",
    )
    init_weights_key = models.CharField(
        max_length=500, blank=True, default="",
        help_text="S3 key (models bucket) of .pt weights to fine-tune from; "
                  "empty = train from the pretrained base architecture.",
    )
    init_from_label = models.CharField(
        max_length=200, blank=True, default="",
        help_text="Human-readable name of the fine-tune source model.",
    )
    epochs = models.IntegerField(default=50)
    image_size = models.IntegerField(default=640)
    batch_size = models.IntegerField(default=16)
    class_subset = models.JSONField(
        default=list, blank=True,
        help_text="Subset of the project's classes to train on. Empty = all classes.",
    )
    frames_with_class = models.BooleanField(
        default=False,
        help_text="Only include frames that contain at least one box of the "
                  "selected class(es) — skip frames without any.",
    )
    val_percent = models.IntegerField(
        default=20,
        help_text="Percent of frames held out for validation (the metrics set).",
    )
    gpu_tier = models.CharField(
        max_length=10,
        choices=Job.GpuTier.choices,
        default=Job.GpuTier.A10G,
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.QUEUED,
    )
    modal_call_id = models.CharField(max_length=200, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text="Training metrics: mAP50, mAP50-95, precision, recall, best_epoch",
    )
    execution_seconds = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.status})"


class CustomModel(models.Model):
    class ModelType(models.TextChoices):
        NEST_DETECTION = "nest_detection", "Nest Detection"
        BEE_TRACKING = "bee_tracking", "Bee Tracking"
        CUSTOM = "custom", "Custom"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="custom_models",
    )
    training_job = models.OneToOneField(
        TrainingJob,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="custom_model",
    )
    name = models.CharField(max_length=200)
    model_type = models.CharField(
        max_length=20,
        choices=ModelType.choices,
        default=ModelType.CUSTOM,
    )
    base_model = models.CharField(max_length=50)
    storage_key = models.CharField(max_length=500, blank=True)
    classes = models.JSONField(default=list)
    metrics = models.JSONField(default=dict, blank=True)

    class Status(models.TextChoices):
        READY = "ready", "Ready"
        TRAINING = "training", "Training"
        FAILED = "failed", "Failed"

    # Lifecycle: uploads land READY; training-produced models flip READY on
    # success. Distinct from is_active (user enable/disable for selection).
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.READY,
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.model_type})"


class DriftReference(models.Model):
    """A baseline DINOv3 distribution for a user's 'known-good' footage. New videos are
    scored against it to detect domain shift — the auto-fine-tuning trigger (memory/25).
    Stores only the centroid + spread (cheap), not the raw frame embeddings."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name="drift_references",
    )
    scope = models.CharField(max_length=100, default="default")  # site/device or "default"
    centroid = models.JSONField(default=list)   # L2-normalised mean embedding
    ref_mean = models.FloatField(default=0.0)   # mean cosine distance of ref frames -> centroid
    ref_std = models.FloatField(default=0.0)    # std of those distances (the "normal" spread)
    dim = models.IntegerField(default=0)
    n_frames = models.IntegerField(default=0)
    # Opt-in: when a drift check against this baseline flags drift, automatically
    # start an AdaptationRun (SAM3 relabel → fine-tune → await approval). Off by
    # default — it spends GPU; promotion is still user-approved.
    auto_adapt = models.BooleanField(default=False)
    note = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("user", "scope")]
        ordering = ["-updated_at"]

    def __str__(self):
        return f"DriftReference({self.user_id}/{self.scope}, n={self.n_frames})"


class DriftCheck(models.Model):
    """Result of scoring one video's DINOv3 embeddings against a DriftReference."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        DONE = "done", "Done"
        ERROR = "error", "Error"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="drift_checks",
    )
    reference = models.ForeignKey(
        DriftReference, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="checks",
    )
    video = models.ForeignKey(
        "videos.Video", on_delete=models.CASCADE, related_name="drift_checks",
    )
    drift_score = models.FloatField(default=0.0)  # mean cosine distance to reference centroid
    z_score = models.FloatField(default=0.0)      # (drift_score - ref_mean) / ref_std
    is_drifted = models.BooleanField(default=False)
    n_frames = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PENDING)
    detail = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"DriftCheck(video={self.video_id}, z={self.z_score:.1f}, drifted={self.is_drifted})"


class AdaptationRun(models.Model):
    """A closed auto-fine-tuning cycle (memory/25, P3): drift → SAM 3 relabel → train
    → evaluate → (user-approved) promote. Chains the existing annotation + training +
    custom-model pieces; advanced by a state machine (orchestrator.advance_run)."""

    class Status(models.TextChoices):
        RELABELING = "relabeling", "Relabeling (SAM 3)"
        TRAINING = "training", "Training"
        EVALUATING = "evaluating", "Evaluating"
        AWAITING = "awaiting_approval", "Awaiting approval"
        PROMOTED = "promoted", "Promoted"
        HELD = "held", "Held (below bar)"
        FAILED = "failed", "Failed"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="adaptation_runs",
    )
    scope = models.CharField(max_length=100, default="default")
    project = models.ForeignKey(
        "annotations.AnnotationProject", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="adaptation_runs",
    )
    training_job = models.ForeignKey(
        TrainingJob, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="adaptation_runs",
    )
    candidate_model = models.ForeignKey(
        CustomModel, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="adaptation_runs",
    )
    status = models.CharField(max_length=24, choices=Status.choices, default=Status.RELABELING)
    min_map50 = models.FloatField(default=0.5)  # promotion guardrail
    detail = models.CharField(max_length=255, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"AdaptationRun({self.scope}, {self.status})"
