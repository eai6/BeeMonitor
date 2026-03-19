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
    epochs = models.IntegerField(default=50)
    image_size = models.IntegerField(default=640)
    batch_size = models.IntegerField(default=16)
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
    azure_model_path = models.CharField(max_length=500, blank=True)
    classes = models.JSONField(default=list)
    metrics = models.JSONField(default=dict, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.model_type})"
