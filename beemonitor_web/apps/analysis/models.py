from django.conf import settings
from django.db import models


class Job(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        INGESTING = "ingesting", "Ingesting"
        PROCESSING = "processing", "Processing"
        POST_PROCESSING = "post_processing", "Post-Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="jobs",
    )
    video = models.ForeignKey(
        "videos.Video",
        on_delete=models.CASCADE,
        related_name="jobs",
    )
    modal_job_id = models.CharField(max_length=100, unique=True, blank=True)
    modal_call_id = models.CharField(max_length=200, blank=True, help_text="Modal FunctionCall ID for async polling")
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.QUEUED,
    )
    config = models.JSONField(default=dict, blank=True)
    progress_pct = models.IntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    compute_cost_usd = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"Job {self.pk} - {self.get_status_display()} ({self.video.title})"


class JobResult(models.Model):
    job = models.OneToOneField(
        Job,
        on_delete=models.CASCADE,
        related_name="result",
    )
    events_csv_path = models.CharField(max_length=500, blank=True)
    tracking_csv_path = models.CharField(max_length=500, blank=True)
    annotated_video_path = models.CharField(max_length=500, blank=True)
    total_events = models.IntegerField(default=0)
    entry_count = models.IntegerField(default=0)
    exit_count = models.IntegerField(default=0)
    unique_tracks = models.IntegerField(default=0)
    nest_count = models.IntegerField(default=0)
    summary_stats = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Result for Job {self.job_id} ({self.total_events} events)"
