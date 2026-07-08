import hashlib
import json

from django.conf import settings
from django.db import models


# GPU tiers with cost per second
GPU_TIERS = {
    "T4": {"label": "T4 (Budget)", "cost_per_sec": 0.000164, "speed": "~10 min/video"},
    "L4": {"label": "L4 (Standard)", "cost_per_sec": 0.000222, "speed": "~8 min/video"},
    "A10G": {"label": "A10G (Fast)", "cost_per_sec": 0.000306, "speed": "~5.5 min/video"},
    "L40S": {"label": "L40S (Faster)", "cost_per_sec": 0.000542, "speed": "~3.5 min/video"},
    "A100": {"label": "A100 (Fastest)", "cost_per_sec": 0.000583, "speed": "~3 min/video"},
}


def compute_config_hash(video_id: int, config: dict) -> str:
    """SHA256 hash of video_id + sorted config. Used to detect duplicate analysis."""
    payload = json.dumps({"video_id": video_id, **config}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


class Job(models.Model):
    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        INGESTING = "ingesting", "Ingesting"
        PROCESSING = "processing", "Processing"
        POST_PROCESSING = "post_processing", "Post-Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    class GpuTier(models.TextChoices):
        T4 = "T4", "T4 (Budget)"
        L4 = "L4", "L4 (Standard)"
        A10G = "A10G", "A10G (Fast)"
        L40S = "L40S", "L40S (Faster)"
        A100 = "A100", "A100 (Fastest)"

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
    config_hash = models.CharField(max_length=32, blank=True, db_index=True,
                                   help_text="SHA256 hash for deduplication")
    gpu_tier = models.CharField(max_length=10, choices=GpuTier.choices, default=GpuTier.A10G)
    progress_pct = models.IntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    execution_seconds = models.FloatField(null=True, blank=True, help_text="Actual GPU seconds consumed")
    compute_cost_usd = models.DecimalField(
        max_digits=8,
        decimal_places=4,
        null=True,
        blank=True,
    )
    created_at = models.DateTimeField(auto_now_add=True)

    def save(self, *args, **kwargs):
        if not self.config_hash and self.video_id and self.config:
            self.config_hash = compute_config_hash(self.video_id, self.config)
        super().save(*args, **kwargs)

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
    foraging_trips_csv_path = models.CharField(max_length=500, blank=True)
    interactions_csv_path = models.CharField(max_length=500, blank=True)
    crops_csv_path = models.CharField(max_length=500, blank=True)
    annotated_video_path = models.CharField(max_length=500, blank=True)
    total_events = models.IntegerField(default=0)
    entry_count = models.IntegerField(default=0)
    exit_count = models.IntegerField(default=0)
    unique_tracks = models.IntegerField(default=0)
    nest_count = models.IntegerField(default=0)
    foraging_trip_count = models.IntegerField(default=0)
    avg_trip_duration_sec = models.FloatField(null=True, blank=True)
    interaction_count = models.IntegerField(default=0)
    summary_stats = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Result for Job {self.job_id} ({self.total_events} events)"


class DailyForagingSummary(models.Model):
    """Aggregated foraging trips across all videos for a site+day.

    Detects cross-video trips where a bee exits in one video and enters
    in the next video at the same nest.
    """

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="daily_foraging",
    )
    site_name = models.CharField(max_length=200)
    # The device whose videos this summary aggregates. Null for legacy rows /
    # videos with no device. Cross-video trips are detected within one device
    # (a single camera), so device-keyed aggregation keeps that valid.
    device = models.ForeignKey(
        "devices.Device",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="daily_foraging",
    )
    date = models.DateField()
    total_trips = models.IntegerField(default=0)
    cross_video_trips = models.IntegerField(default=0)
    avg_duration_sec = models.FloatField(default=0)
    median_duration_sec = models.FloatField(default=0)
    trips_per_nest = models.JSONField(default=dict, blank=True)
    trips_csv_path = models.CharField(max_length=500, blank=True)
    video_count = models.IntegerField(default=0)
    computed_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "site_name", "device", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"{self.site_name} {self.date}: {self.total_trips} trips ({self.cross_video_trips} cross-video)"
