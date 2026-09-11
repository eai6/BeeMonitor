import hashlib
import json

from django.conf import settings
from django.db import models


# Display labels for the hardware a job ran on. The per-second prices that used
# to live here were Modal's, from before the AWS migration, and were applied to
# a tier the user picked rather than the GPU that ran — see
# apps/analysis/pricing.py, which now owns all cost arithmetic.
GPU_TIERS = {
    "T4": {"label": "T4 (Budget)", "speed": "~10 min/video"},
    "L4": {"label": "L4 (Standard)", "speed": "~8 min/video"},
    "A10G": {"label": "A10G (Fast)", "speed": "~5.5 min/video"},
    "L40S": {"label": "L40S (Faster)", "speed": "~3.5 min/video"},
    "A100": {"label": "A100 (Fastest)", "speed": "~3 min/video"},
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
    # Set from the hardware the run REPORTS, not from a user's choice. It was a
    # dropdown that never reached SageMaker: every analysis job runs on whatever
    # the endpoint is pinned to, so picking "A100" only inflated the bill.
    gpu_tier = models.CharField(max_length=10, choices=GpuTier.choices, default=GpuTier.T4)
    progress_pct = models.IntegerField(default=0)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    # Whole-handler wall time: S3 transfer + decode + inference + encode +
    # upload. This is what the instance was busy for, so it is what gets billed
    # (apps/analysis/pricing.py).
    execution_seconds = models.FloatField(
        null=True, blank=True, help_text="Handler wall seconds — what is billed")
    # Inference only. The diagnostic half of the pair: gpu_seconds well under
    # execution_seconds means the run was bound by decode or S3, not the GPU.
    gpu_seconds = models.FloatField(
        null=True, blank=True, help_text="GPU inference seconds (subset of execution_seconds)")
    # Per-stage breakdown from the worker: {stage: {seconds, calls}}. JSON so a
    # new stage is not a migration.
    stage_seconds = models.JSONField(default=dict, blank=True)
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
    # Raw per-frame detector output, before track association. Distinct from
    # tracking_csv_path, which only holds detections the tracker associated into
    # a confirmed track — so counts derived from the two legitimately differ.
    detections_csv_path = models.CharField(max_length=500, blank=True)
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
    # The day's trips paired UNFILTERED (bounds 0–86400s) as compact rows
    # [exit_epoch_sec, duration_sec, nest, is_cross_video], so charts can apply
    # user min/max bounds as a pure read-time filter with no S3 access. NULL =
    # not yet computed by the new path (the reconciler sweep fills it in).
    # NOTE: total_trips/avg/median above stay at the DEFAULT bounds (10/7200),
    # so total_trips != len(trips) by design.
    trips = models.JSONField(null=True, blank=True)
    # Dirty flag: set by job completion, cleared by the recompute sweep. The
    # guarded clear (filter on stale_marked_at) makes concurrent re-marks safe
    # without locks — a row re-marked mid-recompute stays stale for next tick.
    stale = models.BooleanField(default=False)
    stale_marked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ("user", "site_name", "device", "date")
        ordering = ["-date"]

    def __str__(self):
        return f"{self.site_name} {self.date}: {self.total_trips} trips ({self.cross_video_trips} cross-video)"
