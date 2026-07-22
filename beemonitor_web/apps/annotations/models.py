from django.conf import settings
from django.db import models


class AnnotationProject(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="annotation_projects",
    )
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    classes = models.JSONField(
        default=list,
        help_text='List of class labels, e.g. ["bee", "wasp", "nest"]',
    )
    videos = models.ManyToManyField(
        "videos.Video",
        blank=True,
        related_name="annotation_projects",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.classes:
            self.classes = ["bee", "wasp", "nest"]
        super().save(*args, **kwargs)


class Annotation(models.Model):
    project = models.ForeignKey(
        AnnotationProject,
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    video = models.ForeignKey(
        "videos.Video",
        on_delete=models.CASCADE,
        related_name="annotations",
    )
    frame_number = models.IntegerField()
    image_width = models.IntegerField(default=1280)
    image_height = models.IntegerField(default=720)
    boxes = models.JSONField(
        default=list,
        help_text='List of {"x": float, "y": float, "w": float, "h": float, "class": str, "class_id": int}',
    )
    frame_image_path = models.CharField(
        max_length=500, blank=True, default="",
        help_text="S3 key (in processed bucket) of extracted frame JPEG",
    )

    class ReviewSource(models.TextChoices):
        NONE = "", "Not reviewed"
        HUMAN = "human", "Human"
        LLM = "llm", "LLM"

    reviewed = models.BooleanField(
        default=False, help_text="A human or the LLM has vetted these boxes.")
    sampled_only = models.BooleanField(
        default=False,
        help_text="This row is a sampled frame nobody has annotated yet — it "
                  "exists so the editor can navigate to the frame. Distinct from "
                  "a frame a human deliberately marked empty, which IS a valid "
                  "negative example; training excludes these but keeps those.",
    )
    review_source = models.CharField(
        max_length=10, choices=ReviewSource.choices, default="", blank=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = [("project", "video", "frame_number")]
        ordering = ["frame_number"]

    def __str__(self):
        return f"Frame {self.frame_number} of {self.video.title}"

    def to_yolo_format(self) -> str:
        """Convert boxes to YOLO txt format: class_id cx cy w h (normalized)."""
        lines = []
        for box in self.boxes:
            cx = (box["x"] + box["w"] / 2) / self.image_width
            cy = (box["y"] + box["h"] / 2) / self.image_height
            w = box["w"] / self.image_width
            h = box["h"] / self.image_height
            lines.append(f'{box["class_id"]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}')
        return "\n".join(lines)


class PreAnnotationTask(models.Model):
    """A durable record of one pre-annotation run (one video).

    Replaces the old fire-and-forget daemon thread that polled S3 for 15 min:
    the SageMaker invocation is fired once, its output S3 URI is stored here,
    and the background reconciler finalizes the task (writes Annotation rows)
    when the result lands — so a deploy mid-run no longer loses the result and
    the UI can show progress / failure.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"          # created, not yet invoked
        PROCESSING = "processing", "Processing"  # invoked, awaiting result
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"  # user cancelled; any result is discarded

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="preannotation_tasks")
    project = models.ForeignKey(AnnotationProject, on_delete=models.CASCADE,
                                related_name="preannotation_tasks")
    video = models.ForeignKey("videos.Video", on_delete=models.CASCADE,
                              related_name="preannotation_tasks")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    labeler = models.CharField(max_length=20, default="yolo")   # yolo | sam3 (custom rides yolo)
    custom_model_key = models.CharField(max_length=500, blank=True, default="")
    # Sampling / SAM3 knobs needed to build the payload at spawn time.
    params = models.JSONField(default=dict, blank=True)
    # SageMaker async result location (poll target) + failure location.
    output_uri = models.CharField(max_length=700, blank=True, default="")
    failure_uri = models.CharField(max_length=700, blank=True, default="")
    frames_written = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"PreAnnotate {self.video_id} [{self.status}]"


class FrameSamplingTask(models.Model):
    """Extract frames from one video so they can be annotated.

    Sampling used to be fused into the SAM 3 pre-annotation call: the only way to
    get frames was to pay for a g5 auto-labelling pass over the whole video. This
    task decouples the two — it decodes frames on the web container's CPU and
    writes an empty ``Annotation`` row per frame, so the editor can navigate them
    immediately and SAM 3 becomes an opt-in, per-frame action.

    Deliberately shaped like ``PreAnnotationTask`` (durable row, advanced by the
    background reconciler) so the two read the same way on the project page.
    """

    class Status(models.TextChoices):
        QUEUED = "queued", "Queued"
        PROCESSING = "processing", "Processing"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"
        CANCELLED = "cancelled", "Cancelled"

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name="frame_sampling_tasks")
    project = models.ForeignKey(AnnotationProject, on_delete=models.CASCADE,
                                related_name="frame_sampling_tasks")
    video = models.ForeignKey("videos.Video", on_delete=models.CASCADE,
                              related_name="frame_sampling_tasks")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.QUEUED)
    # {"sample_interval": int, "max_frames": int}
    params = models.JSONField(default=dict, blank=True)
    frames_written = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"SampleFrames {self.video_id} [{self.status}]"
