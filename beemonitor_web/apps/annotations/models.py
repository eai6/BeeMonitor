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
        help_text="Azure blob path to extracted frame JPEG",
    )
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
