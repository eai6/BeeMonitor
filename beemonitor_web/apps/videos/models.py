from django.conf import settings
from django.db import models


class Video(models.Model):
    class Status(models.TextChoices):
        UPLOADING = "uploading", "Uploading"
        READY = "ready", "Ready"
        PROCESSING = "processing", "Processing"
        ARCHIVED = "archived", "Archived"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="videos",
    )
    source = models.ForeignKey(
        "sources.DataSource",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="videos",
    )
    title = models.CharField(max_length=300)
    azure_blob_path = models.CharField(max_length=500)
    file_size_bytes = models.BigIntegerField()
    duration_seconds = models.FloatField(null=True, blank=True)
    resolution = models.CharField(max_length=20, blank=True)
    fps = models.FloatField(null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.UPLOADING,
    )
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return f"{self.title} ({self.get_status_display()})"
