from django.conf import settings
from django.db import models


class DataSource(models.Model):
    class SourceType(models.TextChoices):
        AWS_S3 = "aws_s3", "AWS S3"
        GCS = "gcs", "Google Cloud Storage"
        GOOGLE_DRIVE = "google_drive", "Google Drive"

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="data_sources",
    )
    name = models.CharField(max_length=200)
    source_type = models.CharField(
        max_length=20,
        choices=SourceType.choices,
    )
    config_encrypted = models.BinaryField()
    is_connected = models.BooleanField(default=False)
    last_synced_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "Data Source"
        verbose_name_plural = "Data Sources"

    def __str__(self):
        return f"{self.name} ({self.get_source_type_display()})"


class SourceCopy(models.Model):
    """One clip being copied from a user's bucket into ours (memory/54).

    ``keep``: the user chose "Copy into BeeMonitor"; the clip then lives in our
    storage. ``temp``: a clip kept in the user's bucket is being analysed; the
    copy feeds the GPU and is deleted a day later. ``measure``: nothing is
    copied; the clip's duration, frame size and thumbnail are read from the
    user's bucket so it can be listed, priced and drawn on.

    Run by apps/sources/copying.py from the reconcile loop, a few at a time,
    each claimed with one conditional UPDATE so a clip is never copied twice.
    """

    class Purpose(models.TextChoices):
        KEEP = "keep", "Copy into BeeMonitor"
        TEMP = "temp", "Copy for one analysis"
        MEASURE = "measure", "Measure"

    class Status(models.TextChoices):
        QUEUED = "queued", "Waiting"
        RUNNING = "running", "Copying"
        DONE = "done", "Done"
        FAILED = "failed", "Failed"
        REMOVED = "removed", "Removed"

    video = models.ForeignKey("videos.Video", on_delete=models.CASCADE, related_name="source_copies")
    source = models.ForeignKey(DataSource, null=True, blank=True, on_delete=models.SET_NULL,
                               related_name="copies")
    purpose = models.CharField(max_length=10, choices=Purpose.choices)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.QUEUED, db_index=True)
    bytes_total = models.BigIntegerField(default=0)
    bytes_done = models.BigIntegerField(default=0)
    dest_key = models.CharField(max_length=500, blank=True)
    error = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    # Touched while a copy makes progress; one that stops is put back.
    heartbeat_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.video_id} {self.purpose} {self.status}"

    @property
    def percent(self) -> int:
        return int(self.bytes_done * 100 / self.bytes_total) if self.bytes_total else 0
