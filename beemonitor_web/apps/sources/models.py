from django.conf import settings
from django.db import models


class DataSource(models.Model):
    class SourceType(models.TextChoices):
        AWS_S3 = "aws_s3", "AWS S3"
        AZURE_BLOB = "azure_blob", "Azure Blob"
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
