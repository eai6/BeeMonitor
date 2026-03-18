import logging
import os

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.utils import timezone
from django.views.generic import FormView, ListView

from cryptography.fernet import Fernet
import json as _json

from .forms import SourceCreateForm
from .models import DataSource

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """Get Fernet cipher from the app's credential key.

    SOC2 CC6.1: Encryption of sensitive data at rest.
    The key MUST be set via BEEMONITOR_CREDENTIAL_KEY env var in production.
    """
    key = getattr(settings, "BEEMONITOR_CREDENTIAL_KEY", "") or os.environ.get("BEEMONITOR_CREDENTIAL_KEY", "")
    if not key:
        raise ValueError(
            "BEEMONITOR_CREDENTIAL_KEY is not set. "
            "Cannot encrypt credentials without an encryption key."
        )
    return Fernet(key.encode() if isinstance(key, str) else key)


def _mask_secret(value: str) -> str:
    """Mask a secret for safe logging. Shows first 4 and last 2 chars."""
    if not value or len(value) < 8:
        return "***"
    return f"{value[:4]}...{value[-2:]}"


class SourceListView(LoginRequiredMixin, ListView):
    template_name = "sources/list.html"
    context_object_name = "sources"

    def get_queryset(self):
        return DataSource.objects.filter(user=self.request.user)


class SourceCreateView(LoginRequiredMixin, FormView):
    template_name = "sources/add.html"
    form_class = SourceCreateForm
    success_url = reverse_lazy("sources:list")

    def form_valid(self, form):
        credentials = form.get_credentials()
        source_type = form.cleaned_data["source_type"]

        # SOC2 CC6.1: Encrypt credentials at rest with Fernet (AES-128-CBC + HMAC)
        try:
            fernet = _get_fernet()
        except ValueError as e:
            messages.error(self.request, f"Security configuration error: {e}")
            return self.form_invalid(form)

        encrypted = fernet.encrypt(_json.dumps(credentials).encode())

        source = DataSource.objects.create(
            user=self.request.user,
            name=form.cleaned_data["name"],
            source_type=source_type,
            config_encrypted=encrypted,
            is_connected=False,
        )

        # SOC2 CC7.2: Audit log — record credential creation (never log the actual secret)
        logger.info(
            "AUDIT: User %s (id=%s) created data source '%s' (type=%s, id=%s)",
            self.request.user.username,
            self.request.user.pk,
            source.name,
            source_type,
            source.pk,
        )

        # Test connection
        try:
            ok, msg = self._test_connection(source_type, credentials)
            source.is_connected = ok
            source.save(update_fields=["is_connected"])

            # SOC2 CC7.2: Log connection test result
            logger.info(
                "AUDIT: Connection test for source %s (id=%s): %s — %s",
                source.name, source.pk, "SUCCESS" if ok else "FAILED", msg,
            )

            if ok:
                messages.success(self.request, f"Source '{source.name}' connected: {msg}")
            else:
                messages.warning(self.request, f"Source saved but connection failed: {msg}")
        except Exception as e:
            logger.error(
                "AUDIT: Connection test error for source %s (id=%s): %s",
                source.name, source.pk, e,
            )
            messages.warning(self.request, f"Source saved but connection test failed: {e}")

        # SOC2 CC6.7: Clear credentials from memory
        credentials.clear()

        return super().form_valid(form)

    def _test_connection(self, source_type: str, credentials: dict) -> tuple[bool, str]:
        """Test if the credentials work.

        SOC2 CC6.1: Credentials are used only for the test and never logged.
        """
        try:
            if source_type == "aws_s3":
                import boto3
                client = boto3.client(
                    "s3",
                    aws_access_key_id=credentials["access_key_id"],
                    aws_secret_access_key=credentials["secret_access_key"],
                    region_name=credentials.get("region", "us-east-1"),
                )
                client.head_bucket(Bucket=credentials["bucket"])
                return True, f"Connected to s3://{credentials['bucket']}"

            elif source_type == "azure_blob":
                from azure.storage.blob import BlobServiceClient
                service = BlobServiceClient.from_connection_string(credentials["connection_string"])
                container = service.get_container_client(credentials["container"])
                container.get_container_properties()
                return True, f"Connected to Azure container: {credentials['container']}"

            elif source_type == "gcs":
                return True, "GCS credentials saved (test on first use)"

            elif source_type == "google_drive":
                return True, "Google Drive credentials saved (test on first use)"

            return False, "Unknown source type"
        except Exception as e:
            # SOC2: Never log the full exception if it might contain credentials
            error_msg = str(e)
            # Strip any credential-like strings from error messages
            for key in ("access_key_id", "secret_access_key", "connection_string", "oauth_token"):
                if key in credentials and credentials[key] in error_msg:
                    error_msg = error_msg.replace(credentials[key], "***REDACTED***")
            return False, error_msg
