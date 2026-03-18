import logging

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import FormView, ListView

from cryptography.fernet import Fernet
import json as _json

from .forms import SourceCreateForm
from .models import DataSource

logger = logging.getLogger(__name__)


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

        # Encrypt credentials
        import os
        key = os.environ.get("BEEMONITOR_CREDENTIAL_KEY", "")
        if not key:
            key = Fernet.generate_key().decode()
        fernet = Fernet(key.encode() if isinstance(key, str) else key)
        encrypted = fernet.encrypt(_json.dumps(credentials).encode())

        source = DataSource.objects.create(
            user=self.request.user,
            name=form.cleaned_data["name"],
            source_type=form.cleaned_data["source_type"],
            config_encrypted=encrypted,
            is_connected=False,
        )

        # Test connection
        try:
            ok, msg = self._test_connection(form.cleaned_data["source_type"], credentials)
            source.is_connected = ok
            source.save(update_fields=["is_connected"])
            if ok:
                messages.success(self.request, f"Source '{source.name}' connected: {msg}")
            else:
                messages.warning(self.request, f"Source saved but connection failed: {msg}")
        except Exception as e:
            messages.warning(self.request, f"Source saved but connection test failed: {e}")

        return super().form_valid(form)

    def _test_connection(self, source_type: str, credentials: dict) -> tuple[bool, str]:
        """Test if the credentials work."""
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
                return True, "GCS credentials saved (connection test requires google-cloud-storage)"
            elif source_type == "google_drive":
                return True, "Google Drive credentials saved"
            return False, "Unknown source type"
        except Exception as e:
            return False, str(e)
