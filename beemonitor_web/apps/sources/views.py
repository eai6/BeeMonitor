import logging
import os
import uuid

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import get_object_or_404, redirect
from django.urls import reverse, reverse_lazy
from django.utils import timezone
from django.views import View
from django.views.generic import DetailView, FormView, ListView

from cryptography.fernet import Fernet
import json as _json

from apps.videos.models import Video

from .forms import SourceCreateForm
from .models import DataSource

logger = logging.getLogger(__name__)


def _get_fernet() -> Fernet:
    """SOC2 CC6.1: Encryption of sensitive data at rest."""
    key = getattr(settings, "BEEMONITOR_CREDENTIAL_KEY", "") or os.environ.get("BEEMONITOR_CREDENTIAL_KEY", "")
    if not key:
        raise ValueError("BEEMONITOR_CREDENTIAL_KEY is not set.")
    return Fernet(key.encode() if isinstance(key, str) else key)


def _decrypt_credentials(source: DataSource) -> dict:
    """Decrypt stored credentials for a source."""
    fernet = _get_fernet()
    return _json.loads(fernet.decrypt(bytes(source.config_encrypted)).decode())


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

        logger.info(
            "AUDIT: User %s (id=%s) created data source '%s' (type=%s, id=%s)",
            self.request.user.username, self.request.user.pk,
            source.name, source_type, source.pk,
        )

        try:
            ok, msg = _test_connection(source_type, credentials)
            source.is_connected = ok
            source.save(update_fields=["is_connected"])
            if ok:
                messages.success(self.request, f"Source '{source.name}' connected: {msg}")
            else:
                messages.warning(self.request, f"Source saved but connection failed: {msg}")
        except Exception as e:
            messages.warning(self.request, f"Source saved but connection test failed: {e}")

        credentials.clear()
        return super().form_valid(form)


class SourceDeleteView(LoginRequiredMixin, View):
    """Delete a data source and its encrypted credentials."""

    def post(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        name = source.name
        source.delete()
        logger.info(
            "AUDIT: User %s (id=%s) deleted data source '%s' (id=%s)",
            request.user.username, request.user.pk, name, pk,
        )
        messages.success(request, f"Source '{name}' deleted.")
        return redirect("sources:list")


class SourceBrowseView(LoginRequiredMixin, DetailView):
    """Browse files in a connected source and optionally import them."""
    template_name = "sources/browse.html"
    context_object_name = "source"

    def get_queryset(self):
        return DataSource.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        source = self.object
        prefix = self.request.GET.get("prefix", "")
        ctx["prefix"] = prefix
        ctx["files"] = []
        ctx["error"] = ""

        try:
            credentials = _decrypt_credentials(source)
            files = _list_files(source.source_type, credentials, prefix=prefix, limit=200)
            credentials.clear()
            ctx["files"] = files
        except Exception as e:
            ctx["error"] = str(e)

        return ctx


class SourceSyncView(LoginRequiredMixin, View):
    """Import selected files from a source into BeeMonitor as Video records."""

    def post(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        selected_files = request.POST.getlist("selected_files")

        if not selected_files:
            messages.warning(request, "No files selected.")
            return redirect("sources:browse", pk=pk)

        try:
            credentials = _decrypt_credentials(source)
        except Exception as e:
            messages.error(request, f"Could not decrypt credentials: {e}")
            return redirect("sources:browse", pk=pk)

        imported = 0
        for file_key in selected_files:
            filename = file_key.split("/")[-1] if "/" in file_key else file_key

            # Skip non-video files
            if not any(filename.lower().endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
                continue

            # Check if already imported
            if Video.objects.filter(user=request.user, azure_blob_path__endswith=filename).exists():
                continue

            # Parse timestamp from filename
            site_name, recorded_at = Video.parse_timestamp_from_filename(filename)
            title = filename.rsplit(".", 1)[0] if "." in filename else filename

            # Create video record (not yet downloaded to Azure — just registered)
            Video.objects.create(
                user=request.user,
                source=source,
                title=title,
                azure_blob_path=f"s3://{file_key}",  # Mark as S3 reference
                file_size_bytes=0,
                status=Video.Status.UPLOADING,
                recorded_at=recorded_at,
                site_name=site_name,
                metadata={
                    "source_id": source.pk,
                    "source_type": source.source_type,
                    "remote_key": file_key,
                },
            )
            imported += 1

        credentials.clear()

        source.last_synced_at = timezone.now()
        source.save(update_fields=["last_synced_at"])

        logger.info(
            "AUDIT: User %s synced %d files from source '%s' (id=%s)",
            request.user.username, imported, source.name, pk,
        )
        messages.success(request, f"Imported {imported} video(s) from '{source.name}'.")
        return redirect("videos:list")


# ── Helpers ──────────────────────────────────────────────────────────

def _test_connection(source_type: str, credentials: dict) -> tuple[bool, str]:
    """Test if credentials work. SOC2: credentials never logged."""
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
            return True, "GCS credentials saved"
        elif source_type == "google_drive":
            return True, "Google Drive credentials saved"
        return False, "Unknown source type"
    except Exception as e:
        error_msg = str(e)
        for key in ("access_key_id", "secret_access_key", "connection_string", "oauth_token"):
            if key in credentials and credentials[key] in error_msg:
                error_msg = error_msg.replace(credentials[key], "***REDACTED***")
        return False, error_msg


def _list_files(source_type: str, credentials: dict, prefix: str = "", limit: int = 200) -> list[dict]:
    """List files from a remote source."""
    files = []
    if source_type == "aws_s3":
        import boto3
        client = boto3.client(
            "s3",
            aws_access_key_id=credentials["access_key_id"],
            aws_secret_access_key=credentials["secret_access_key"],
            region_name=credentials.get("region", "us-east-1"),
        )
        bucket = credentials["bucket"]
        search_prefix = credentials.get("prefix", "") + prefix

        paginator = client.get_paginator("list_objects_v2")
        count = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix, MaxKeys=limit):
            for obj in page.get("Contents", []):
                name = obj["Key"]
                if name.endswith("/"):
                    continue
                files.append({
                    "key": name,
                    "name": name.split("/")[-1],
                    "size": obj["Size"],
                    "size_mb": round(obj["Size"] / (1024 * 1024), 1),
                    "last_modified": obj.get("LastModified", "").isoformat() if obj.get("LastModified") else "",
                })
                count += 1
                if count >= limit:
                    return files
    return files
