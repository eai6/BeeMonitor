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
    raw = bytes(source.config_encrypted)
    if not raw:
        raise ValueError("No credentials stored for this source.")
    return _json.loads(fernet.decrypt(raw).decode())


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
    """Browse files in a connected source with pagination."""
    template_name = "sources/browse.html"
    context_object_name = "source"

    def get_queryset(self):
        return DataSource.objects.filter(user=self.request.user)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        source = self.object
        prefix = self.request.GET.get("prefix", "")
        page = int(self.request.GET.get("page", 1))
        per_page = 100

        ctx["prefix"] = prefix
        ctx["page"] = page
        ctx["per_page"] = per_page
        ctx["files"] = []
        ctx["total_files"] = 0
        ctx["has_next"] = False
        ctx["has_prev"] = page > 1
        ctx["error"] = ""

        try:
            credentials = _decrypt_credentials(source)
            all_files = _list_files(source.source_type, credentials, prefix=prefix, limit=10000)
            credentials.clear()

            ctx["total_files"] = len(all_files)
            start = (page - 1) * per_page
            end = start + per_page
            ctx["files"] = all_files[start:end]
            ctx["has_next"] = end < len(all_files)
            ctx["total_pages"] = (len(all_files) + per_page - 1) // per_page

        except Exception as e:
            logger.error("Browse error for source %s: %s", source.pk, e)
            ctx["error"] = str(e)

        return ctx


class SourceSyncView(LoginRequiredMixin, View):
    """Import selected files from a source into BeeMonitor as Video records."""

    def post(self, request, pk):
        source = get_object_or_404(DataSource, pk=pk, user=request.user)
        selected_files = request.POST.getlist("selected_files")
        import_all = request.POST.get("import_all") == "true"
        prefix = request.POST.get("prefix", "")

        try:
            credentials = _decrypt_credentials(source)
        except Exception as e:
            messages.error(request, f"Could not decrypt credentials: {e}")
            return redirect("sources:browse", pk=pk)

        try:
            # If "import all", list all files from source
            if import_all:
                all_files = _list_files(source.source_type, credentials, prefix=prefix, limit=10000)
                selected_files = [f["key"] for f in all_files]

            if not selected_files:
                messages.warning(request, "No files selected.")
                credentials.clear()
                return redirect("sources:browse", pk=pk)

            # Build a set of already-imported keys for fast duplicate check
            existing_remote_keys = set()
            for v in Video.objects.filter(user=request.user).exclude(metadata={}).only("metadata"):
                meta = v.metadata
                if isinstance(meta, dict) and "remote_key" in meta:
                    existing_remote_keys.add(meta["remote_key"])

            imported = 0
            skipped = 0
            errors = 0
            last_error = ""

            for file_key in selected_files:
                try:
                    filename = file_key.split("/")[-1] if "/" in file_key else file_key

                    # Skip non-video files
                    if not any(filename.lower().endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
                        continue

                    # Check if already imported
                    if file_key in existing_remote_keys:
                        skipped += 1
                        continue

                    # Parse timestamp from filename
                    site_name, recorded_at = Video.parse_timestamp_from_filename(filename)
                    title = filename.rsplit(".", 1)[0] if "." in filename else filename

                    Video.objects.create(
                        user=request.user,
                        source=source,
                        title=title,
                        azure_blob_path=f"s3://{file_key}",
                        file_size_bytes=0,
                        status=Video.Status.READY,
                        recorded_at=recorded_at,
                        site_name=site_name,
                        metadata={
                            "source_id": source.pk,
                            "source_type": source.source_type,
                            "remote_key": file_key,
                            "remote_bucket": credentials.get("bucket", ""),
                        },
                    )
                    imported += 1
                except Exception as e:
                    logger.error("Failed to import %s: %s", file_key, e, exc_info=True)
                    last_error = f"{file_key}: {e}"
                    errors += 1

            credentials.clear()

            source.last_synced_at = timezone.now()
            source.save(update_fields=["last_synced_at"])

            logger.info(
                "AUDIT: User %s synced %d files from source '%s' (id=%s), skipped %d, errors %d",
                request.user.username, imported, source.name, pk, skipped, errors,
            )

            msg = f"Imported {imported} video(s) from '{source.name}'."
            if skipped:
                msg += f" Skipped {skipped} already imported."
            if errors:
                msg += f" {errors} error(s). Last: {last_error}"
            messages.success(request, msg) if imported > 0 else messages.warning(request, msg)

        except Exception as e:
            credentials.clear()
            logger.exception("Sync failed for source %s: %s", pk, e)
            messages.error(request, f"Import failed: {e}")

        return redirect("videos:list")


# ── Helpers ──────────────────────────────────────────────────────────

def _test_connection(source_type: str, credentials: dict) -> tuple[bool, str]:
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


def _list_files(source_type: str, credentials: dict, prefix: str = "", limit: int = 10000) -> list[dict]:
    """List files from a remote source. Handles pagination for large buckets."""
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
        for page in paginator.paginate(Bucket=bucket, Prefix=search_prefix):
            for obj in page.get("Contents", []):
                name = obj["Key"]
                if name.endswith("/"):
                    continue
                # Only include video files
                if not any(name.lower().endswith(ext) for ext in (".mp4", ".avi", ".mov", ".mkv")):
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
