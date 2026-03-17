from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.accounts.models import APIKey
from apps.analysis.models import Job
from apps.developer.models import UsageLog, WebhookEndpoint
from apps.sources.models import DataSource
from apps.videos.models import Video

from .serializers import (
    APIKeyCreateSerializer,
    APIKeySerializer,
    DataSourceCreateSerializer,
    DataSourceSerializer,
    JobCreateSerializer,
    JobSerializer,
    UsageLogSerializer,
    VideoSerializer,
    VideoUploadSerializer,
    WebhookSerializer,
)


# ── Videos ────────────────────────────────────────────────────────────────────


class VideoViewSet(viewsets.ModelViewSet):
    """CRUD for the authenticated user's videos."""

    serializer_class = VideoSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status"]

    def get_queryset(self):
        return Video.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "create":
            return VideoUploadSerializer
        return VideoSerializer

    def perform_create(self, serializer):
        # VideoUploadSerializer.create is not a ModelSerializer; build the
        # Video manually so we can set user and placeholder blob fields.
        data = serializer.validated_data
        Video.objects.create(
            user=self.request.user,
            title=data["title"],
            source=data.get("source"),
            azure_blob_path="",  # populated after upload completes
            file_size_bytes=0,
            status=Video.Status.UPLOADING,
        )


# ── Analysis / Jobs ──────────────────────────────────────────────────────────


class JobViewSet(viewsets.ModelViewSet):
    """CRUD + submit action for analysis jobs."""

    serializer_class = JobSerializer
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["status"]

    def get_queryset(self):
        return Job.objects.filter(user=self.request.user).select_related(
            "video"
        )

    def get_serializer_class(self):
        if self.action == "create":
            return JobCreateSerializer
        return JobSerializer

    @action(detail=True, methods=["post"])
    def submit(self, request, pk=None):
        """Queue a job for processing via Celery."""
        job = self.get_object()
        if job.status != Job.Status.QUEUED:
            return Response(
                {"detail": "Only queued jobs can be submitted."},
                status=status.HTTP_409_CONFLICT,
            )
        from apps.analysis.tasks import submit_analysis_job

        submit_analysis_job.delay(job.id)
        job.status = Job.Status.INGESTING
        job.save(update_fields=["status"])
        return Response(JobSerializer(job).data)


# ── Sources ───────────────────────────────────────────────────────────────────


class DataSourceViewSet(viewsets.ModelViewSet):
    """CRUD + connection test for external data sources."""

    serializer_class = DataSourceSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return DataSource.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "create":
            return DataSourceCreateSerializer
        return DataSourceSerializer

    @action(detail=True, methods=["post"])
    def test_connection(self, request, pk=None):
        """Ping the remote source to verify credentials."""
        source = self.get_object()
        from apps.sources.connectors import get_connector

        connector = get_connector(source)
        ok, message = connector.test_connection()
        source.is_connected = ok
        source.save(update_fields=["is_connected"])
        return Response({"ok": ok, "message": message})


# ── API Keys ──────────────────────────────────────────────────────────────────


class APIKeyViewSet(
    mixins.CreateModelMixin,
    mixins.ListModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    """Create, list, and revoke API keys (no update)."""

    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return APIKey.objects.filter(user=self.request.user)

    def get_serializer_class(self):
        if self.action == "create":
            return APIKeyCreateSerializer
        return APIKeySerializer

    def perform_destroy(self, instance):
        """Soft-delete by deactivating the key."""
        instance.is_active = False
        instance.save(update_fields=["is_active"])


# ── Webhooks ──────────────────────────────────────────────────────────────────


class WebhookViewSet(viewsets.ModelViewSet):
    """CRUD for webhook endpoints."""

    serializer_class = WebhookSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return WebhookEndpoint.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)


# ── Health ────────────────────────────────────────────────────────────────────


class HealthViewSet(viewsets.ViewSet):
    """Unauthenticated health-check endpoint."""

    authentication_classes = []
    permission_classes = []

    def list(self, request):
        return Response({"status": "ok"})
