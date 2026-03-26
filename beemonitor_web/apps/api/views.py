import uuid

from django.conf import settings
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from apps.accounts.models import APIKey
from apps.analysis.models import Job, JobResult
from apps.developer.models import UsageLog, WebhookEndpoint
from apps.sources.models import DataSource
from apps.videos.models import Video

from .serializers import (
    APIKeyCreateSerializer,
    APIKeySerializer,
    BatchSubmitSerializer,
    DataSourceCreateSerializer,
    DataSourceSerializer,
    JobCreateSerializer,
    JobSerializer,
    LoginSerializer,
    RegisterSerializer,
    UsageLogSerializer,
    VideoFileUploadSerializer,
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

    @action(detail=False, methods=["post"], parser_classes=[MultiPartParser])
    def upload_file(self, request):
        """Accept multipart video file, upload to Azure, create Video record.

        Used by the desktop GUI and other programmatic clients so they don't
        need Azure credentials directly.
        """
        serializer = VideoFileUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        video_file = serializer.validated_data["video_file"]
        title = serializer.validated_data.get("title") or ""
        if not title:
            name = video_file.name
            title = name.rsplit(".", 1)[0] if "." in name else name
        site_name_override = serializer.validated_data.get("site_name", "")

        upload_id = uuid.uuid4().hex[:12]
        blob_path = f"{request.user.pk}/{upload_id}/{video_file.name}"

        # Reuse existing Azure upload helper
        from apps.videos.views import _upload_to_azure

        if not _upload_to_azure(blob_path, video_file):
            return Response(
                {"detail": "Azure upload failed."},
                status=status.HTTP_502_BAD_GATEWAY,
            )

        parsed_site, recorded_at = Video.parse_timestamp_from_filename(
            video_file.name
        )
        final_site = site_name_override or parsed_site

        video = Video.objects.create(
            user=request.user,
            title=title,
            azure_blob_path=blob_path,
            file_size_bytes=video_file.size,
            status=Video.Status.READY,
            recorded_at=recorded_at,
            site_name=final_site,
        )

        return Response(VideoSerializer(video).data, status=status.HTTP_201_CREATED)


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

    @action(detail=True, methods=["get"])
    def download(self, request, pk=None):
        """Generate a short-lived SAS URL for downloading a result file.

        Query param ``file`` must be one of the result path fields:
        events_csv_path, tracking_csv_path, foraging_trips_csv_path,
        interactions_csv_path, annotated_video_path.
        """
        job = self.get_object()
        file_field = request.query_params.get("file", "")

        allowed_fields = {
            "events_csv_path",
            "tracking_csv_path",
            "foraging_trips_csv_path",
            "interactions_csv_path",
            "annotated_video_path",
        }
        if file_field not in allowed_fields:
            return Response(
                {"detail": f"Invalid file field. Must be one of: {', '.join(sorted(allowed_fields))}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            result = job.result
        except JobResult.DoesNotExist:
            return Response(
                {"detail": "No results available for this job."},
                status=status.HTTP_404_NOT_FOUND,
            )

        blob_path = getattr(result, file_field, "")
        if not blob_path:
            return Response(
                {"detail": f"No {file_field} available."},
                status=status.HTTP_404_NOT_FOUND,
            )

        # Generate SAS URL
        try:
            from datetime import datetime, timedelta, timezone

            from azure.storage.blob import (
                BlobSasPermissions,
                BlobServiceClient,
                generate_blob_sas,
            )

            conn_str = settings.AZURE_STORAGE_CONNECTION_STRING
            if not conn_str:
                return Response(
                    {"detail": "Storage not configured."},
                    status=status.HTTP_503_SERVICE_UNAVAILABLE,
                )

            service = BlobServiceClient.from_connection_string(conn_str)
            account_name = service.account_name
            account_key = ""
            for part in conn_str.split(";"):
                if part.startswith("AccountKey="):
                    account_key = part.split("=", 1)[1]
                    break

            token = generate_blob_sas(
                account_name=account_name,
                container_name="processed",
                blob_name=blob_path,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            )
            url = f"https://{account_name}.blob.core.windows.net/processed/{blob_path}?{token}"
            return Response({"url": url, "file": file_field, "blob_path": blob_path})

        except Exception as e:
            return Response(
                {"detail": f"Failed to generate download URL: {e}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

    @action(detail=False, methods=["post"])
    def batch_submit(self, request):
        """Submit multiple videos for batch GPU analysis.

        Mirrors the web app's BatchJobView: creates Job records, chunks them,
        and spawns ``process_video_chunk`` on Modal so multiple videos share
        one GPU (models loaded once). Each chunk runs on a separate container.

        Returns a list of created job summaries.
        """
        import threading

        from django.utils import timezone as tz

        from apps.analysis.models import compute_config_hash
        from apps.analysis.views import _spawn_modal_batch

        serializer = BatchSubmitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        video_ids = serializer.validated_data["video_ids"]
        config = serializer.validated_data.get("config", {})

        detection_mode = config.get("detection_mode", "yolo")
        confidence = float(config.get("confidence_threshold", 0.25))
        custom_nest_model_path = config.get("custom_nest_model_path", "")
        custom_bee_model_path = config.get("custom_bee_model_path", "")

        # Fetch videos owned by user with status=READY
        videos = Video.objects.filter(
            user=request.user,
            pk__in=video_ids,
            status=Video.Status.READY,
        ).select_related("source")

        if not videos.exists():
            return Response(
                {"detail": "No ready videos found for the given IDs."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Deduplicate: skip videos already analyzed with the same config
        jobs_data = []
        skipped = 0
        for video in videos:
            config_hash = compute_config_hash(video.pk, config)
            existing = Job.objects.filter(
                user=request.user,
                video=video,
                config_hash=config_hash,
                status__in=[
                    Job.Status.COMPLETED,
                    Job.Status.PROCESSING,
                    Job.Status.QUEUED,
                ],
            ).exists()
            if existing:
                skipped += 1
                continue

            modal_job_id = f"modal_{uuid.uuid4().hex[:12]}"
            job = Job.objects.create(
                user=request.user,
                video=video,
                config=config,
                config_hash=config_hash,
                gpu_tier="A10G",
                status=Job.Status.PROCESSING,
                started_at=tz.now(),
                modal_job_id=modal_job_id,
            )
            jobs_data.append({
                "job_pk": job.pk,
                "job_id": modal_job_id,
                "user_id": str(request.user.pk),
                "video": video,
            })

        if not jobs_data:
            return Response({
                "jobs": [],
                "submitted": 0,
                "skipped": skipped,
                "detail": "All videos already analyzed with this configuration.",
            })

        # Spawn on Modal in background thread (same as web app)
        thread = threading.Thread(
            target=_spawn_modal_batch,
            args=(jobs_data, detection_mode, confidence,
                  custom_nest_model_path, custom_bee_model_path),
            daemon=True,
        )
        thread.start()

        # Return created job summaries
        job_summaries = [
            {
                "id": jd["job_pk"],
                "modal_job_id": jd["job_id"],
                "video_id": jd["video"].pk,
                "video_title": jd["video"].title,
                "status": "processing",
            }
            for jd in jobs_data
        ]

        return Response(
            {
                "jobs": job_summaries,
                "submitted": len(jobs_data),
                "skipped": skipped,
            },
            status=status.HTTP_201_CREATED,
        )


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


# ── Auth ─────────────────────────────────────────────────────────────────────


def _build_auth_response(user):
    """Build the standard auth response dict with API key and profile info."""
    from apps.accounts.models import UserProfile

    # Reuse existing device key or create a new one
    existing = APIKey.objects.filter(
        user=user, key_type=APIKey.KeyType.DEVICE, is_active=True, name="Desktop App",
    ).first()

    if existing:
        # Rotate: deactivate old, create new
        existing.is_active = False
        existing.save(update_fields=["is_active"])

    key_obj, raw_key = APIKey.create_key(
        user=user, name="Desktop App", key_type=APIKey.KeyType.DEVICE,
    )

    profile, _ = UserProfile.objects.get_or_create(user=user)

    return {
        "api_key": raw_key,
        "username": user.username,
        "email": user.email,
        "tier": profile.tier,
        "monthly_credits": profile.monthly_credits,
        "used_credits": profile.used_credits,
    }


class LoginView(viewsets.ViewSet):
    """Username/password login — returns a device API key."""

    authentication_classes = []
    permission_classes = []

    def create(self, request):
        from django.contrib.auth import authenticate

        serializer = LoginSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        user = authenticate(
            username=serializer.validated_data["username"],
            password=serializer.validated_data["password"],
        )
        if user is None:
            return Response(
                {"detail": "Invalid username or password."},
                status=status.HTTP_401_UNAUTHORIZED,
            )
        if not user.is_active:
            return Response(
                {"detail": "Account is disabled."},
                status=status.HTTP_403_FORBIDDEN,
            )

        return Response(_build_auth_response(user))


class RegisterView(viewsets.ViewSet):
    """Create a new account — returns a device API key."""

    authentication_classes = []
    permission_classes = []

    def create(self, request):
        from django.contrib.auth.models import User

        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        username = serializer.validated_data["username"]
        email = serializer.validated_data["email"]
        password = serializer.validated_data["password"]

        if User.objects.filter(username=username).exists():
            return Response(
                {"detail": "Username already taken."},
                status=status.HTTP_409_CONFLICT,
            )
        if User.objects.filter(email=email).exists():
            return Response(
                {"detail": "Email already registered."},
                status=status.HTTP_409_CONFLICT,
            )

        user = User.objects.create_user(
            username=username, email=email, password=password,
        )

        return Response(_build_auth_response(user), status=status.HTTP_201_CREATED)


class ProfileView(viewsets.ViewSet):
    """Get the authenticated user's profile."""

    permission_classes = [IsAuthenticated]

    def list(self, request):
        from apps.accounts.models import UserProfile

        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        return Response({
            "username": request.user.username,
            "email": request.user.email,
            "tier": profile.tier,
            "monthly_credits": profile.monthly_credits,
            "used_credits": profile.used_credits,
            "remaining_credits": profile.remaining_credits,
            "total_gpu_seconds": profile.total_gpu_seconds,
        })
