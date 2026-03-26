from django.contrib.auth import get_user_model
from rest_framework import serializers

from apps.accounts.models import APIKey, UserProfile
from apps.analysis.models import Job, JobResult
from apps.developer.models import UsageLog, WebhookEndpoint
from apps.sources.models import DataSource
from apps.videos.models import Video

User = get_user_model()


# ── Accounts ──────────────────────────────────────────────────────────────────


class UserProfileSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    email = serializers.EmailField(source="user.email", read_only=True)

    class Meta:
        model = UserProfile
        fields = [
            "username",
            "email",
            "organization",
            "tier",
            "monthly_job_count",
            "storage_used_bytes",
        ]
        read_only_fields = fields


class APIKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = APIKey
        fields = [
            "id",
            "name",
            "prefix",
            "key_type",
            "permissions",
            "rate_limit",
            "is_active",
            "last_used_at",
            "created_at",
        ]
        read_only_fields = fields


class APIKeyCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=100)
    key_type = serializers.ChoiceField(choices=APIKey.KeyType.choices)

    # Returned in response only
    raw_key = serializers.CharField(read_only=True)

    def create(self, validated_data):
        user = self.context["request"].user
        instance, raw_key = APIKey.create_key(
            user=user,
            name=validated_data["name"],
            key_type=validated_data["key_type"],
        )
        # Attach the raw key so the serializer can surface it in the response.
        instance._raw_key = raw_key
        return instance

    def to_representation(self, instance):
        data = APIKeySerializer(instance).data
        # Include the raw key exactly once at creation time.
        data["raw_key"] = getattr(instance, "_raw_key", None)
        return data


# ── Videos ────────────────────────────────────────────────────────────────────


class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = [
            "id",
            "title",
            "source",
            "azure_blob_path",
            "file_size_bytes",
            "duration_seconds",
            "resolution",
            "fps",
            "uploaded_at",
            "status",
            "metadata",
        ]
        read_only_fields = [
            "id",
            "azure_blob_path",
            "file_size_bytes",
            "duration_seconds",
            "resolution",
            "fps",
            "uploaded_at",
            "status",
            "metadata",
        ]


class VideoUploadSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=300)
    source = serializers.PrimaryKeyRelatedField(
        queryset=DataSource.objects.all(),
        required=False,
        allow_null=True,
    )

    def validate_source(self, value):
        """Ensure the source belongs to the requesting user."""
        if value and value.user != self.context["request"].user:
            raise serializers.ValidationError(
                "You do not own this data source."
            )
        return value


class VideoFileUploadSerializer(serializers.Serializer):
    """Accepts multipart file upload for desktop/programmatic clients."""

    video_file = serializers.FileField()
    title = serializers.CharField(max_length=300, required=False, allow_blank=True)
    site_name = serializers.CharField(max_length=200, required=False, allow_blank=True)


# ── Analysis ──────────────────────────────────────────────────────────────────


class JobResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobResult
        fields = [
            "id",
            "events_csv_path",
            "tracking_csv_path",
            "foraging_trips_csv_path",
            "interactions_csv_path",
            "annotated_video_path",
            "total_events",
            "entry_count",
            "exit_count",
            "unique_tracks",
            "nest_count",
            "foraging_trip_count",
            "avg_trip_duration_sec",
            "interaction_count",
            "summary_stats",
        ]
        read_only_fields = fields


class JobSerializer(serializers.ModelSerializer):
    result = serializers.SerializerMethodField()

    class Meta:
        model = Job
        fields = [
            "id",
            "video",
            "modal_job_id",
            "status",
            "config",
            "progress_pct",
            "started_at",
            "completed_at",
            "error_message",
            "compute_cost_usd",
            "created_at",
            "result",
        ]
        read_only_fields = fields

    def get_result(self, obj):
        if obj.status == Job.Status.COMPLETED:
            try:
                return JobResultSerializer(obj.result).data
            except JobResult.DoesNotExist:
                return None
        return None


class JobCreateSerializer(serializers.Serializer):
    video_id = serializers.PrimaryKeyRelatedField(
        queryset=Video.objects.all(),
        source="video",
    )
    config = serializers.JSONField(required=False, default=dict)

    def validate_video(self, value):
        if value.user != self.context["request"].user:
            raise serializers.ValidationError("You do not own this video.")
        return value

    def create(self, validated_data):
        user = self.context["request"].user
        return Job.objects.create(user=user, **validated_data)


class BatchSubmitSerializer(serializers.Serializer):
    """Submit multiple videos for batch GPU analysis (chunked like the web app)."""

    video_ids = serializers.ListField(
        child=serializers.IntegerField(), min_length=1
    )
    config = serializers.JSONField(required=False, default=dict)


# ── Auth ─────────────────────────────────────────────────────────────────────


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)


# ── Sources ───────────────────────────────────────────────────────────────────


class DataSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSource
        fields = [
            "id",
            "name",
            "source_type",
            "is_connected",
            "last_synced_at",
            "created_at",
        ]
        read_only_fields = fields


class DataSourceCreateSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=200)
    source_type = serializers.ChoiceField(choices=DataSource.SourceType.choices)
    credentials = serializers.DictField(child=serializers.CharField())

    def create(self, validated_data):
        from django.core.serializers.json import DjangoJSONEncoder
        import json

        user = self.context["request"].user
        # Store credentials as JSON-encoded bytes.  In production this should
        # be encrypted with a KMS key before persisting.
        raw_bytes = json.dumps(
            validated_data["credentials"], cls=DjangoJSONEncoder
        ).encode("utf-8")

        return DataSource.objects.create(
            user=user,
            name=validated_data["name"],
            source_type=validated_data["source_type"],
            config_encrypted=raw_bytes,
        )


# ── Developer ─────────────────────────────────────────────────────────────────


class WebhookSerializer(serializers.ModelSerializer):
    class Meta:
        model = WebhookEndpoint
        fields = [
            "id",
            "url",
            "events",
            "secret",
            "is_active",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]
        extra_kwargs = {"secret": {"write_only": True}}


class UsageLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = UsageLog
        fields = [
            "id",
            "api_key",
            "endpoint",
            "method",
            "status_code",
            "response_time_ms",
            "created_at",
        ]
        read_only_fields = fields
