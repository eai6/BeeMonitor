from django.contrib.auth import get_user_model
from rest_framework import serializers
from rest_framework.exceptions import APIException

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
    def validate_source(self, value):
        # A video's source decides whose cloud credentials fetch it; only the
        # owner's own sources may be attached.
        if value is not None and value.user_id != self.context["request"].user.id:
            raise serializers.ValidationError("No such source.")
        return value

    class Meta:
        model = Video
        fields = [
            "id",
            "title",
            "source",
            "storage_key",
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
            "storage_key",
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


MODEL_KEYS = (("custom_nest_model", "custom_nest_model_path"),
              ("custom_bee_model", "custom_bee_model_path"))


def resolve_custom_models(user, config):
    """The job config with custom models checked against what ``user`` may run.

    The worker downloads ``custom_*_model_path`` from the models bucket and
    loads it, so the path must never be taken from the caller as-is: that ran
    any tenant's private model. A model is given by id (``custom_nest_model``)
    or by its storage path, and either must be one of ``CustomModel.usable``.
    """
    from apps.training.models import CustomModel

    config = dict(config or {})
    usable = CustomModel.usable(user)
    for id_key, path_key in MODEL_KEYS:
        model_id, path = config.pop(id_key, None), config.pop(path_key, None)
        if not model_id and not path:
            continue
        found = (usable.filter(pk=model_id) if model_id else usable.filter(storage_key=path)).first()
        if found is None:
            raise serializers.ValidationError({id_key: "No such model, or you may not use it."})
        config[path_key] = found.storage_key
    return config


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

    def validate_config(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("config must be an object.")
        return resolve_custom_models(self.context["request"].user, value)

    def create(self, validated_data):
        from apps.accounts import quota

        user = self.context["request"].user
        cfg = validated_data.get("config") or {}
        try:
            quota.check(user, [quota.video_work(validated_data["video"], cfg.get("detector_kind", "yolo"))])
        except quota.QuotaExceeded as e:
            raise OutOfCredits(e.message)
        return Job.objects.create(user=user, **validated_data)


class OutOfCredits(APIException):
    """402: the launch doesn't fit the user's GPU credits (apps/accounts/quota.py)."""
    status_code = 402
    default_code = "out_of_credits"


class BatchSubmitSerializer(serializers.Serializer):
    """Submit multiple videos for batch GPU analysis (chunked like the web app)."""

    video_ids = serializers.ListField(
        child=serializers.IntegerField(), min_length=1
    )
    config = serializers.JSONField(required=False, default=dict)

    def validate_config(self, value):
        if not isinstance(value, dict):
            raise serializers.ValidationError("config must be an object.")
        return resolve_custom_models(self.context["request"].user, value)


# ── Auth ─────────────────────────────────────────────────────────────────────


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True, min_length=8)

    def validate(self, data):
        # The same password rules as the web sign-up (AUTH_PASSWORD_VALIDATORS).
        from django.contrib.auth.models import User
        from django.contrib.auth.password_validation import validate_password
        from django.core.exceptions import ValidationError as DjangoValidationError

        try:
            validate_password(data["password"], User(username=data["username"], email=data["email"]))
        except DjangoValidationError as e:
            raise serializers.ValidationError({"password": list(e.messages)})
        return data


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
    # S3 only for now (memory/54): the Google connectors don't work.
    source_type = serializers.ChoiceField(choices=[DataSource.SourceType.AWS_S3])
    credentials = serializers.DictField(child=serializers.CharField(allow_blank=True), write_only=True)

    def validate_credentials(self, value):
        # Without its own keys a connector would run on BeeMonitor's AWS role.
        missing = [k for k in ("bucket", "access_key_id", "secret_access_key") if not value.get(k)]
        if missing:
            raise serializers.ValidationError(f"Required: {', '.join(missing)}.")
        allowed = {"bucket", "prefix", "access_key_id", "secret_access_key", "region"}
        return {k: str(v).strip() for k, v in value.items() if k in allowed}

    def create(self, validated_data):
        from apps.sources.crypto import encrypt

        return DataSource.objects.create(
            user=self.context["request"].user,
            name=validated_data["name"],
            source_type=validated_data["source_type"],
            config_encrypted=encrypt(validated_data["credentials"]),
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
