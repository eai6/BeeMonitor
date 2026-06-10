import csv
import logging
import re

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect
from django.views import View
from django.views.generic import DetailView, ListView, TemplateView
from .models import Video

logger = logging.getLogger(__name__)

_NATALIES_RE = re.compile(r"natalies?", re.IGNORECASE)
_SITEA_RE = re.compile(r"SiteA", re.IGNORECASE)


def _sanitize_site(value: str) -> str:
    """Replace occurrences of 'natalies' with 'SiteA' in display strings."""
    if not value:
        return value
    return _NATALIES_RE.sub("SiteA", value)


def _unsanitize_site(value: str) -> str:
    """Reverse-map 'SiteA' back to 'natalies' for DB queries."""
    if not value:
        return value
    return _SITEA_RE.sub("natalies", value)


class VideoListView(LoginRequiredMixin, ListView):
    template_name = "videos/list.html"
    context_object_name = "videos"
    paginate_by = 20

    def get_queryset(self):
        qs = Video.objects.filter(user=self.request.user)

        # Apply filters from query params
        search = self.request.GET.get("q", "").strip()
        site = self.request.GET.get("site")
        year = self.request.GET.get("year")
        month = self.request.GET.get("month")
        day = self.request.GET.get("day")
        hour = self.request.GET.get("hour")
        device = self.request.GET.get("device")

        if search:
            qs = qs.filter(title__icontains=search)
        if device:
            qs = qs.filter(device_id=device)
        if site:
            qs = qs.filter(site_name=_unsanitize_site(site))
        if year:
            try:
                qs = qs.filter(year=int(year))
            except (ValueError, TypeError):
                pass
        if month:
            try:
                qs = qs.filter(month=int(month))
            except (ValueError, TypeError):
                pass
        if day:
            try:
                qs = qs.filter(day=int(day))
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                qs = qs.filter(hour=int(hour))
            except (ValueError, TypeError):
                pass

        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user_videos = Video.objects.filter(user=self.request.user)

        # Build filter options from existing data (use set() to guarantee uniqueness)
        ctx["site_names"] = sorted(set(
            _sanitize_site(s) for s in
            user_videos.exclude(site_name="").values_list("site_name", flat=True)
        ))
        ctx["years"] = sorted(set(
            user_videos.exclude(year=None).values_list("year", flat=True)
        ))
        ctx["months"] = sorted(set(
            user_videos.exclude(month=None).values_list("month", flat=True)
        ))
        ctx["days"] = sorted(set(
            user_videos.exclude(day=None).values_list("day", flat=True)
        ))
        ctx["hours"] = sorted(set(
            user_videos.exclude(hour=None).values_list("hour", flat=True)
        ))
        ctx["devices"] = [
            {"id": r["device_id"], "name": r["device__name"]}
            for r in user_videos.exclude(device=None)
            .values("device_id", "device__name").distinct().order_by("device__name")
        ]

        # Preserve current filter selections
        ctx["current_search"] = self.request.GET.get("q", "")
        ctx["current_device"] = self.request.GET.get("device", "")
        ctx["current_site"] = self.request.GET.get("site", "")
        ctx["current_year"] = self.request.GET.get("year", "")
        ctx["current_month"] = self.request.GET.get("month", "")
        ctx["current_day"] = self.request.GET.get("day", "")
        ctx["current_hour"] = self.request.GET.get("hour", "")

        # All filtered video IDs (for "Select All Filtered" across pages)
        filtered_qs = self.get_queryset()
        ctx["all_video_ids"] = list(filtered_qs.values_list("pk", flat=True))
        ctx["video_count"] = len(ctx["all_video_ids"])

        # Custom models for config panel (split by type for nest/bee dropdowns)
        try:
            from apps.training.models import CustomModel
            all_models = CustomModel.objects.filter(user=self.request.user, is_active=True)
            ctx["custom_models"] = all_models  # all models for generic dropdown
            ctx["custom_nest_models"] = all_models.filter(model_type__in=["nest_detection", "custom"])
            ctx["custom_bee_models"] = all_models.filter(model_type__in=["bee_tracking", "custom"])
        except Exception:
            ctx["custom_models"] = []
            ctx["custom_nest_models"] = []
            ctx["custom_bee_models"] = []

        return ctx


def _upload_to_storage(blob_path, video_file):
    """Upload an uploaded-file stream to the raw-videos S3 bucket. True on success."""
    try:
        from config.storage import get_s3_client
        get_s3_client().upload_stream(
            "raw-videos", blob_path, video_file, content_type="video/mp4",
        )
        logger.info("Uploaded %s to S3 raw-videos", blob_path)
        return True
    except Exception as e:
        logger.error("S3 upload failed for %s: %s", blob_path, e)
        return False


class VideoUploadView(LoginRequiredMixin, TemplateView):
    """Just renders the upload page.

    The page's JS calls /api/v1/web-uploads/initiate, PUTs directly to S3,
    then calls /api/v1/web-uploads/complete — bytes never pass through
    this Django process. See apps/api/web_uploads.py.
    """
    template_name = "videos/upload.html"


class VideoBatchUploadView(LoginRequiredMixin, TemplateView):
    """Just renders the batch upload page.

    Same direct-to-S3 flow as VideoUploadView; the page loops over each
    selected file client-side and reports per-file progress.
    """
    template_name = "videos/batch_upload.html"


class VideoDetailView(LoginRequiredMixin, DetailView):
    template_name = "videos/detail.html"
    context_object_name = "video"

    def get_queryset(self):
        # Own videos, plus videos from a device shared with me (view-only).
        u = self.request.user
        return Video.objects.filter(
            Q(user=u) | Q(device__shares__user=u)
        ).distinct()

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        video = self.object
        # Shared (non-owner) viewers see the video read-only — hide owner actions.
        ctx["is_owner"] = video.user_id == self.request.user.id
        blob_path = video.storage_key

        # Presigned URL for playback (external-S3-ingested videos have no URL
        # until they're copied into our raw-videos bucket).
        if blob_path and not blob_path.startswith("s3://"):
            try:
                from config.storage import get_s3_client
                ctx["video_url"] = get_s3_client().generate_presigned_url(
                    "raw-videos", blob_path,
                )
            except Exception as e:
                logger.error("Failed to presign video URL: %s", e)

        return ctx


def _delete_storage_objects_for_video(video):
    """Delete all S3 objects associated with a video and its analysis results."""
    try:
        from config.storage import get_s3_client
        s3 = get_s3_client()

        blob_path = video.storage_key
        if blob_path and not blob_path.startswith("s3://"):
            try:
                s3.delete_blob("raw-videos", blob_path)
                logger.info("Deleted raw video: %s", blob_path)
            except Exception as e:
                logger.warning("Could not delete raw video %s: %s", blob_path, e)

        from apps.analysis.models import JobResult
        for result in JobResult.objects.filter(job__video=video):
            for path in [
                result.events_csv_path,
                result.tracking_csv_path,
                result.foraging_trips_csv_path,
                result.annotated_video_path,
            ]:
                if path:
                    try:
                        s3.delete_blob("processed", path)
                    except Exception:
                        pass
            logger.info("Deleted processed objects for job %s", result.job_id)

    except Exception as e:
        logger.error("S3 cleanup failed for video %s: %s", video.pk, e)


class VideoExportCSVView(LoginRequiredMixin, View):
    """Export filtered video filenames as CSV."""

    def get(self, request):
        qs = Video.objects.filter(user=request.user)

        search = request.GET.get("q", "").strip()
        site = request.GET.get("site")
        year = request.GET.get("year")
        month = request.GET.get("month")
        day = request.GET.get("day")
        hour = request.GET.get("hour")

        if search:
            qs = qs.filter(title__icontains=search)
        if site:
            qs = qs.filter(site_name=_unsanitize_site(site))
        if year:
            try:
                qs = qs.filter(year=int(year))
            except (ValueError, TypeError):
                pass
        if month:
            try:
                qs = qs.filter(month=int(month))
            except (ValueError, TypeError):
                pass
        if day:
            try:
                qs = qs.filter(day=int(day))
            except (ValueError, TypeError):
                pass
        if hour:
            try:
                qs = qs.filter(hour=int(hour))
            except (ValueError, TypeError):
                pass

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = 'attachment; filename="video_filenames.csv"'

        writer = csv.writer(response)
        writer.writerow(["title", "site", "recorded_at", "status", "size_bytes"])
        for v in qs.order_by("-recorded_at"):
            writer.writerow([
                _sanitize_site(v.title),
                _sanitize_site(v.site_name or ""),
                v.recorded_at.isoformat() if v.recorded_at else "",
                v.get_status_display(),
                v.file_size_bytes,
            ])

        return response


def _tombstone_device_copy(video):
    """Leave a tombstone so the device frees the clip's SD copy after the cloud
    Video row is deleted (otherwise an un-freed on-device copy is orphaned)."""
    if video.device_id and video.device_deleted_at is None:
        from .models import PendingDeviceDeletion
        PendingDeviceDeletion.objects.get_or_create(
            device_id=video.device_id, video_id=video.id,
        )


class VideoDeleteView(LoginRequiredMixin, View):
    """Delete a single video and its S3 objects."""

    def post(self, request, pk):
        video = get_object_or_404(Video, pk=pk, user=request.user)
        title = video.title

        _delete_storage_objects_for_video(video)
        _tombstone_device_copy(video)  # keep the device cleanup intent alive
        video.delete()  # CASCADE handles Jobs, JobResults, Annotations

        messages.success(request, f"Deleted video: {title}")
        return redirect("videos:list")


class VideoBatchDeleteView(LoginRequiredMixin, View):
    """Delete multiple videos and their S3 objects."""

    def post(self, request):
        video_ids = request.POST.getlist("video_ids")
        if not video_ids:
            messages.warning(request, "No videos selected.")
            return redirect("videos:list")

        videos = Video.objects.filter(user=request.user, pk__in=video_ids)
        count = videos.count()

        for video in videos:
            _delete_storage_objects_for_video(video)
            _tombstone_device_copy(video)

        videos.delete()

        messages.success(request, f"Deleted {count} video(s) and their analysis data.")
        return redirect("videos:list")


class VideoDeviceDeleteView(LoginRequiredMixin, View):
    """Clear a single uploaded video for deletion on its SOURCE DEVICE (free SD
    space). This is the human-confirm gate only — it does NOT delete the server
    copy (use VideoDeleteView for that). The device frees the local file on its
    next cleanup pass and confirms back. POST ``cancel=1`` to undo before then.
    """

    def post(self, request, pk):
        video = get_object_or_404(Video, pk=pk, user=request.user)
        if request.POST.get("cancel"):
            if video.device_deleted_at is None and video.device_delete_requested:
                video.device_delete_requested = False
                video.save(update_fields=["device_delete_requested"])
                messages.info(request, "Cancelled — the device will keep this clip.")
            return redirect("videos:detail", pk=pk)
        if video.device_id is None:
            messages.warning(request, "This video has no source device to free.")
            return redirect("videos:detail", pk=pk)
        if not video.device_delete_requested:
            video.device_delete_requested = True
            video.save(update_fields=["device_delete_requested"])
        messages.success(
            request,
            f"'{video.title}' cleared for deletion on its device — the server copy is "
            "kept; the device frees the local file on its next check-in.",
        )
        return redirect("videos:detail", pk=pk)


class VideoBatchDeviceDeleteView(LoginRequiredMixin, View):
    """Clear multiple uploaded videos for deletion on their source devices.

    Only affects videos that came from a device; the server copies are untouched.
    """

    def post(self, request):
        video_ids = request.POST.getlist("video_ids")
        if not video_ids:
            messages.warning(request, "No videos selected.")
            return redirect("videos:list")
        n = (
            Video.objects.filter(user=request.user, pk__in=video_ids, device__isnull=False)
            .update(device_delete_requested=True)
        )
        if n:
            messages.success(request, f"{n} video(s) cleared for deletion on their devices (server copies kept).")
        else:
            messages.warning(request, "None of the selected videos came from a device.")
        return redirect("videos:list")
