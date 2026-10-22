import csv
import logging
import re

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import Http404, HttpResponse, HttpResponseRedirect, JsonResponse
from django.db import models
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
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
        # Own videos + videos from devices shared with me (viewer or manager).
        qs = Video.accessible(self.request.user)

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
        user_videos = Video.accessible(self.request.user)
        # Pure viewers (no owned/manager videos) get a read-only list — hide the
        # analyze/delete bulk actions. Server-side scoping enforces this too.
        ctx["can_manage_any"] = Video.manageable(self.request.user).exists()

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


def _user_devices(user):
    """Devices the user can attribute uploads to (owned)."""
    from apps.devices.models import Device
    return Device.objects.filter(owner=user).order_by("name")


class VideoUploadView(LoginRequiredMixin, TemplateView):
    """The upload page (memory/44): any number of files of any size, uploaded
    in parts straight from the browser to S3 (apps/api/multipart.py), each
    with a recording time from the file, its name or the uploader, and an
    optional site, device and batch label. Bytes never pass through Django.
    """
    template_name = "videos/upload.html"

    def get_context_data(self, **kwargs):
        import json as _json
        from apps.pipelines.models import Pipeline
        from .models import Site
        ctx = super().get_context_data(**kwargs)
        user = self.request.user
        ctx["devices"] = _user_devices(user)
        ctx["sites_json"] = _json.dumps([_site_json(s) for s in Site.objects.filter(user=user)])
        from apps.pipelines.registry import pipeline_input_kind
        for key, qs in (("pipelines", Pipeline.objects.filter(user=user, is_template=False)),
                        ("templates", Pipeline.objects.filter(is_template=True))):
            items = list(qs.order_by("title"))
            for p in items:
                p.input_kind = pipeline_input_kind(p.steps)   # video | photo
            ctx[key] = items
        return ctx


class VideoBatchUploadView(LoginRequiredMixin, View):
    """The old batch page; the upload page now takes any number of files."""

    def get(self, request):
        return redirect("videos:upload")


class VideoThumbnailView(LoginRequiredMixin, View):
    """Redirect to the clip's sampled still.

    A redirect rather than an inlined URL because the review grid renders 200
    cards: presigning all of them server-side would be 200 S3 calls on every
    page load, for images the viewer mostly never scrolls to. The <img> is
    lazy, so only what reaches the viewport costs anything.
    """

    def get(self, request, pk):
        video = get_object_or_404(Video.accessible(request.user, photos=True), pk=pk)
        return thumbnail_redirect(video)


def thumbnail_redirect(video, expiry_hours=None):
    """A redirect to the clip's still (or a photo's preview), presigned.

    Shared by the signed-in grid and a public batch share link, which passes a
    short ``expiry_hours``.
    """
    pk = video.pk
    from config.storage import get_s3_client
    extra = (expiry_hours,) if expiry_hours else ()  # else the client's default
    if video.is_photo:
        # The device's own 1280 px preview, in raw-videos beside the photo.
        key = (video.metadata or {}).get("thumb_key") or video.storage_key
        try:
            return HttpResponseRedirect(
                get_s3_client().generate_presigned_url("raw-videos", key, *extra))
        except Exception:
            logger.exception("Failed to presign photo preview %s", pk)
            raise Http404("Could not read the photo.")
    key = video.thumbnail_key
    if not key:
        # Uploaded before stills existed. Make one now and keep it, so the
        # grid fills in as it is browsed instead of waiting on a backfill.
        from .thumbnails import extract_on_demand
        key = extract_on_demand(video)
    if not key:
        raise Http404("No still for this clip.")
    try:
        url = get_s3_client().generate_presigned_url("processed", key, *extra)
    except Exception:
        logger.exception("Failed to presign thumbnail for video %s", pk)
        raise Http404("Could not read the still.")
    return HttpResponseRedirect(url)


class VideoStreamView(LoginRequiredMixin, View):
    """Redirect to a playable URL for the raw clip.

    Same reasoning as the thumbnail: the grid asks for this only when a card is
    actually hovered or opened, so a presign is spent per clip WATCHED rather
    than per clip listed.
    """

    def get(self, request, pk):
        video = get_object_or_404(Video.accessible(request.user, photos=True), pk=pk)
        return stream_redirect(video)


def stream_redirect(video, expiry_hours=None):
    """A redirect to the raw clip, presigned (a share link passes a short expiry)."""
    blob_path = video.storage_key or ""
    if not blob_path or blob_path.startswith("s3://"):
        raise Http404("This clip has no stored file yet.")
    from config.storage import get_s3_client
    extra = (expiry_hours,) if expiry_hours else ()  # else the client's default
    try:
        url = get_s3_client().generate_presigned_url("raw-videos", blob_path, *extra)
    except Exception:
        logger.exception("Failed to presign video %s", video.pk)
        raise Http404("Could not read the clip.")
    return HttpResponseRedirect(url)


class VideoDetailView(LoginRequiredMixin, DetailView):
    template_name = "videos/detail.html"
    context_object_name = "video"

    def get_queryset(self):
        # Own videos, plus videos from a device shared with me (view-only).
        # Photos too: a periodic photo opens here, and so does each photo of a
        # motion burst (from its clip's page).
        return Video.accessible(self.request.user, photos=True)

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        video = self.object
        # Owners and device-managers can act (run/delete); viewers are read-only.
        ctx["is_owner"] = video.user_id == self.request.user.id
        ctx["can_manage"] = video.managed_by(self.request.user)
        from apps.analysis.models import Job
        ctx["jobs"] = Job.objects.for_viewer(self.request.user).filter(video=video)
        # Its ROI and reference objects, and where they come from (memory/52).
        if not video.is_photo:
            from apps.devices.layouts import layout_for_video
            ctx["layout"] = layout_for_video(video)
            ctx["can_edit_layout"] = (self.request.user.is_superuser
                                      or Video.deletable(self.request.user).filter(pk=video.pk).exists())
        blob_path = video.storage_key

        # Presigned URL for playback (external-S3-ingested videos have no URL
        # until they're copied into our raw-videos bucket).
        if blob_path and not blob_path.startswith("s3://"):
            try:
                from config.storage import get_s3_client
                s3 = get_s3_client()
                ctx["video_url"] = s3.generate_presigned_url("raw-videos", blob_path)
                if video.is_photo:
                    # Fit shows the device's 1280 px preview; 100% loads the
                    # original only when asked.
                    ctx["preview_url"] = s3.generate_presigned_url(
                        "raw-videos", (video.metadata or {}).get("thumb_key") or blob_path)
            except Exception as e:
                logger.error("Failed to presign video URL: %s", e)

        # A clip's motion-burst photos (taken right before it), and a burst
        # photo's clip and its siblings.
        if video.is_photo:
            ctx["clip"] = video.parent
            ctx["burst"] = list(video.parent.burst_photos()) if video.parent_id else []
        else:
            ctx["burst"] = list(video.burst_photos())

        # Prev/next consecutive video on the SAME device, ordered by recorded time
        # (tie-broken by pk) — so you can step through a device's footage in order
        # without going back to the list. "Prev" = earlier, "Next" = later.
        from django.db.models import Q
        # Photos step through photos; a burst photo steps through its burst
        # (via the strip), so prev/next here stay among top-level items.
        siblings = Video.accessible(self.request.user, photos=True).filter(
            device_id=video.device_id, kind=video.kind, parent__isnull=True,
        ).exclude(pk=video.pk)
        ts = video.recorded_at
        if ts:
            ctx["prev_video"] = siblings.filter(
                Q(recorded_at__lt=ts) | Q(recorded_at=ts, pk__lt=video.pk)
            ).order_by("-recorded_at", "-pk").first()
            ctx["next_video"] = siblings.filter(
                Q(recorded_at__gt=ts) | Q(recorded_at=ts, pk__gt=video.pk)
            ).order_by("recorded_at", "pk").first()
        else:
            up = video.uploaded_at
            ctx["prev_video"] = siblings.filter(uploaded_at__lt=up).order_by("-uploaded_at", "-pk").first() if up else None
            ctx["next_video"] = siblings.filter(uploaded_at__gt=up).order_by("uploaded_at", "pk").first() if up else None

        return ctx


def _delete_storage_objects_for_video(video):
    """Delete all S3 objects associated with a video and its analysis results —
    and, for a clip, those of the motion-burst photos that go with it."""
    if not video.is_photo:
        for photo in video.burst_photos():
            _delete_storage_objects_for_video(photo)
            _tombstone_device_copy(photo)
    try:
        from config.storage import get_s3_client
        s3 = get_s3_client()

        thumb = (video.metadata or {}).get("thumb_key") if video.is_photo else ""
        if thumb:
            try:
                s3.delete_blob("raw-videos", thumb)
            except Exception as e:
                logger.warning("Could not delete photo preview %s: %s", thumb, e)

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
        qs = Video.accessible(request.user)

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
        # A photo's sidecar holds its still id (the old DeviceStill id for
        # migrated photos) and is freed through the stills path.
        sid = ((video.metadata or {}).get("legacy_still_id") or video.id
               if video.is_photo else video.id)
        PendingDeviceDeletion.objects.get_or_create(
            device_id=video.device_id, video_id=sid, is_photo=video.is_photo,
        )


class VideoDeleteView(LoginRequiredMixin, View):
    """Delete a single video and its S3 objects."""

    def post(self, request, pk):
        video = get_object_or_404(Video.deletable(request.user, photos=True), pk=pk)
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

        videos = Video.deletable(request.user).filter(pk__in=video_ids)
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
        video = get_object_or_404(Video.deletable(request.user, photos=True), pk=pk)
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
            Video.deletable(request.user)
            .filter(pk__in=video_ids, device__isnull=False)
            .update(device_delete_requested=True)
        )
        if n:
            messages.success(request, f"{n} video(s) cleared for deletion on their devices (server copies kept).")
        else:
            messages.warning(request, "None of the selected videos came from a device.")
        return redirect("videos:list")


class SiteListCreateView(LoginRequiredMixin, View):
    """The user's sites (GET) and a new one (POST, JSON) for the upload page."""

    def get(self, request):
        from .models import Site
        return JsonResponse({"sites": [_site_json(s) for s in Site.objects.filter(user=request.user)]})

    def post(self, request):
        import json as _json
        from .models import Site
        try:
            data = _json.loads(request.body or b"{}")
        except ValueError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
        name = str(data.get("name") or "").strip()[:200]
        if not name:
            return JsonResponse({"error": "A site needs a name."}, status=400)

        def _coord(key, lo, hi):
            raw = data.get(key)
            if raw in (None, ""):
                return None
            value = float(str(raw).replace("\u2212", "-"))
            if not lo <= value <= hi:
                raise ValueError
            return value
        try:
            lat, lon = _coord("lat", -90, 90), _coord("lon", -180, 180)
        except (TypeError, ValueError):
            return JsonResponse({"error": "Latitude is -90 to 90 and longitude -180 to 180."}, status=400)
        site, created = Site.objects.get_or_create(
            user=request.user, name=name,
            defaults={"lat": lat, "lon": lon, "notes": str(data.get("notes") or "")[:500]})
        if not created and (lat is not None and lon is not None):
            site.lat, site.lon = lat, lon
            site.save(update_fields=["lat", "lon"])
        return JsonResponse({"site": _site_json(site)}, status=201 if created else 200)


def _site_json(site):
    return {"id": site.id, "name": site.name, "lat": site.lat, "lon": site.lon, "notes": site.notes}


# ── ROI and reference objects for sites and clips (memory/52) ────────────────

class VideoFrameView(LoginRequiredMixin, View):
    """One full-resolution frame of a clip at ``?t=`` seconds, for drawing on."""

    def get(self, request, pk):
        from django.http import HttpResponse

        from .frames import frame_jpeg

        video = get_object_or_404(Video.accessible(request.user), pk=pk)
        try:
            t = float(request.GET.get("t") or 0)
        except ValueError:
            t = 0.0
        jpeg = frame_jpeg(video, t)
        if jpeg is None:
            return HttpResponse(status=404)
        resp = HttpResponse(jpeg, content_type="image/jpeg")
        resp["Cache-Control"] = "private, max-age=3600"
        return resp


def _layout_versions(layouts):
    """The editor's history list for a site's or a clip's layout versions."""
    from django.utils.formats import date_format
    from django.utils.timezone import localtime

    out = []
    for i, v in enumerate(layouts):
        n = len(v.nest_layout or [])
        shape = "No ROI" if not v.roi_override else "Polygon ROI" if v.roi_polygon else "Rectangle ROI"
        detail = f"{shape} · {n} reference object{'s' if n != 1 else ''}"
        if v.saved_by:
            detail += f" · {v.saved_by.get_username()}"
        out.append({"number": v.number, "state": "in_use" if i == 0 else "used",
                    "span": f"Saved {date_format(localtime(v.saved_at), 'M j, Y H:i')}",
                    "detail": detail,
                    "roi": ({"box": v.roi_override, "points": v.roi_polygon} if v.roi_override else None),
                    "nests": v.nest_layout or []})
    return out


def _roi_shape(layout):
    if not layout.get("roi_override"):
        return None
    shape = {"box": layout["roi_override"]}
    if layout.get("roi_polygon"):
        shape["points"] = layout["roi_polygon"]
    return shape


class LayoutEditorView(LoginRequiredMixin, View):
    """Draw the ROI and reference objects for a clip (``video_pk``) or a site
    (``site_pk``), on a full-resolution frame of a clip. Saving from a clip
    goes to its site by default (every clip recorded there), or to the clip
    alone; a clip whose site is only a name gets a site of that name."""

    template_name = "videos/layout_editor.html"

    def _target(self, request, video_pk=None, site_pk=None, picked=None):
        from .models import Site
        if site_pk is not None:
            sites = Site.objects.all() if request.user.is_superuser else Site.objects.filter(user=request.user)
            site = get_object_or_404(sites, pk=site_pk)
            # The frame comes from a clip the user picks (?clip=), else the
            # site's newest. A new site has none: the page offers the owner's
            # recent clips, and the picked one joins the site on save.
            picked = picked or request.GET.get("clip")
            clip = None
            if picked and str(picked).isdigit():
                clip = self._site_candidates(site).filter(pk=picked).first()
            if clip is None:
                clip = (Video.objects.filter(site=site, raw_deleted_at__isnull=True)
                        .order_by("-recorded_at", "-uploaded_at").first())
            return site, None, clip
        # Editing a layout changes how a clip is analysed: its owner, a manager
        # of its unit, or a platform admin (who analyses anyone's clips; the
        # change goes in the audit log).
        clips = Video.objects.all() if request.user.is_superuser else Video.deletable(request.user)
        video = get_object_or_404(clips.select_related("site", "device", "user"), pk=video_pk)
        return video.site, video, video

    @staticmethod
    def _site_candidates(site):
        """Clips that can give a site its drawing frame: the site owner's."""
        return Video.objects.filter(user=site.user, raw_deleted_at__isnull=True)

    def get(self, request, video_pk=None, site_pk=None):
        import json as _json

        from apps.devices.layouts import layout_for_video

        from .models import Layout

        site, video, clip = self._target(request, video_pk, site_pk)
        if video is not None:
            current = layout_for_video(video)
            own = Layout.objects.filter(video=video).select_related("saved_by")
            history = _layout_versions(own) if own.exists() else _layout_versions(
                Layout.objects.filter(site=site).select_related("saved_by")) if site else []
        else:
            latest = Layout.current(site=site)
            current = ({"roi_override": latest.roi_override, "roi_polygon": latest.roi_polygon,
                        "nest_layout": latest.nest_layout} if latest else
                       {"roi_override": None, "roi_polygon": None, "nest_layout": []})
            current["source_label"] = f"site {site.name}" if latest else ""
            history = _layout_versions(Layout.objects.filter(site=site).select_related("saved_by"))
        t = request.GET.get("t") or "0"
        recent = []
        if video is None:
            recent = list(self._site_candidates(site).order_by("-uploaded_at")[:12])
        return render(request, self.template_name, {
            "recent_clips": recent,
            "clip_at_site": bool(clip and site and clip.site_id == site.pk),
            "site": site, "video": video, "clip": clip,
            "site_name": (site.name if site else (video.site_name if video else "")),
            "site_clips": Video.objects.filter(site=site).count() if site else 0,
            "image_url": (reverse("videos:frame", args=[clip.pk]) + f"?t={t}") if clip else "",
            "t": t,
            "source_label": current.get("source_label", ""),
            "roi_json": _json.dumps(_roi_shape(current)),
            "nests_json": _json.dumps(current.get("nest_layout") or []),
            "versions_json": history,
        })

    def post(self, request, video_pk=None, site_pk=None):
        import json as _json

        from apps.devices.views import parse_layout_body

        from .models import Layout, Site

        try:
            body = _json.loads(request.body or "{}")
        except ValueError:
            return JsonResponse({"error": "Invalid JSON."}, status=400)
        site, video, _clip = self._target(request, video_pk, site_pk, picked=body.get("clip"))
        roi, polygon, nests = parse_layout_body(body)
        scope = body.get("scope") or ("site" if site_pk is not None else "site" if (video and (site or video.site_name)) else "clip")
        if video is not None and scope == "site" and site is None:
            if not video.site_name:
                return JsonResponse({"error": "This clip has no site. Save it for this clip only."}, status=400)
            # A site that was only ever a typed name becomes a real one.
            site, _ = Site.objects.get_or_create(user=video.user, name=video.site_name)
            Video.everything.filter(user=video.user, site__isnull=True, site_name=video.site_name).update(site=site)
        if site_pk is not None and _clip is not None and _clip.site_id != site.pk and body.get("add_clip", True):
            # The clip it was drawn on joins the site, so the drawing applies to it.
            Video.everything.filter(pk=_clip.pk).update(site=site, site_name=site.name)
        if scope == "site":
            saved = Layout.save_new(site=site, roi=roi, polygon=polygon, nests=nests, user=request.user)
            versions = _layout_versions(Layout.objects.filter(site=site).select_related("saved_by"))
            n = Video.objects.filter(site=site).count()
            message = f"Saved for every clip at {site.name} ({n})."
        else:
            saved = Layout.save_new(video=video, roi=roi, polygon=polygon, nests=nests, user=request.user)
            versions = _layout_versions(Layout.objects.filter(video=video).select_related("saved_by"))
            message = "Saved for this clip only."
        owner = video.user if video is not None else site.user
        if owner != request.user:
            from apps.accounts.models import AdminAction
            AdminAction.record(request.user, "edit_layout",
                               f"Drew the ROI and references for {owner.username}'s "
                               + (f"site {site.name}" if scope == "site" else f"clip {video.title}"),
                               target=owner, layout=saved.pk)
        return JsonResponse({"ok": True, "roi": saved.roi_override, "roi_polygon": saved.roi_polygon,
                             "nests": saved.nest_layout, "versions": versions,
                             "message": message + " New runs use it."})


class SitesPageView(LoginRequiredMixin, View):
    """Sites: places recorded without a unit, with their layouts (memory/52)."""

    def get(self, request):
        from django.db.models import Count

        from .models import Layout, Site

        sites = Site.objects.filter(user=request.user).annotate(
            clips=Count("videos", filter=models.Q(videos__kind=Video.Kind.VIDEO)))
        rows = [{"site": s, "layout": Layout.current(site=s)} for s in sites]
        return render(request, "videos/sites.html", {"rows": rows})

    def post(self, request):
        from .models import Site

        name = (request.POST.get("name") or "").strip()[:200]
        if not name:
            messages.error(request, "A site needs a name.")
            return redirect("videos:sites_page")
        site, created = Site.objects.get_or_create(user=request.user, name=name)
        if created:
            messages.success(request, f"Added {name}. Draw its ROI and reference objects next.")
            return redirect("videos:site_layout", site_pk=site.pk)
        messages.info(request, f"You already have a site called {name}.")
        return redirect("videos:sites_page")
