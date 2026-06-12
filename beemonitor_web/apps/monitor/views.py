"""User-facing activity pages.

An *activity* is one motion event on a device; this app lets a user browse
activities and open one to see the sampled frames and (once the perception
pipeline runs) the BioCLIP taxonomic results. Access is scoped to devices the
user can see, reusing ``Device.accessible``.
"""

import logging

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView, ListView

from apps.devices.models import Device

from .models import Activity

logger = logging.getLogger(__name__)


def _presign(storage_key: str):
    """Short-lived presigned GET URL for a frame crop, or None on failure."""
    if not storage_key:
        return None
    try:
        from config.storage import get_s3_client
        return get_s3_client().generate_presigned_url(
            "raw-videos", storage_key, expiry_hours=1,
        )
    except Exception as e:  # pragma: no cover - S3 hiccup shouldn't 500 the page
        logger.warning("presign failed for %s: %s", storage_key, e)
        return None


class ActivityListView(LoginRequiredMixin, ListView):
    """Recent activities across the user's devices (optionally one device)."""

    template_name = "monitor/activity_list.html"
    context_object_name = "activities"
    paginate_by = 24

    def get_queryset(self):
        qs = (Activity.objects
              .filter(device__in=Device.accessible(self.request.user))
              .select_related("device", "best_taxon"))
        device_id = self.request.GET.get("device")
        if device_id and device_id.isdigit():
            qs = qs.filter(device_id=int(device_id))
        status = self.request.GET.get("status")
        if status in dict(Activity.Status.choices):
            qs = qs.filter(status=status)
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        for a in ctx["activities"]:
            first = a.frames.first()
            a.thumb_url = _presign(first.storage_key) if first else None
            a.thumb_count = a.frames.count()
        device_id = self.request.GET.get("device")
        if device_id and device_id.isdigit():
            ctx["filter_device"] = (Device.accessible(self.request.user)
                                    .filter(pk=int(device_id)).first())
        return ctx


class ActivityDetailView(LoginRequiredMixin, DetailView):
    """One activity: sampled frames + taxonomic analysis (when available)."""

    template_name = "monitor/activity_detail.html"
    context_object_name = "activity"

    def get_queryset(self):
        return (Activity.objects
                .filter(device__in=Device.accessible(self.request.user))
                .select_related("device", "best_taxon", "video"))

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        activity = self.object
        frames = list(activity.frames.prefetch_related("detections__taxon"))
        for fr in frames:
            fr.url = _presign(fr.storage_key)
            fr.detection_list = list(fr.detections.all())
        ctx["frames"] = frames
        ctx["observations"] = list(
            activity.observations.select_related("taxon", "representative_frame")
        )
        ctx["has_analysis"] = bool(ctx["observations"]) or any(f.detection_list for f in frames)
        return ctx
