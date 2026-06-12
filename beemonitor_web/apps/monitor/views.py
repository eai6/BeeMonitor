"""User-facing activity pages.

An *activity* is one motion event on a device; this app lets a user browse
activities and open one to see the sampled frames and (once the perception
pipeline runs) the BioCLIP taxonomic results. Access is scoped to devices the
user can see, reusing ``Device.accessible``.
"""

import logging

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import DetailView, ListView, TemplateView

from apps.devices.models import Device
from apps.devices.views import _display_zone  # device timezone resolver (reuse)
from config.storage import presigned_get as _presign

from .agent import client as agent_client
from .models import Activity, DailyDigest

logger = logging.getLogger(__name__)


class DigestListView(LoginRequiredMixin, ListView):
    """Recent daily digests across the user's devices (optionally one device)."""

    template_name = "monitor/digests.html"
    context_object_name = "digests"
    paginate_by = 30

    def get_queryset(self):
        qs = (DailyDigest.objects
              .filter(device__in=Device.accessible(self.request.user))
              .select_related("device"))
        device_id = self.request.GET.get("device")
        if device_id and device_id.isdigit():
            qs = qs.filter(device_id=int(device_id))
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        device_id = self.request.GET.get("device")
        if device_id and device_id.isdigit():
            ctx["filter_device"] = (Device.accessible(self.request.user)
                                    .filter(pk=int(device_id)).first())
        return ctx


class AgentChatView(LoginRequiredMixin, TemplateView):
    """Chat page for the taxonomic monitoring agent."""

    template_name = "monitor/agent_chat.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        ctx["devices"] = Device.accessible(self.request.user).order_by("name")
        ctx["agent_enabled"] = agent_client.is_enabled()
        try:
            ctx["preselect_device"] = int(self.request.GET.get("device", ""))
        except (TypeError, ValueError):
            ctx["preselect_device"] = None
        return ctx


class ActivityListView(LoginRequiredMixin, ListView):
    """Recent activities across the user's devices (optionally one device)."""

    template_name = "monitor/activity_list.html"
    context_object_name = "activities"
    paginate_by = 24

    def get_queryset(self):
        # prefetch frames so the per-row thumbnail + count come from one extra
        # query for the whole page, not two queries per row (N+1).
        qs = (Activity.objects
              .filter(device__in=Device.accessible(self.request.user))
              .select_related("device", "best_taxon")
              .prefetch_related("frames"))
        device_id = self.request.GET.get("device")
        if device_id and device_id.isdigit():
            qs = qs.filter(device_id=int(device_id))
        status = self.request.GET.get("status")
        if status in dict(Activity.Status.choices):
            qs = qs.filter(status=status)
        return qs

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        zones: dict = {}  # device id -> ZoneInfo, resolved once per device
        for a in ctx["activities"]:
            frames = list(a.frames.all())  # prefetched — no extra query
            first = frames[0] if frames else None
            a.thumb_url = _presign(first.storage_key) if first else None
            a.thumb_count = len(frames)
            if a.device_id not in zones:
                zones[a.device_id] = _display_zone(a.device)[0]
            a.local_started = a.started_at.astimezone(zones[a.device_id])
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
        # Render the event time in the device's display timezone (same resolver
        # the device pages use), not raw UTC.
        ctx["local_started"] = activity.started_at.astimezone(_display_zone(activity.device)[0])
        return ctx
