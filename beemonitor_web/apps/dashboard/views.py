from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Sum
from django.views.generic import TemplateView

from apps.analysis.models import Job, JobResult
from apps.sources.models import DataSource
from apps.videos.models import Video


class DashboardView(LoginRequiredMixin, TemplateView):
    template_name = "dashboard/index.html"

    def get_context_data(self, **kwargs):
        ctx = super().get_context_data(**kwargs)
        user = self.request.user
        ctx["total_videos"] = Video.objects.filter(user=user).count()
        ctx["total_events"] = (
            JobResult.objects.filter(job__user=user).aggregate(
                total=Sum("total_events")
            )["total"]
            or 0
        )
        ctx["running_jobs"] = Job.objects.filter(
            user=user, status__in=["queued", "ingesting", "processing", "post_processing"]
        ).count()
        ctx["connected_sources"] = DataSource.objects.filter(user=user, is_connected=True).count()
        ctx["recent_jobs"] = Job.objects.filter(user=user).select_related("video")[:10]
        return ctx
