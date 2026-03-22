from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import RedirectView


class DashboardView(LoginRequiredMixin, RedirectView):
    pattern_name = "analysis:analytics"
