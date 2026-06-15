from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import RedirectView


class DashboardView(LoginRequiredMixin, RedirectView):
    # Landing page is the device list — the operational home of the platform.
    # Video processing / analysis lives under the "Processing" menu.
    pattern_name = "devices:list"
