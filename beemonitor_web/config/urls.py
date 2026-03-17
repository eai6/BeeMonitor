"""Root URL configuration for BeeMonitor Web."""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/v1/", include("apps.api.urls")),
    path("accounts/", include("apps.accounts.urls")),
    path("videos/", include("apps.videos.urls")),
    path("analysis/", include("apps.analysis.urls")),
    path("sources/", include("apps.sources.urls")),
    path("developer/", include("apps.developer.urls")),
    path("", include("apps.dashboard.urls")),
]
