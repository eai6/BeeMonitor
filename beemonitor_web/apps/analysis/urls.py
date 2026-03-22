from django.urls import path
from django.views.generic import RedirectView

from . import views
from .video_proxy import VideoProxyView

app_name = "analysis"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="analysis:analytics", permanent=False), name="list"),
    path("analytics/", views.AnalyticsDashboardView.as_view(), name="analytics"),
    path("analytics/download-events/", views.DownloadEventsCSVView.as_view(), name="download_events"),
    path("analytics/download-tracking/", views.DownloadTrackingCSVView.as_view(), name="download_tracking"),
    path("analytics/download-trips/", views.DownloadTripsCSVView.as_view(), name="download_trips"),
    path("new/", views.JobCreateView.as_view(), name="new"),
    path("batch/", views.BatchJobView.as_view(), name="batch"),
    path("poll/", views.PollJobsView.as_view(), name="poll"),
    path("<int:pk>/", views.JobDetailView.as_view(), name="detail"),
    path("<int:pk>/results/", views.JobResultsView.as_view(), name="results"),
    path("<int:pk>/video/", VideoProxyView.as_view(), name="video_proxy"),
]
