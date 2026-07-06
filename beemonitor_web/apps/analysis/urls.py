from django.urls import path
from django.views.generic import RedirectView

from . import views
from .video_proxy import VideoProxyView

app_name = "analysis"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="analysis:processing", permanent=False), name="list"),
    path("processing/", views.ProcessingHubView.as_view(), name="processing"),
    path("analytics/download-events/", views.DownloadEventsCSVView.as_view(), name="download_events"),
    path("analytics/download-tracking/", views.DownloadTrackingCSVView.as_view(), name="download_tracking"),
    path("analytics/download-trips/", views.DownloadTripsCSVView.as_view(), name="download_trips"),
    path("analytics/download-interactions/", views.DownloadInteractionsCSVView.as_view(), name="download_interactions"),
    path("analytics/download-nest-data/", views.DownloadNestDataCSVView.as_view(), name="download_nest_data"),
    path("download-species/", views.DownloadSpeciesCSVView.as_view(), name="download_species"),
    path("new/", views.JobCreateView.as_view(), name="new"),
    path("batch/", views.BatchJobView.as_view(), name="batch"),
    path("poll/", views.PollJobsView.as_view(), name="poll"),
    path("cancel-all/", views.JobCancelAllView.as_view(), name="cancel_all"),
    path("<int:pk>/cancel/", views.JobCancelView.as_view(), name="cancel"),
    path("<int:pk>/", views.JobDetailView.as_view(), name="detail"),
    path("<int:pk>/results/", views.JobResultsView.as_view(), name="results"),
    path("<int:pk>/annotate/", views.GenerateAnnotatedVideoView.as_view(), name="annotate_video"),
    path("<int:pk>/video/", VideoProxyView.as_view(), name="video_proxy"),
]
