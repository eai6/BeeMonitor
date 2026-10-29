from django.urls import path
from django.views.generic import RedirectView

from . import views
from .video_proxy import VideoProxyView

app_name = "analysis"

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="analysis:processing", permanent=False), name="list"),
    path("processing/", views.ProcessingHubView.as_view(), name="processing"),
    path("download-species/", views.DownloadSpeciesCSVView.as_view(), name="download_species"),
    path("new/", views.JobCreateView.as_view(), name="new"),
    path("batch/", views.BatchJobView.as_view(), name="batch"),
    path("poll/", views.PollJobsView.as_view(), name="poll"),
    path("cancel-all/", views.JobCancelAllView.as_view(), name="cancel_all"),
    path("<int:pk>/cancel/", views.JobCancelView.as_view(), name="cancel"),
    path("<int:pk>/", views.JobDetailView.as_view(), name="detail"),
    path("<int:pk>/results/", views.JobResultsView.as_view(), name="results"),
    path("<int:pk>/tracks/<int:track_id>/crops.json", views.TrackCropsView.as_view(),
         name="track_crops"),
    path("<int:pk>/overlay.json", views.TrackOverlayView.as_view(), name="track_overlay"),
    path("<int:pk>/results/<str:kind>.csv", views.JobPrimitiveCsvView.as_view(),
         name="results_csv"),
    path("<int:pk>/annotate/", views.GenerateAnnotatedVideoView.as_view(), name="annotate_video"),
    path("<int:pk>/video/", VideoProxyView.as_view(), name="video_proxy"),
]
