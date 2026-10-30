from django.urls import path
from django.views.generic import RedirectView

from . import period_views, views
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
    path("<int:pk>/events/review", views.EventReviewView.as_view(), name="event_review"),
    # Period review across clips, and its trips (memory/62).
    path("review/", period_views.PeriodReviewView.as_view(), name="period_review"),
    path("review/trips", period_views.TripReviewView.as_view(), name="trip_review"),
    path("review/trips.csv", period_views.PeriodTripsCsvView.as_view(), name="period_trips_csv"),
    path("<int:pk>/stream/", period_views.PeriodStreamView.as_view(), name="period_stream"),
    path("<int:pk>/results/<str:kind>.csv", views.JobPrimitiveCsvView.as_view(),
         name="results_csv"),
    path("<int:pk>/annotate/", views.GenerateAnnotatedVideoView.as_view(), name="annotate_video"),
    path("<int:pk>/video/", VideoProxyView.as_view(), name="video_proxy"),
]
