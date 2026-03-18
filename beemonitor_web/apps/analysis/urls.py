from django.urls import path

from . import views
from .video_proxy import VideoProxyView

app_name = "analysis"

urlpatterns = [
    path("", views.JobListView.as_view(), name="list"),
    path("analytics/", views.AnalyticsDashboardView.as_view(), name="analytics"),
    path("new/", views.JobCreateView.as_view(), name="new"),
    path("batch/", views.BatchJobView.as_view(), name="batch"),
    path("<int:pk>/", views.JobDetailView.as_view(), name="detail"),
    path("<int:pk>/results/", views.JobResultsView.as_view(), name="results"),
    path("<int:pk>/video/", VideoProxyView.as_view(), name="video_proxy"),
]
