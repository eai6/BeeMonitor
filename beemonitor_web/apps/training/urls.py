from django.urls import path

from . import views

app_name = "training"

urlpatterns = [
    path("", views.TrainingListView.as_view(), name="list"),
    path("new/", views.TrainingCreateView.as_view(), name="create"),
    path("<int:pk>/", views.TrainingDetailView.as_view(), name="detail"),
    path("poll/", views.PollTrainingJobsView.as_view(), name="poll"),
    path("models/", views.CustomModelListView.as_view(), name="models"),
    path("models/upload/", views.UploadModelView.as_view(), name="upload_model"),
    path("models/<int:pk>/", views.CustomModelDetailView.as_view(), name="model_detail"),
    # Domain-drift detection (P2c)
    path("drift/", views.DriftDashboardView.as_view(), name="drift"),
    path("drift/baseline/", views.SetDriftBaselineView.as_view(), name="drift_baseline"),
    path("drift/auto-adapt/", views.ToggleAutoAdaptView.as_view(), name="drift_auto_adapt"),
    path("drift/check/", views.CheckDriftView.as_view(), name="drift_check"),
    # Closed auto-fine-tuning loop (P3)
    path("adapt/", views.AdaptationDashboardView.as_view(), name="adaptation"),
    path("adapt/start/", views.StartAdaptationView.as_view(), name="adapt_start"),
    path("adapt/promote/", views.PromoteAdaptationView.as_view(), name="adapt_promote"),
]
