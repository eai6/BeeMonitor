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
    # Domain-drift detection (DINOv3 on the SAM 3 endpoint) removed 2026-07-14
    # to cut g5 GPU spend; auto-adaptation stays (SAM 3 relabel + fine-tune).
    path("adapt/", views.AdaptationDashboardView.as_view(), name="adaptation"),
    path("adapt/start/", views.StartAdaptationView.as_view(), name="adapt_start"),
    path("adapt/promote/", views.PromoteAdaptationView.as_view(), name="adapt_promote"),
]
