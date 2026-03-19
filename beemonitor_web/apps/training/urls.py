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
]
