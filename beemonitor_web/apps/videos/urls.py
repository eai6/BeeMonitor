from django.urls import path

from . import views

app_name = "videos"

urlpatterns = [
    path("", views.VideoListView.as_view(), name="list"),
    path("upload/", views.VideoUploadView.as_view(), name="upload"),
    path("batch-upload/", views.VideoBatchUploadView.as_view(), name="batch_upload"),
    path("<int:pk>/", views.VideoDetailView.as_view(), name="detail"),
    path("<int:pk>/delete/", views.VideoDeleteView.as_view(), name="delete"),
    path("batch-delete/", views.VideoBatchDeleteView.as_view(), name="batch_delete"),
    path("export-csv/", views.VideoExportCSVView.as_view(), name="export_csv"),
]
