from django.urls import path
from django.views.generic import RedirectView

from . import views

app_name = "videos"

urlpatterns = [
    # Merged into the Processing hub (browse + filter + select + run in one place).
    path("", RedirectView.as_view(pattern_name="analysis:processing", permanent=False), name="list"),
    path("upload/", views.VideoUploadView.as_view(), name="upload"),
    path("batch-upload/", views.VideoBatchUploadView.as_view(), name="batch_upload"),
    path("<int:pk>/", views.VideoDetailView.as_view(), name="detail"),
    path("<int:pk>/delete/", views.VideoDeleteView.as_view(), name="delete"),
    path("batch-delete/", views.VideoBatchDeleteView.as_view(), name="batch_delete"),
    path("<int:pk>/device-delete/", views.VideoDeviceDeleteView.as_view(), name="device_delete"),
    path("batch-device-delete/", views.VideoBatchDeviceDeleteView.as_view(), name="batch_device_delete"),
    path("export-csv/", views.VideoExportCSVView.as_view(), name="export_csv"),
]
