from django.urls import path

from . import views

app_name = "videos"

urlpatterns = [
    path("", views.VideoListView.as_view(), name="list"),
    path("upload/", views.VideoUploadView.as_view(), name="upload"),
    path("<int:pk>/", views.VideoDetailView.as_view(), name="detail"),
]
