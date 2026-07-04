from django.urls import path

from . import views

app_name = "annotations"

urlpatterns = [
    path("", views.ProjectListView.as_view(), name="list"),
    path("new/", views.ProjectCreateView.as_view(), name="create"),
    path("<int:pk>/", views.ProjectDetailView.as_view(), name="detail"),
    path("<int:pk>/add-videos/", views.AddVideosView.as_view(), name="add_videos"),
    path("<int:pk>/remove-video/", views.RemoveVideoView.as_view(), name="remove_video"),
    path("<int:pk>/edit/", views.AnnotationEditorView.as_view(), name="editor"),
    path("<int:pk>/transfer/", views.TransferVideoView.as_view(), name="transfer_video"),
    path("<int:pk>/save/", views.SaveAnnotationView.as_view(), name="save"),
    path("<int:pk>/pre-annotate/", views.PreAnnotateView.as_view(), name="pre_annotate"),
    path("<int:pk>/pre-annotate-all/", views.PreAnnotateAllView.as_view(), name="pre_annotate_all"),
    path("<int:pk>/export/", views.ExportProjectView.as_view(), name="export"),
    path("<int:pk>/frame/", views.FrameImageView.as_view(), name="frame_image"),
    # Review redirects to detail (review is now integrated into project page + editor)
    path("<int:pk>/review/", views.ProjectDetailView.as_view(), name="review"),
]
