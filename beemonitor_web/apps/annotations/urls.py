from django.urls import path

from . import views

app_name = "annotations"

urlpatterns = [
    path("", views.ProjectListView.as_view(), name="list"),
    path("new/", views.ProjectCreateView.as_view(), name="create"),
    path("<int:pk>/", views.ProjectDetailView.as_view(), name="detail"),
    path("<int:pk>/add-videos/", views.AddVideosView.as_view(), name="add_videos"),
    path("<int:pk>/edit/", views.AnnotationEditorView.as_view(), name="editor"),
    path("<int:pk>/save/", views.SaveAnnotationView.as_view(), name="save"),
    path("<int:pk>/export/", views.ExportProjectView.as_view(), name="export"),
]
