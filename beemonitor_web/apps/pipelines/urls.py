from django.urls import path

from . import views

app_name = "pipelines"

urlpatterns = [
    path("", views.pipeline_list, name="list"),
    path("create/", views.pipeline_create, name="create"),
    path("lessons/", views.lesson_list, name="lesson_list"),
    path("lessons/<slug:slug>/", views.lesson_detail, name="lesson_detail"),
    path("run-on-videos/", views.run_on_videos, name="run_on_videos"),
    path("<uuid:pk>/", views.pipeline_editor, name="editor"),
    path("<uuid:pk>/save-graph/", views.save_graph, name="save_graph"),
    path("<uuid:pk>/notebook/", views.export_notebook, name="export_notebook"),
    path("<uuid:pk>/rename/", views.pipeline_rename, name="rename"),
    path("<uuid:pk>/delete/", views.pipeline_delete, name="delete"),
    path("<uuid:pk>/clone/", views.clone_pipeline, name="clone"),
    path("<uuid:pk>/run/", views.run_pipeline, name="run"),
    path("<uuid:pk>/run/<uuid:run_id>/", views.run_detail, name="run_detail"),
    path("<uuid:pk>/run/<uuid:run_id>/status/", views.run_status, name="run_status"),
]
