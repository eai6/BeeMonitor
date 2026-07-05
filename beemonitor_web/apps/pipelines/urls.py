from django.urls import path

from . import views

app_name = "pipelines"

urlpatterns = [
    path("", views.pipeline_list, name="list"),
    path("create/", views.pipeline_create, name="create"),
    path("lessons/", views.lesson_list, name="lesson_list"),
    path("lessons/<slug:slug>/", views.lesson_detail, name="lesson_detail"),
    path("run-on-videos/", views.run_on_videos, name="run_on_videos"),
    path("runs/", views.run_list, name="run_list"),
    path("batch/<uuid:batch_id>/", views.batch_detail, name="batch_detail"),
    path("batch/<uuid:batch_id>/trips.csv", views.batch_trips_csv, name="batch_trips_csv"),
    path("batch/<uuid:batch_id>/combined/<str:kind>.csv", views.batch_combined_csv, name="batch_combined_csv"),
    path("<uuid:pk>/", views.pipeline_editor, name="editor"),
    path("<uuid:pk>/save-graph/", views.save_graph, name="save_graph"),
    path("<uuid:pk>/colab/", views.open_in_colab, name="open_in_colab"),
    path("<uuid:pk>/rename/", views.pipeline_rename, name="rename"),
    path("<uuid:pk>/delete/", views.pipeline_delete, name="delete"),
    path("<uuid:pk>/clone/", views.clone_pipeline, name="clone"),
    path("<uuid:pk>/run/", views.run_pipeline, name="run"),
    path("<uuid:pk>/run/<uuid:run_id>/", views.run_detail, name="run_detail"),
    path("<uuid:pk>/run/<uuid:run_id>/status/", views.run_status, name="run_status"),
    path("<uuid:pk>/run/<uuid:run_id>/rerun/", views.rerun, name="rerun"),
    path("<uuid:pk>/run/<uuid:run_id>/output/<str:step_id>.csv", views.run_output_csv, name="run_output_csv"),
]
