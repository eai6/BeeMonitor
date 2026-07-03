from django.urls import path

from . import views

app_name = "pipelines"

urlpatterns = [
    path("", views.pipeline_list, name="list"),
    path("create/", views.pipeline_create, name="create"),
    path("<uuid:pk>/", views.pipeline_editor, name="editor"),
    path("<uuid:pk>/rename/", views.pipeline_rename, name="rename"),
    path("<uuid:pk>/delete/", views.pipeline_delete, name="delete"),
    path("<uuid:pk>/clone/", views.clone_pipeline, name="clone"),
    path("<uuid:pk>/add-step/", views.add_step, name="add_step"),
    path("<uuid:pk>/step/<str:step_id>/remove/", views.remove_step, name="remove_step"),
    path("<uuid:pk>/step/<str:step_id>/move/<str:direction>/", views.move_step, name="move_step"),
    path("<uuid:pk>/step/<str:step_id>/configure/", views.configure_step, name="configure_step"),
    path("<uuid:pk>/run/", views.run_pipeline, name="run"),
    path("<uuid:pk>/run/<uuid:run_id>/", views.run_detail, name="run_detail"),
    path("<uuid:pk>/run/<uuid:run_id>/status/", views.run_status, name="run_status"),
]
