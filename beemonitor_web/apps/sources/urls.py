from django.urls import path

from . import views

app_name = "sources"

urlpatterns = [
    path("", views.SourceListView.as_view(), name="list"),
    path("add/", views.SourceCreateView.as_view(), name="add"),
    path("<int:pk>/browse/", views.SourceBrowseView.as_view(), name="browse"),
    path("<int:pk>/sync/", views.SourceSyncView.as_view(), name="sync"),
    path("<int:pk>/delete/", views.SourceDeleteView.as_view(), name="delete"),
]
