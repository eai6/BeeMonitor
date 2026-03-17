from django.urls import path

from . import views

app_name = "analysis"

urlpatterns = [
    path("", views.JobListView.as_view(), name="list"),
    path("new/", views.JobCreateView.as_view(), name="new"),
    path("<int:pk>/", views.JobDetailView.as_view(), name="detail"),
    path("<int:pk>/results/", views.JobResultsView.as_view(), name="results"),
]
