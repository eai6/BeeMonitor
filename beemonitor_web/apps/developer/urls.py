from django.urls import path

from . import views

app_name = "developer"

urlpatterns = [
    path("", views.DeveloperPortalView.as_view(), name="index"),
]
