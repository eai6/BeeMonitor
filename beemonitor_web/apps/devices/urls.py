from django.urls import path

from . import views

app_name = "devices"

urlpatterns = [
    path("", views.DeviceListView.as_view(), name="list"),
    path("add/", views.DeviceCreateView.as_view(), name="add"),
    path("<int:pk>/", views.DeviceDetailView.as_view(), name="detail"),
    path("<int:pk>/created/", views.DeviceCreatedView.as_view(), name="created"),
    path("<int:pk>/revoke/", views.DeviceRevokeView.as_view(), name="revoke"),
    path("<int:pk>/reactivate/", views.DeviceReactivateView.as_view(), name="reactivate"),
    path("<int:pk>/delete/", views.DeviceDeleteView.as_view(), name="delete"),
]
