from django.urls import path

from . import views

app_name = "devices"

urlpatterns = [
    path("", views.DeviceListView.as_view(), name="list"),
    path("add/", views.DeviceCreateView.as_view(), name="add"),
    path("<int:pk>/", views.DeviceDetailView.as_view(), name="detail"),
    path("<int:pk>/created/", views.DeviceCreatedView.as_view(), name="created"),
    path("<int:pk>/edit/", views.DeviceEditView.as_view(), name="edit"),
    path("<int:pk>/revoke/", views.DeviceRevokeView.as_view(), name="revoke"),
    path("<int:pk>/reactivate/", views.DeviceReactivateView.as_view(), name="reactivate"),
    path("<int:pk>/delete/", views.DeviceDeleteView.as_view(), name="delete"),
    path("<int:pk>/wifi/", views.DeviceWifiView.as_view(), name="wifi"),
    path("<int:pk>/update/", views.DeviceUpdateView.as_view(), name="update"),
    path("<int:pk>/usb-transfer/", views.DeviceUsbTransferView.as_view(), name="usb_transfer"),
    path("<int:pk>/request-image/", views.DeviceRequestImageView.as_view(), name="request_image"),
    path("<int:pk>/latest-image.json", views.DeviceLatestImageView.as_view(), name="latest_image"),
    path("<int:pk>/share/add/", views.DeviceShareAddView.as_view(), name="share_add"),
    path("<int:pk>/share/remove/", views.DeviceShareRemoveView.as_view(), name="share_remove"),
]
