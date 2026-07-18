from django.urls import path

from . import views

app_name = "devices"

urlpatterns = [
    path("", views.DeviceListView.as_view(), name="list"),
    path("add/", views.DeviceCreateView.as_view(), name="add"),
    path("enrollment/", views.DeviceEnrollmentView.as_view(), name="enrollment"),
    path("golden-image/", views.GoldenImageDownloadView.as_view(), name="golden_image"),
    path("fleet/update/", views.DeviceFleetUpdateView.as_view(), name="fleet_update"),
    path("enrollment/<int:pk>/revoke/", views.EnrollmentTokenRevokeView.as_view(), name="enrollment_revoke"),
    path("<int:pk>/", views.DeviceDetailView.as_view(), name="detail"),
    path("<int:pk>/created/", views.DeviceCreatedView.as_view(), name="created"),
    path("<int:pk>/edit/", views.DeviceEditView.as_view(), name="edit"),
    path("<int:pk>/revoke/", views.DeviceRevokeView.as_view(), name="revoke"),
    path("<int:pk>/reactivate/", views.DeviceReactivateView.as_view(), name="reactivate"),
    path("<int:pk>/delete/", views.DeviceDeleteView.as_view(), name="delete"),
    path("<int:pk>/wifi/", views.DeviceWifiView.as_view(), name="wifi"),
    path("<int:pk>/wifi-scan/", views.DeviceWifiScanView.as_view(), name="wifi_scan"),
    path("<int:pk>/cellular/", views.DeviceCellularView.as_view(), name="cellular"),
    path("<int:pk>/update/", views.DeviceUpdateView.as_view(), name="update"),
    path("<int:pk>/usb-transfer/", views.DeviceUsbTransferView.as_view(), name="usb_transfer"),
    path("<int:pk>/usb-eject/", views.DeviceUsbEjectView.as_view(), name="usb_eject"),
    path("<int:pk>/telemetry-rate/", views.DeviceTelemetryRateView.as_view(), name="telemetry_rate"),
    path("<int:pk>/bee-confirm/", views.DeviceBeeConfirmView.as_view(), name="bee_confirm"),
    path("<int:pk>/upload-mode/", views.DeviceVideoUploadModeView.as_view(), name="upload_mode"),
    path("<int:pk>/record-settings/", views.DeviceRecordSettingsView.as_view(), name="record_settings"),
    path("<int:pk>/upload-now/", views.DeviceUploadNowView.as_view(), name="upload_now"),
    path("<int:pk>/review-crops/", views.DeviceActivityCropsView.as_view(), name="review_crops"),
    path("<int:pk>/motion-tuning/", views.DeviceMotionTuningView.as_view(), name="motion_tuning"),
    path("<int:pk>/schedule/", views.DeviceScheduleView.as_view(), name="schedule"),
    path("<int:pk>/roi-editor/", views.DeviceRoiEditorView.as_view(), name="roi_editor"),
    path("<int:pk>/display-tz/", views.DeviceDisplayTzView.as_view(), name="display_tz"),
    path("<int:pk>/sync-clock/", views.DeviceSyncClockView.as_view(), name="sync_clock"),
    path("<int:pk>/request-image/", views.DeviceRequestImageView.as_view(), name="request_image"),
    path("<int:pk>/latest-image.json", views.DeviceLatestImageView.as_view(), name="latest_image"),
    path("<int:pk>/status.json", views.DeviceStatusView.as_view(), name="status"),
    path("<int:pk>/share/add/", views.DeviceShareAddView.as_view(), name="share_add"),
    path("<int:pk>/share/remove/", views.DeviceShareRemoveView.as_view(), name="share_remove"),
]
