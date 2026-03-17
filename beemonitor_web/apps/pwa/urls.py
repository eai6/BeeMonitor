from django.urls import path

from apps.pwa.views import manifest, service_worker, OfflineView

urlpatterns = [
    path("manifest.json", manifest, name="manifest"),
    path("sw.js", service_worker, name="service-worker"),
    path("offline/", OfflineView.as_view(), name="offline"),
]
