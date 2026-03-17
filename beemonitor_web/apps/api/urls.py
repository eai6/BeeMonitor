from rest_framework.routers import DefaultRouter

from .views import (
    APIKeyViewSet,
    DataSourceViewSet,
    HealthViewSet,
    JobViewSet,
    VideoViewSet,
    WebhookViewSet,
)

router = DefaultRouter()
router.register(r"videos", VideoViewSet, basename="video")
router.register(r"jobs", JobViewSet, basename="job")
router.register(r"sources", DataSourceViewSet, basename="datasource")
router.register(r"api-keys", APIKeyViewSet, basename="apikey")
router.register(r"webhooks", WebhookViewSet, basename="webhook")
router.register(r"health", HealthViewSet, basename="health")

urlpatterns = router.urls
