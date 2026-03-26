from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from rest_framework.routers import DefaultRouter

from .views import (
    APIKeyViewSet,
    DataSourceViewSet,
    HealthViewSet,
    JobViewSet,
    LoginView,
    ProfileView,
    RegisterView,
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

urlpatterns = [
    path("auth/login/", csrf_exempt(LoginView.as_view({"post": "create"})), name="auth-login"),
    path("auth/register/", csrf_exempt(RegisterView.as_view({"post": "create"})), name="auth-register"),
    path("auth/profile/", ProfileView.as_view({"get": "list"}), name="auth-profile"),
] + router.urls
