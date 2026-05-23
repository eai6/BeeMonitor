from django.urls import path
from django.views.decorators.csrf import csrf_exempt
from rest_framework.routers import DefaultRouter

from .uploads import UploadCompleteView, UploadInitiateView
from .views import (
    APIKeyViewSet,
    DataSourceViewSet,
    HealthViewSet,
    JobViewSet,
    LoginView,
    ProfileView,
    RegisterView,
    ResetPasswordView,
    SyncJobsView,
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
    path("auth/reset-password/", csrf_exempt(ResetPasswordView.as_view({"post": "create"})), name="auth-reset-password"),
    path("auth/profile/", ProfileView.as_view({"get": "list"}), name="auth-profile"),
    path("jobs/sync/", SyncJobsView.as_view({"post": "create"}), name="jobs-sync"),
    # Pi-side video upload (Phase 3 of the AWS migration plan).
    path("uploads/initiate", csrf_exempt(UploadInitiateView.as_view()), name="uploads-initiate"),
    path("uploads/complete", csrf_exempt(UploadCompleteView.as_view()), name="uploads-complete"),
] + router.urls
