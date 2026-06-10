from django.urls import path

from . import views
from .assistant import views as assistant_views

app_name = "setup"

urlpatterns = [
    path("", views.SetupIndexView.as_view(), name="index"),
    path("d/<int:pk>/", views.WalkthroughView.as_view(), name="walkthrough"),
    path("d/<int:pk>/unit/", views.SetUnitTypeView.as_view(), name="set_unit"),
    path("d/<int:pk>/step/", views.StepActionView.as_view(), name="step_action"),
    path("d/<int:pk>/verify/<slug:check_id>/", views.VerifyView.as_view(), name="verify"),
    # AI assistant
    path("assistant/send/", assistant_views.AssistantSendView.as_view(), name="assistant_send"),
]
