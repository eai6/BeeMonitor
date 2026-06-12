from django.urls import path

from . import views
from .agent.views import AgentSendView

app_name = "monitor"

urlpatterns = [
    path("", views.ActivityListView.as_view(), name="activity_list"),
    path("agent/", views.AgentChatView.as_view(), name="agent_chat"),
    path("agent/send", AgentSendView.as_view(), name="agent_send"),
    path("<int:pk>/", views.ActivityDetailView.as_view(), name="activity_detail"),
]
