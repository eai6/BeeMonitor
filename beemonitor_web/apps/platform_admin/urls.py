from django.urls import path

from . import views

app_name = "platform_admin"

urlpatterns = [
    path("", views.OverviewView.as_view(), name="overview"),
    path("users/", views.UserListView.as_view(), name="users"),
    path("users/<int:pk>/", views.UserDetailView.as_view(), name="user"),
    path("users/<int:pk>/<slug:action>/", views.UserActionView.as_view(), name="user_action"),
    path("codes/", views.CodesView.as_view(), name="codes"),
    path("codes/<int:pk>/deactivate/", views.CodeDeactivateView.as_view(), name="code_deactivate"),
    path("activity/", views.ActivityView.as_view(), name="activity"),
]
