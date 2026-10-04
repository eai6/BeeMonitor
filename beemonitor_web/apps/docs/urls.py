from django.urls import path
from django.views.generic import RedirectView

app_name = "docs"

urlpatterns = [
    # The docs live on the public site; one copy, not two.
    path("", RedirectView.as_view(url="https://eai6.github.io/BeeMonitor/"), name="index"),
]
