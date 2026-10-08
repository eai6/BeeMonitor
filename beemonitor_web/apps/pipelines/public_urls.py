"""Batch share links, open without signing in (memory/47). Mounted at /s/."""
from django.urls import path

from . import public_views

urlpatterns = [
    path("<str:token>/", public_views.public_batch, name="public_batch"),
    path("<str:token>/data/<str:kind>.csv", public_views.public_csv, name="public_batch_csv"),
    path("<str:token>/clip/<int:video_pk>/video", public_views.public_video, name="public_video"),
    path("<str:token>/clip/<int:video_pk>/still", public_views.public_thumbnail, name="public_thumbnail"),
    path("<str:token>/tracks/<int:job_pk>.json", public_views.public_overlay, name="public_overlay"),
]
