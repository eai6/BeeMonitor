from django.contrib import admin

from .models import Annotation, AnnotationProject


@admin.register(AnnotationProject)
class AnnotationProjectAdmin(admin.ModelAdmin):
    list_display = ["name", "user", "created_at"]
    list_filter = ["created_at"]
    search_fields = ["name", "user__username"]


@admin.register(Annotation)
class AnnotationAdmin(admin.ModelAdmin):
    list_display = ["project", "video", "frame_number", "created_at"]
    list_filter = ["project", "created_at"]
    search_fields = ["video__title"]
