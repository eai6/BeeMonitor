from django.contrib import admin

from .models import Video


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "status", "file_size_bytes", "duration_seconds", "uploaded_at")
    list_filter = ("status",)
    search_fields = ("title", "user__username")
    readonly_fields = ("uploaded_at",)
