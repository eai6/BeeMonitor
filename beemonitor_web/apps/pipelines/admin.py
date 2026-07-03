from django.contrib import admin

from .models import Pipeline, PipelineRun


@admin.register(Pipeline)
class PipelineAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "is_template", "updated_at")
    list_filter = ("is_template",)
    search_fields = ("title", "description", "user__username")
    readonly_fields = ("created_at", "updated_at")


@admin.register(PipelineRun)
class PipelineRunAdmin(admin.ModelAdmin):
    list_display = ("id", "pipeline", "user", "status", "started_at", "completed_at")
    list_filter = ("status",)
    search_fields = ("pipeline__title", "user__username")
    readonly_fields = ("started_at", "completed_at", "execution_time_ms")
