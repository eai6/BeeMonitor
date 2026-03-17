from django.contrib import admin

from .models import Job, JobResult


class JobResultInline(admin.StackedInline):
    model = JobResult
    extra = 0


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = ("pk", "user", "video", "status", "progress_pct", "created_at", "completed_at")
    list_filter = ("status",)
    search_fields = ("user__username", "video__title", "modal_job_id")
    readonly_fields = ("created_at",)
    inlines = [JobResultInline]


@admin.register(JobResult)
class JobResultAdmin(admin.ModelAdmin):
    list_display = ("job", "total_events", "entry_count", "exit_count", "unique_tracks", "nest_count")
