from django.contrib import admin

from .models import UsageLog, WebhookEndpoint


@admin.register(UsageLog)
class UsageLogAdmin(admin.ModelAdmin):
    list_display = ("user", "api_key", "method", "endpoint", "status_code", "response_time_ms", "created_at")
    list_filter = ("method", "status_code")
    search_fields = ("user__username", "endpoint")
    readonly_fields = ("created_at",)


@admin.register(WebhookEndpoint)
class WebhookEndpointAdmin(admin.ModelAdmin):
    list_display = ("user", "url", "is_active", "created_at")
    list_filter = ("is_active",)
    search_fields = ("user__username", "url")
    readonly_fields = ("created_at",)
