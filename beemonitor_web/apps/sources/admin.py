from django.contrib import admin

from .models import DataSource


@admin.register(DataSource)
class DataSourceAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "source_type", "is_connected", "last_synced_at", "created_at")
    list_filter = ("source_type", "is_connected")
    search_fields = ("name", "user__username")
    readonly_fields = ("created_at",)
