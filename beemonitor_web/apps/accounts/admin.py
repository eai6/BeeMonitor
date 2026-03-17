from django.contrib import admin

from .models import APIKey, UserProfile


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("user", "organization", "tier", "monthly_job_count", "storage_used_bytes")
    list_filter = ("tier",)
    search_fields = ("user__username", "user__email", "organization")


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ("name", "user", "prefix", "key_type", "is_active", "last_used_at", "created_at")
    list_filter = ("key_type", "is_active")
    search_fields = ("name", "user__username", "prefix")
    readonly_fields = ("key_hash", "prefix", "created_at")
