"""Admin for the Device model.

Creating a Device via the admin auto-generates its bmk_device_* credential
and surfaces the raw value via a success message — the only time it's
visible. Operators record it (or hand it to the Pi via ssh) before the
admin page navigates away.
"""

from django.contrib import admin, messages

from .models import Device


@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "prefix", "is_active", "last_seen_at", "created_at")
    list_filter = ("is_active", "owner")
    search_fields = ("name", "location", "prefix", "owner__username", "owner__email")
    readonly_fields = ("prefix", "key_hash", "last_seen_at", "created_at")
    actions = ["revoke_devices"]
    fields = (
        "owner",
        "name",
        "location",
        "is_active",
        "prefix",
        "key_hash",
        "last_seen_at",
        "created_at",
    )

    def save_model(self, request, obj, form, change):
        """On create: generate the credential. On edit: normal save."""
        if change:
            super().save_model(request, obj, form, change)
            return

        # First save — mint a fresh device key and show the raw value once.
        device, raw_key = Device.create_with_key(
            owner=obj.owner,
            name=obj.name,
            location=obj.location,
        )
        # Replace the in-memory object the admin is about to save so the
        # ModelAdmin's response_add uses the real instance.
        obj.pk = device.pk
        obj.key_hash = device.key_hash
        obj.prefix = device.prefix
        messages.success(
            request,
            f"Device created. Save this credential — it will not be shown again: "
            f" {raw_key}",
        )

    @admin.action(description="Revoke selected devices (is_active=False)")
    def revoke_devices(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f"Revoked {updated} device(s).")
