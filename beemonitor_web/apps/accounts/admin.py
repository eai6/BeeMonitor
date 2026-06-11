from django.contrib import admin
from django.contrib.auth import get_user_model
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin

from .dedupe import dedupe_all
from .models import APIKey, Coupon, CouponRedemption, UserProfile

User = get_user_model()
admin.site.unregister(User)


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    """Default user admin + a duplicate-email consolidation action."""

    actions = ["merge_duplicate_emails"]

    @admin.action(description="Merge duplicate-email accounts (keep most active)")
    def merge_duplicate_emails(self, request, queryset):
        emails = {u.email.lower() for u in queryset if u.email}
        if not emails:
            self.message_user(request, "Selected users have no email to dedupe.")
            return
        deleted, groups = 0, 0
        for em in emails:
            for grp in dedupe_all(apply=True, merge=True, email=em):
                groups += 1
                deleted += sum(1 for a, _u, _n in grp["actions"] if a == "deleted")
        self.message_user(
            request,
            f"Consolidated {groups} duplicate group(s); removed {deleted} duplicate "
            "account(s) after merging their data into the kept account.")


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


class CouponRedemptionInline(admin.TabularInline):
    model = CouponRedemption
    extra = 0
    readonly_fields = ("user", "redeemed_at", "credits_added", "previous_tier", "new_tier")


@admin.register(Coupon)
class CouponAdmin(admin.ModelAdmin):
    list_display = ("code", "coupon_type", "credits_amount", "upgrade_tier", "times_redeemed", "max_redemptions", "is_active", "expires_at")
    list_filter = ("coupon_type", "is_active")
    search_fields = ("code",)
    readonly_fields = ("times_redeemed", "created_at")
    inlines = [CouponRedemptionInline]


@admin.register(CouponRedemption)
class CouponRedemptionAdmin(admin.ModelAdmin):
    list_display = ("user", "coupon", "redeemed_at", "credits_added", "previous_tier", "new_tier")
    list_filter = ("coupon__coupon_type",)
    search_fields = ("user__username", "coupon__code")
    readonly_fields = ("redeemed_at",)
