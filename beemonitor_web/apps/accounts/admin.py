from django.contrib import admin

from .models import APIKey, Coupon, CouponRedemption, UserProfile


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
