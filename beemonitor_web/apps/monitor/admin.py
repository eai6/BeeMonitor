from django.contrib import admin

from .models import Activity, ActivityFrame, Detection, Observation, Taxon


class ActivityFrameInline(admin.TabularInline):
    model = ActivityFrame
    extra = 0
    fields = ("kind", "storage_key", "motion_score", "captured_at")
    readonly_fields = fields


@admin.register(Activity)
class ActivityAdmin(admin.ModelAdmin):
    list_display = ("id", "device", "activity_uid", "started_at", "status",
                    "best_taxon", "best_confidence")
    list_filter = ("status", "device")
    search_fields = ("activity_uid", "device__name")
    date_hierarchy = "started_at"
    inlines = [ActivityFrameInline]


@admin.register(Taxon)
class TaxonAdmin(admin.ModelAdmin):
    list_display = ("name", "rank", "common_name", "gbif_id")
    list_filter = ("rank",)
    search_fields = ("name", "common_name", "gbif_id")


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ("id", "frame", "model", "taxon", "confidence")
    list_filter = ("model",)


@admin.register(Observation)
class ObservationAdmin(admin.ModelAdmin):
    list_display = ("id", "activity", "taxon", "individual_count", "confidence", "status")
    list_filter = ("status",)
