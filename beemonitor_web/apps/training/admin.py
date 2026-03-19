from django.contrib import admin

from .models import CustomModel, TrainingJob


@admin.register(TrainingJob)
class TrainingJobAdmin(admin.ModelAdmin):
    list_display = ["name", "user", "base_model", "status", "epochs", "created_at"]
    list_filter = ["status", "base_model", "gpu_tier", "created_at"]
    search_fields = ["name", "user__username"]


@admin.register(CustomModel)
class CustomModelAdmin(admin.ModelAdmin):
    list_display = ["name", "user", "model_type", "base_model", "is_active", "created_at"]
    list_filter = ["model_type", "is_active", "created_at"]
    search_fields = ["name", "user__username"]
