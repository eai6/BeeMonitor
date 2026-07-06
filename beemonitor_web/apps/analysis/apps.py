from django.apps import AppConfig


class AnalysisConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.analysis"
    verbose_name = "Analysis"

    def ready(self):
        # Browser-independent convergence loop for jobs + pipeline runs.
        from . import reconcile
        reconcile.start_background_reconciler()
