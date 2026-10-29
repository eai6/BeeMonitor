"""Recount every finished run's events and interactions now (memory/60).

The reconcile loop does the same, 20 runs per pass; this does them all at once.
No GPU. Runs whose tracking table predates the ``detected`` column are left as
they are.

    python manage.py recount_runs            # runs below the current rules
    python manage.py recount_runs --all      # every finished run
"""

from django.core.management.base import BaseCommand

from apps.pipelines import recount
from apps.pipelines.models import PipelineRun


class Command(BaseCommand):
    help = "Recount finished runs' events and interactions from their saved tracks"

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true",
                            help="Recount every finished run, not only older-rules ones.")

    def handle(self, *args, **opts):
        runs = PipelineRun.objects.filter(status=PipelineRun.Status.COMPLETED)
        if not opts["all"]:
            runs = runs.filter(counting_rules__lt=recount.CURRENT_RULES)
        recounted = left = failed = 0
        for run in runs.order_by("-completed_at").iterator():
            try:
                counts = recount.recount_run(run)
            except Exception as exc:  # noqa: BLE001 - one bad run must not stop the rest
                failed += 1
                self.stderr.write(f"run {run.pk}: {exc}")
                continue
            PipelineRun.objects.filter(pk=run.pk).update(counting_rules=recount.CURRENT_RULES)
            if counts:
                recounted += 1
            else:
                left += 1
        self.stdout.write(f"Recounted {recounted} run(s); left {left} as they were "
                          f"(no events/interactions step, or tracked before the detected "
                          f"column); {failed} failed.")
