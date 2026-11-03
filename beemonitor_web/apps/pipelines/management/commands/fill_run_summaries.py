"""Fill or refresh each run's slim copy of its results (memory/76), one run at
a time, and log how big results and summaries are."""

import json
import logging

from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Build PipelineRun.summary for runs whose summary is missing or out of date."

    def add_arguments(self, parser):
        parser.add_argument("--all", action="store_true", help="Rebuild every run's summary.")

    def handle(self, *args, **options):
        from apps.pipelines import runsummary
        from apps.pipelines.models import PipelineRun

        # Read just the version (a missing key must count as out of date,
        # which an exclude() on a JSON key would not promise everywhere).
        todo = [pk for pk, v in PipelineRun.objects.values_list("pk", "summary__v")
                if options["all"] or v != runsummary.VERSION]
        sizes, slim = [], []
        for pk in todo:
            context = PipelineRun.objects.filter(pk=pk).values_list("context", flat=True).first()
            summary = runsummary.build(context)
            PipelineRun.objects.filter(pk=pk).update(summary=summary)
            sizes.append(len(json.dumps(context or {}, default=str)))
            slim.append(len(json.dumps(summary, default=str)))
        if not sizes:
            self.stdout.write("run summaries: all current")
            return

        def pct(values, q):
            values = sorted(values)
            return values[min(len(values) - 1, int(q * len(values)))] // 1024

        msg = (f"run summaries: {len(sizes)} filled; results KB p50 {pct(sizes, .5)} "
               f"p95 {pct(sizes, .95)} max {max(sizes) // 1024}, total {sum(sizes) // 2**20} MB; "
               f"summary KB p50 {pct(slim, .5)} p95 {pct(slim, .95)} max {max(slim) // 1024}")
        logger.warning(msg)
        self.stdout.write(msg)
