"""
Seed the three flagship pipeline templates (foraging trips, flower/ROI visitation,
colony activity) described in ``memory/23_pipeline_builder_port_design.md``.

Templates need an owner (Pipeline.user is required); pass ``--user <username>`` or
the command falls back to the first superuser. Idempotent: re-running updates the
steps of the existing template rather than duplicating it.

    python manage.py seed_pipeline_templates --user alice
"""

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from apps.pipelines.models import Pipeline
from apps.pipelines.registry import validate_steps


def _s(step_id, block_type, config=None, inputs=None):
    step = {"id": step_id, "block_type": block_type, "config": config or {}}
    if inputs:
        step["inputs"] = inputs
    return step


TEMPLATES = [
    {
        "title": "Foraging trips",
        "description": "Detect the nest/hotel, track bees, and derive foraging-trip events.",
        "steps": [
            _s("v", "input.video"),
            _s("r", "roi.nest_layout", {"source": "device"}, {"in": "v"}),
            _s("t", "track.bee", {"confidence": 0.4}, {"video": "v", "rois": "r"}),
            _s("f", "analyze.foraging_trips", {}, {"tracks": "t"}),
            _s("o", "output.table", {}, {"in": "f"}),
        ],
    },
    {
        "title": "Flower / ROI visitation",
        "description": "Track bees, count visits to a drawn region, and identify species.",
        "steps": [
            _s("v", "input.video"),
            _s("r", "roi.draw", {"regions": "[]"}, {"in": "v"}),
            _s("t", "track.bee", {"confidence": 0.4}, {"video": "v", "rois": "r"}),
            _s("g", "analyze.visitation", {}, {"tracks": "t"}),
            _s("i", "identify.taxon", {"region_prior": "on"}, {"in": "t"}),
            _s("o", "output.chart", {"chart_type": "bar"}, {"in": "g"}),
        ],
    },
    {
        "title": "Colony activity",
        "description": "Measure in-nest colony activity (occupancy / motion over time).",
        "steps": [
            _s("v", "input.video"),
            _s("a", "analyze.colony_activity", {"metric": "occupancy"}, {"in": "v"}),
            _s("o", "output.table", {}, {"in": "a"}),
        ],
    },
]


class Command(BaseCommand):
    help = "Create/update the flagship pipeline templates."

    def add_arguments(self, parser):
        parser.add_argument("--user", help="Username to own the templates.")

    def handle(self, *args, **options):
        User = get_user_model()
        username = options.get("user")
        if username:
            owner = User.objects.filter(username=username).first()
            if not owner:
                raise CommandError(f"No user named '{username}'.")
        else:
            owner = User.objects.filter(is_superuser=True).order_by("id").first()
            if not owner:
                raise CommandError("No superuser found — pass --user <username>.")

        for spec in TEMPLATES:
            errs = validate_steps(spec["steps"])
            if errs:
                self.stderr.write(self.style.WARNING(
                    f"'{spec['title']}' has validation notes: {errs}"
                ))
            pipeline, created = Pipeline.objects.update_or_create(
                title=spec["title"],
                is_template=True,
                defaults={
                    "user": owner,
                    "description": spec["description"],
                    "steps": spec["steps"],
                },
            )
            verb = "Created" if created else "Updated"
            self.stdout.write(self.style.SUCCESS(f"{verb} template: {pipeline.title}"))
