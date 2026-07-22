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


def _detect(label, **extra):
    """A Detect node for one class. Every Detect node on a video shares one GPU
    pass, so a template can name as many classes as it needs for free."""
    return {"model_family": "yolo", "label": label, "confidence": 0.4, **extra}


_MOT = {"tracker": "beetrack"}

# Every template is the same three modules recombined — that is the point of the
# abstraction. Titles are load-bearing: ``lessons.py`` resolves each lesson to its
# template by title, so renaming one orphans a lesson.
TEMPLATES = [
    {
        "title": "Foraging trips",
        "description": "Detect bees and the nest tubes they use, track the bees, and "
                       "derive foraging-trip events (exit → entry at a tube).",
        "steps": [
            _s("v", "input.video"),
            _s("d", "detect.objects", _detect("bee"), {"video": "v"}),
            _s("m", "track.mot", _MOT, {"detections": "d"}),
            # The reference is the device's saved hotel + nest-tube layout.
            _s("r", "reference.layout", {"source": "device_layout"}, {"video": "v"}),
            _s("f", "analyze.foraging_trips", {"event_confidence": 0.6},
               {"tracks": "m", "rois": "r"}),
        ],
    },
    {
        "title": "Flower / ROI visitation",
        "description": "Track insects and count unique visits to a region you draw "
                       "(a flower, a patch, a nest entrance).",
        "steps": [
            _s("v", "input.video"),
            _s("d", "detect.objects", _detect("bee"), {"video": "v"}),
            _s("m", "track.mot", _MOT, {"detections": "d"}),
            _s("r", "reference.layout", {"source": "drawn", "regions": "[]"}, {"video": "v"}),
            _s("g", "analyze.visitation", {}, {"tracks": "m", "rois": "r"}),
        ],
    },
    {
        "title": "Individual bee IDs",
        "description": "Track bees and read their colour marker IDs per trajectory.",
        "steps": [
            _s("v", "input.video"),
            _s("d", "detect.objects", _detect("bee"), {"video": "v"}),
            _s("m", "track.mot", _MOT, {"detections": "d"}),
            _s("i", "identify.marker", {"marker_type": "auto"}, {"tracks": "m"}),
        ],
    },
    {
        "title": "Colony activity",
        "description": "Measure how much insect activity there is over time, without "
                       "asking who went where.",
        "steps": [
            _s("v", "input.video"),
            _s("d", "detect.objects", _detect("bee"), {"video": "v"}),
            _s("m", "track.mot", _MOT, {"detections": "d"}),
            _s("a", "analyze.detection_count",
               {"metric": "over_time", "bin_seconds": 5}, {"detections": "m"}),
        ],
    },
    {
        "title": "Interactions",
        "description": "Find proximity interactions — insect ↔ insect, and insect ↔ "
                       "reference object. Two Detect nodes, one per class: the bees "
                       "are tracked, the nest tubes are the reference.",
        "steps": [
            _s("v", "input.video"),
            _s("d", "detect.objects", _detect("bee"), {"video": "v"}),
            _s("m", "track.mot", _MOT, {"detections": "d"}),
            # A second Detect node, aimed at the reference class — it rides the
            # same GPU pass as the bee detector.
            _s("n", "detect.objects", _detect("nest"), {"video": "v"}),
            _s("x", "analyze.interaction", {"interaction_type": "all"},
               {"tracks": "m", "rois": "n"}),
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
