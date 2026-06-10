"""Generate a markdown setup guide from the structured content.

This proves the single-source-of-truth principle: the dashboard walkthrough and
this markdown render from the same ``apps/setup/content.py``. We write to a
*separate* generated file rather than overwriting the rich, hand-tuned
``hardware/README.md`` (which carries deep reference material the structured
content intentionally condenses).

    python manage.py export_setup_guide
    python manage.py export_setup_guide --out ../hardware/SETUP_GUIDE.generated.md
"""

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.setup import content


class Command(BaseCommand):
    help = "Render the guided-setup content to a markdown file."

    def add_arguments(self, parser):
        default = Path(settings.BASE_DIR).parent / "hardware" / "SETUP_GUIDE.generated.md"
        parser.add_argument("--out", default=str(default),
                            help="Output markdown path (default: hardware/SETUP_GUIDE.generated.md)")

    def handle(self, *args, **opts):
        out = Path(opts["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content.as_markdown())
        self.stdout.write(self.style.SUCCESS(
            f"Wrote {len(content.STEPS)} steps across "
            f"{len(content.PHASES)} phases to {out}"))
