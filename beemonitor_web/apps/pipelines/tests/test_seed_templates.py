"""The seeded templates are the first thing a user sees — they must not drift.

The load-bearing assertion here is that no template uses a hidden block: that is
what stops the palette and the shipped templates from diverging again (a template
built on a block nobody can add is a dead end the user can't reproduce).
"""

from io import StringIO

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase

from apps.pipelines.lessons import LESSONS
from apps.pipelines.models import Pipeline
from apps.pipelines.registry import get_block, validate_steps

User = get_user_model()


class SeedTemplateTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        call_command("seed_pipeline_templates", user="alice", stdout=StringIO())

    def _templates(self):
        return list(Pipeline.objects.filter(is_template=True))

    def test_templates_are_created(self):
        titles = {p.title for p in self._templates()}
        self.assertEqual(titles, {
            "Foraging trips", "Flower / ROI visitation", "Individual bee IDs",
            "Colony activity", "Interactions",
        })

    def test_seeding_twice_is_idempotent(self):
        before = Pipeline.objects.count()
        call_command("seed_pipeline_templates", user="alice", stdout=StringIO())
        self.assertEqual(Pipeline.objects.count(), before)

    def test_every_template_validates_clean(self):
        for template in self._templates():
            self.assertEqual(validate_steps(template.steps), [], template.title)

    def test_no_template_uses_a_hidden_block(self):
        """A template built on a block that isn't in the palette can't be
        reproduced by the user who clones it."""
        for template in self._templates():
            for step in template.steps:
                block = get_block(step["block_type"])
                self.assertFalse(
                    block.get("hidden"),
                    f"{template.title} uses hidden block {step['block_type']}",
                )

    def test_every_template_starts_from_a_video(self):
        for template in self._templates():
            self.assertTrue(
                any(s["block_type"] == "input.video" for s in template.steps),
                template.title,
            )

    def test_every_template_is_the_three_modules(self):
        """video → Detect → MOT → Analyzer/Identity, plus optional reference
        nodes. Order is not fixed: a template may add a reference.layout or a
        second Detect node (one per class) anywhere before the analyzer."""
        for template in self._templates():
            kinds = [s["block_type"].split(".", 1)[0] for s in template.steps]
            self.assertEqual(kinds[0], "input", template.title)
            self.assertIn("detect", kinds, template.title)
            self.assertIn("track", kinds, template.title)
            self.assertTrue({"analyze", "identify"} & set(kinds), template.title)

    def test_detect_nodes_always_name_a_class(self):
        """One node per class only works if every Detect node says which."""
        for template in self._templates():
            for step in template.steps:
                if step["block_type"] != "detect.objects":
                    continue
                self.assertTrue((step.get("config") or {}).get("label"),
                                f"{template.title}: Detect node with no label")

    def test_every_lesson_resolves_to_a_seeded_template(self):
        titles = {p.title for p in self._templates()}
        for slug, lesson in LESSONS.items():
            self.assertIn(lesson["template"], titles, slug)
