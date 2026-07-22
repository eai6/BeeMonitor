from django.contrib.auth import get_user_model
from django.test import TestCase
from apps.annotations.models import AnnotationProject
from apps.videos.models import Video

User = get_user_model()


class WorkflowBarTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Summer_2026", classes=["bee", "nest hole", "box"])
        v = Video.objects.create(user=self.user, title="clip", storage_key="a/c.mp4",
                                 file_size_bytes=1, status=Video.Status.READY)
        self.project.videos.add(v)
        self.client.force_login(self.user)
        self.html = self.client.get(f"/annotations/{self.project.pk}/").content.decode()

    # Anchored on the heading markup, not bare text: "Annotate" is a prefix of
    # the "Annotated Frames" stat card further up the page.
    HEADINGS = [
        '>Add videos<', '>Sample frames<', '>Annotate<',
        '>Auto-label every sampled frame<', '>Export dataset<',
    ]

    def test_all_steps_render_in_workflow_order(self):
        positions = []
        for h in self.HEADINGS:
            self.assertIn(h, self.html, f"missing heading {h}")
            positions.append(self.html.index(h))
        self.assertEqual(positions, sorted(positions),
                         "workflow steps are out of order")

    def test_duplicate_field_labels_are_disambiguated(self):
        """'Every Nth'/'Max frames' appeared twice meaning different things."""
        self.assertNotIn(">Every Nth<", self.html)
        self.assertIn("Every Nth frame", self.html)
        self.assertIn("Max per video", self.html)

    def test_gpu_cost_is_visible_on_each_step(self):
        self.assertIn("No GPU", self.html)
        self.assertIn("Uses GPU", self.html)

    def test_both_forms_and_their_actions_survive(self):
        self.assertIn('id="sample-form"', self.html)
        self.assertIn('id="preannotate-form"', self.html)
        self.assertIn(f"/annotations/{self.project.pk}/sample-frames/", self.html)
        self.assertIn(f"/annotations/{self.project.pk}/pre-annotate-all/", self.html)

    def test_every_project_class_is_offered(self):
        for cls in ("bee", "nest hole", "box"):
            self.assertIn(f'value="{cls}"', self.html)

    def test_video_checkboxes_still_bind_to_the_preannotate_form(self):
        self.assertIn('form="preannotate-form"', self.html)

    def test_stale_step_numbers_are_gone_from_buttons(self):
        self.assertNotIn("1 · Sample", self.html)
        self.assertNotIn("2 · Auto-label", self.html)
