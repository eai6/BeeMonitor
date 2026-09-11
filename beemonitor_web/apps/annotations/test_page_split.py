"""The clip workspace and the frame review are two pages, not one.

One page carried stage tiles, auto-label failures, clip filters, assignment,
selection actions, a row per clip, AND a filtered grid of every annotated frame.
Choosing what to annotate and checking what came back are different sittings
with different filters — clips by device and hour, frames by class and review
state — so stacking them meant scrolling past all of one to reach the other.

The progress bars are gone too. The per-clip bar's width was
"{labelled+reviewed}-{reviewed}%": Django's add filter falls back to string
concatenation, so it emitted invalid CSS and a clip with 219 of 219 frames
labelled drew an empty track.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.annotations.models import Annotation, AnnotationProject
from apps.videos.models import Video

User = get_user_model()


class PageSplitTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("ps", password="x")
        self.client.force_login(self.user)
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Split", classes=["bee"])
        self.video = Video.objects.create(
            user=self.user, title="clip-one", storage_key="ps/c.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now())
        self.project.videos.add(self.video)
        for i in range(3):
            Annotation.objects.create(project=self.project, video=self.video,
                                      frame_number=i, boxes=[{"label": "bee"}],
                                      frame_image_path=f"ps/f{i}.jpg")

    def _get(self, name):
        return self.client.get(reverse(name, kwargs={"pk": self.project.pk}))

    def test_both_pages_render(self):
        self.assertEqual(self._get("annotations:detail").status_code, 200)
        self.assertEqual(self._get("annotations:review").status_code, 200)

    def test_they_use_different_templates(self):
        self.assertIn("annotations/detail.html",
                      [t.name for t in self._get("annotations:detail").templates])
        self.assertIn("annotations/review.html",
                      [t.name for t in self._get("annotations:review").templates])

    def test_the_clips_page_no_longer_carries_the_frame_grid(self):
        html = self._get("annotations:detail").content.decode()

        self.assertIn("clip-one", html)          # the clip table is still here
        self.assertNotIn("Review</label>", html)  # the frame filters are not

    def test_the_review_page_carries_the_frames_and_their_filters(self):
        html = self._get("annotations:review").content.decode()

        self.assertIn("All Classes", html)
        self.assertIn("Not reviewed", html)

    def test_each_page_links_to_the_other(self):
        self.assertIn(reverse("annotations:review", kwargs={"pk": self.project.pk}),
                      self._get("annotations:detail").content.decode())
        self.assertIn(reverse("annotations:detail", kwargs={"pk": self.project.pk}),
                      self._get("annotations:review").content.decode())

    def test_review_respects_the_same_access_rules(self):
        self.client.force_login(User.objects.create_user("nosy", password="x"))

        self.assertEqual(self._get("annotations:review").status_code, 404)


class NoProgressBarsTests(TestCase):
    """The counts were right all along; only the graphics lied."""

    def setUp(self):
        self.user = User.objects.create_user("nb", password="x")
        self.client.force_login(self.user)
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Bars", classes=["bee"])
        video = Video.objects.create(
            user=self.user, title="c", storage_key="nb/c.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now())
        self.project.videos.add(video)
        Annotation.objects.create(project=self.project, video=video, frame_number=0,
                                  boxes=[{"label": "bee"}], frame_image_path="nb/f.jpg")

    def test_no_bar_survives_on_either_page(self):
        for name in ("annotations:detail", "annotations:review"):
            with self.subTest(page=name):
                html = self.client.get(
                    reverse(name, kwargs={"pk": self.project.pk})).content.decode()
                self.assertNotIn("h-1.5 rounded-full", html)

    def test_no_width_style_carries_a_broken_expression(self):
        """The old row bar emitted width:"100-0%" — a number, a dash, a number."""
        import re

        html = self.client.get(
            reverse("annotations:detail", kwargs={"pk": self.project.pk})).content.decode()

        for width in re.findall(r"width:\s*([^;\"']+)", html):
            self.assertNotRegex(width.strip(), r"^\d+-")

    def test_the_counts_are_still_reported(self):
        html = self.client.get(
            reverse("annotations:detail", kwargs={"pk": self.project.pk})).content.decode()

        self.assertIn("labelled", html)
