"""The project page: frames to review first, clips second (memory/39).

The frame grid had moved to its own /review/ page, under a clip workspace that
led the project page. Now that sampling and SAM 3 label in one GPU pass, what
people come to do is review, so the grid is the project page's first tab and
the clip table its second. /review/ redirects there.

The old per-clip progress bar's width was "{labelled+reviewed}-{reviewed}%":
Django's add filter falls back to string concatenation, so it emitted invalid
CSS. Bars are back, fed only by integers the view computed.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.annotations.models import Annotation, AnnotationProject
from apps.videos.models import Video

User = get_user_model()
BEE = {"x": 1, "y": 1, "w": 4, "h": 4, "class": "bee"}


class ProjectPageTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("ps", password="x")
        self.client.force_login(self.user)
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Split", classes=["bee", "wasp"])
        self.video = Video.objects.create(
            user=self.user, title="clip-one", storage_key="ps/c.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now())
        self.project.videos.add(self.video)
        for i in range(3):
            Annotation.objects.create(project=self.project, video=self.video,
                                      frame_number=i, boxes=[BEE, BEE] if i else [BEE],
                                      frame_image_path=f"ps/f{i}.jpg", reviewed=i == 0)

    def get(self, **params):
        return self.client.get(reverse("annotations:detail", args=[self.project.pk]), params)


class TabTests(ProjectPageTestCase):
    def test_frames_to_review_is_the_first_tab(self):
        r = self.get()
        self.assertEqual(r.context["tab"], "frames")
        self.assertIn("annotations/_frames_tab.html", [t.name for t in r.templates])
        self.assertEqual(len(r.context["frame_cards"]), 2)     # the two unreviewed

    def test_the_clip_table_is_the_second(self):
        r = self.get(tab="clips")
        self.assertIn("annotations/_clips_tab.html", [t.name for t in r.templates])
        self.assertIn("clip-one", r.content.decode())

    def test_three_numbers(self):
        m = self.get().context["metrics"]
        self.assertEqual((m["clips"], m["labelled"], m["reviewed"], m["to_review"], m["boxes"]),
                         (1, 3, 1, 2, 5))

    def test_export_replaces_back(self):
        html = self.get().content.decode()
        self.assertIn(reverse("annotations:export", args=[self.project.pk]), html)
        self.assertIn("All projects", html)
        self.assertNotIn(">Back<", html)

    def test_no_auto_label(self):
        html = self.get(tab="clips").content.decode()
        self.assertNotIn("Auto-label", html)
        self.assertNotIn("pre-annotate-all", html)


class FrameGridTests(ProjectPageTestCase):
    def test_status_toggle_counts(self):
        c = self.get().context["status_counts"]
        self.assertEqual((c["review"], c["reviewed"], c["all"]), (2, 1, 3))

    def test_reviewed_shows_the_reviewed(self):
        cards = self.get(status="reviewed").context["frame_cards"]
        self.assertEqual([c["frame_number"] for c in cards], [0])

    def test_class_filter(self):
        v = Video.objects.create(user=self.user, title="w", storage_key="ps/w.mp4",
                                 file_size_bytes=1, status=Video.Status.READY)
        self.project.videos.add(v)
        Annotation.objects.create(project=self.project, video=v, frame_number=9,
                                  boxes=[dict(BEE, **{"class": "wasp"})])
        cards = self.get(status="all", cls="wasp").context["frame_cards"]
        self.assertEqual([c["video_pk"] for c in cards], [v.pk])

    def test_cards_link_to_the_editor_with_the_filter(self):
        html = self.get(cls="bee").content.decode()
        # (The filter's own & is escaped in the attribute, as it should be.)
        self.assertIn(f"?video={self.video.pk}&frame=1&status=review&amp;cls=bee", html)

    def test_pages_of_sixty(self):
        for i in range(3, 70):
            Annotation.objects.create(project=self.project, video=self.video,
                                      frame_number=i, boxes=[BEE])
        r = self.get()
        self.assertEqual(len(r.context["frame_cards"]), 60)
        self.assertEqual(len(self.get(page=2).context["frame_cards"]), 69 - 60)

    def test_a_malformed_filter_is_ignored(self):
        r = self.get(status="bogus", who="x;drop", device="abc", **{"from": "yesterday"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.context["frame_filter"]["status"], "review")

    def test_review_redirects_here(self):
        r = self.client.get(reverse("annotations:review", args=[self.project.pk]))
        self.assertRedirects(r, reverse("annotations:detail", args=[self.project.pk]),
                             fetch_redirect_response=False)

    def test_a_stranger_cannot(self):
        self.client.force_login(User.objects.create_user("nosy", password="x"))
        self.assertEqual(self.get().status_code, 404)
        self.assertEqual(self.client.get(
            reverse("annotations:review", args=[self.project.pk])).status_code, 404)


class NoBrokenBarsTests(ProjectPageTestCase):
    def test_no_width_style_carries_a_broken_expression(self):
        """The old row bar emitted width:"100-0%" — a number, a dash, a number."""
        import re

        for tab in ("frames", "clips"):
            html = self.get(tab=tab).content.decode()
            for width in re.findall(r"width:\s*([^;\"']+)", html):
                self.assertNotRegex(width.strip(), r"^\d+-")
