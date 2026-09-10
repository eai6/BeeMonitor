"""The rules, enforced by the views that matter.

test_access pins the rules as properties. These pin that the views actually
consult them — the conversion across ~20 call sites is where a mistake would
live, and its failure mode is silent.
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations.models import (Annotation, AnnotationProject,
                                     ClipAssignment, ProjectShare)
from apps.videos.models import Video

User = get_user_model()


class ViewAccessTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.project = AnnotationProject.objects.create(
            user=self.owner, name="P", classes=["bee"])
        self.people = {"owner": self.owner}
        for role in ("viewer", "annotator", "reviewer", "manager"):
            u = User.objects.create_user(role, password="x")
            ProjectShare.objects.create(project=self.project, user=u, role=role)
            self.people[role] = u
        self.stranger = User.objects.create_user("stranger", password="x")
        self.video = Video.objects.create(
            user=self.owner, title="c", storage_key="va/c.mp4",
            file_size_bytes=1, status=Video.Status.READY)
        self.project.videos.add(self.video)
        Annotation.objects.create(project=self.project, video=self.video,
                                  frame_number=0, boxes=[])

    def as_(self, who):
        self.client.force_login(self.people[who] if who in self.people else who)

    def status(self, who, url, method="get", **kw):
        self.as_(who)
        return getattr(self.client, method)(url, **kw).status_code


class ReadPathTests(ViewAccessTestCase):
    def test_every_role_can_open_the_project(self):
        url = reverse("annotations:detail", args=[self.project.pk])

        for role in self.people:
            self.assertEqual(self.status(role, url), 200, role)

    def test_a_stranger_cannot(self):
        url = reverse("annotations:detail", args=[self.project.pk])

        self.assertEqual(self.status(self.stranger, url), 404)

    def test_a_shared_project_appears_in_the_list(self):
        self.as_("annotator")

        html = self.client.get(reverse("annotations:list")).content.decode()

        self.assertIn("P", html)

    def test_a_viewer_may_export_the_dataset(self):
        """Sharing a dataset is what sharing is for."""
        from unittest.mock import MagicMock, patch

        url = reverse("annotations:export", args=[self.project.pk])
        self.as_("viewer")

        # Storage is mocked: the access decision is what is under test, not
        # whether an S3 round-trip succeeds without credentials.
        with patch("config.storage.get_s3_client", return_value=MagicMock()):
            self.assertNotEqual(self.client.get(url).status_code, 404)

    def test_a_stranger_may_not_export(self):
        from unittest.mock import MagicMock, patch

        url = reverse("annotations:export", args=[self.project.pk])
        self.as_(self.stranger)

        with patch("config.storage.get_s3_client", return_value=MagicMock()):
            self.assertEqual(self.client.get(url).status_code, 404)


class FrameServingTests(ViewAccessTestCase):
    """The hinge: check ownership and shares break; check nothing and frames leak."""

    def url(self):
        return (reverse("annotations:frame_image", args=[self.project.pk])
                + f"?video={self.video.pk}&frame=0")

    def test_a_collaborator_can_see_the_frames(self):
        for role in ("viewer", "annotator", "reviewer", "manager"):
            self.assertNotEqual(self.status(role, self.url()), 404, role)

    def test_a_stranger_cannot(self):
        self.assertEqual(self.status(self.stranger, self.url()), 404)

    def test_a_clip_outside_the_project_is_refused(self):
        """Reaching another project's frames by guessing a video id."""
        other = Video.objects.create(user=self.owner, title="x",
                                     storage_key="va/x.mp4", file_size_bytes=1,
                                     status=Video.Status.READY)
        url = (reverse("annotations:frame_image", args=[self.project.pk])
               + f"?video={other.pk}&frame=0")

        self.assertEqual(self.status("manager", url), 404)


class WritePathTests(ViewAccessTestCase):
    def save_url(self):
        return reverse("annotations:save", args=[self.project.pk])

    def payload(self):
        return json.dumps({"video_id": self.video.pk, "frame_number": 0,
                           "boxes": [{"x": 1, "y": 1, "w": 1, "h": 1,
                                      "label": "bee"}]})

    def save_as(self, who):
        self.as_(who)
        return self.client.post(self.save_url(), self.payload(),
                                content_type="application/json").status_code

    def test_a_viewer_cannot_draw(self):
        self.assertEqual(self.save_as("viewer"), 404)

    def test_an_annotator_cannot_draw_on_an_unassigned_clip(self):
        self.assertEqual(self.save_as("annotator"), 403)

    def test_an_annotator_can_draw_on_their_own_clip(self):
        ClipAssignment.objects.create(project=self.project, video=self.video,
                                      user=self.people["annotator"])

        self.assertEqual(self.save_as("annotator"), 200)

    def test_a_reviewer_can_draw_on_anyone_s_clip(self):
        ClipAssignment.objects.create(project=self.project, video=self.video,
                                      user=self.people["annotator"])

        self.assertEqual(self.save_as("reviewer"), 200)

    def test_a_stranger_cannot(self):
        self.as_(self.stranger)
        resp = self.client.post(self.save_url(), self.payload(),
                                content_type="application/json")

        self.assertEqual(resp.status_code, 404)


class GpuAndStructureTests(ViewAccessTestCase):
    """A labeller must not spend someone else's GPU budget, or restructure the
    project underneath work already done."""

    def test_only_a_manager_may_sample(self):
        url = reverse("annotations:sample_frames", args=[self.project.pk])

        self.assertEqual(self.status("annotator", url, "post"), 404)
        self.assertEqual(self.status("reviewer", url, "post"), 404)
        self.assertIn(self.status("manager", url, "post"), (200, 302))

    def test_only_a_manager_may_auto_label(self):
        url = reverse("annotations:pre_annotate", args=[self.project.pk])

        self.assertEqual(self.status("annotator", url, "post"), 404)
        self.assertEqual(self.status("reviewer", url, "post"), 404)

    def test_only_a_manager_may_add_clips(self):
        url = reverse("annotations:add_videos_page", args=[self.project.pk])

        self.assertEqual(self.status("reviewer", url), 404)
        self.assertEqual(self.status("manager", url), 200)

    def test_only_a_manager_may_edit_the_class_list(self):
        """Changing classes mid-project invalidates finished work."""
        url = reverse("annotations:settings", args=[self.project.pk])

        self.assertEqual(self.status("reviewer", url), 404)
        self.assertEqual(self.status("manager", url), 200)

    def test_only_the_owner_may_delete(self):
        url = reverse("annotations:delete", args=[self.project.pk])

        self.assertEqual(self.status("manager", url, "post"), 404)
        self.assertTrue(AnnotationProject.objects.filter(pk=self.project.pk).exists())


class TrainingOnSharedProjectTests(ViewAccessTestCase):
    """A shared dataset is trainable by the people it is shared with.

    Reading the dataset is a viewer capability, and the training run spends the
    TRAINER's GPU budget, not the project owner's — so there is no reason to
    keep a collaborator from training on the work they helped produce.
    """

    def test_a_shared_project_is_offered_on_the_training_form(self):
        from apps.training.forms import TrainingCreateForm

        form = TrainingCreateForm(user=self.people["annotator"])

        self.assertIn(self.project, form.fields["project"].queryset)

    def test_a_strangers_project_is_not(self):
        from apps.training.forms import TrainingCreateForm

        form = TrainingCreateForm(user=self.stranger)

        self.assertNotIn(self.project, form.fields["project"].queryset)
