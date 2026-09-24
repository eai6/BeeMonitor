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

    def test_managers_see_add_videos_and_others_do_not(self):
        url = reverse("annotations:detail", args=[self.project.pk])
        add = reverse("annotations:add_videos_page", args=[self.project.pk])

        for role in self.people:
            self.as_(role)
            html = self.client.get(url).content.decode()
            self.assertEqual(f'href="{add}"' in html, role in ("owner", "manager"), role)

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


class PeoplePageTests(ViewAccessTestCase):
    def url(self):
        return reverse("annotations:people", args=[self.project.pk])

    def test_everyone_on_the_project_can_see_who_else_is(self):
        """Knowing who else is working on it is part of working on it."""
        for role in self.people:
            self.assertEqual(self.status(role, self.url()), 200, role)

    def test_a_stranger_cannot(self):
        self.assertEqual(self.status(self.stranger, self.url()), 404)

    def test_only_the_owner_gets_the_controls(self):
        self.as_("manager")
        manager_view = self.client.get(self.url()).content.decode()
        self.as_("owner")
        owner_view = self.client.get(self.url()).content.decode()

        invite = reverse("annotations:share_invite", args=[self.project.pk])
        self.assertNotIn(invite, manager_view)
        self.assertIn(invite, owner_view)

    def test_only_the_owner_may_invite(self):
        url = reverse("annotations:share_invite", args=[self.project.pk])

        self.assertEqual(self.status("manager", url, "post",
                                     data={"who": "stranger"}), 404)
        self.assertFalse(self.project.shares.filter(user=self.stranger).exists())

    def test_the_owner_can_invite_by_username(self):
        self.as_("owner")

        self.client.post(reverse("annotations:share_invite", args=[self.project.pk]),
                         {"who": "stranger", "role": "annotator"})

        self.assertEqual(self.project.role_for(self.stranger), "annotator")

    def test_inviting_an_unknown_account_says_so_rather_than_failing_quietly(self):
        self.as_("owner")

        resp = self.client.post(
            reverse("annotations:share_invite", args=[self.project.pk]),
            {"who": "nobody@example.com"}, follow=True)

        self.assertIn("No account matches", " ".join(
            str(m) for m in resp.context["messages"]))

    def test_removing_someone_returns_their_clips_to_the_pool(self):
        """The work is the project's, not theirs — it must not vanish with them."""
        ClipAssignment.objects.create(project=self.project, video=self.video,
                                      user=self.people["annotator"])
        share = self.project.shares.get(user=self.people["annotator"])
        self.as_("owner")

        self.client.post(reverse("annotations:share_update", args=[self.project.pk]),
                         {"share_id": share.pk, "remove": "1"})

        self.assertFalse(self.project.shares.filter(pk=share.pk).exists())
        self.assertFalse(self.project.assignments.exists())
        self.assertTrue(Annotation.objects.filter(project=self.project).exists())


class AssignmentViewTests(ViewAccessTestCase):
    def test_only_a_manager_may_assign(self):
        url = reverse("annotations:assign", args=[self.project.pk])

        self.assertEqual(self.status("reviewer", url, "post", data={
            "video_ids": [self.video.pk],
            "assignee": [self.people["annotator"].pk]}), 404)
        self.assertFalse(self.project.assignments.exists())

    def test_a_manager_can_hand_a_clip_out(self):
        self.as_("manager")

        self.client.post(reverse("annotations:assign", args=[self.project.pk]),
                         {"video_ids": [self.video.pk],
                          "assignee": [self.people["annotator"].pk]})

        self.assertEqual(self.project.assignments.get().user,
                         self.people["annotator"])

    def test_assigning_to_somebody_not_on_the_project_is_refused(self):
        """Otherwise the assignment names someone who cannot open it."""
        self.as_("manager")

        self.client.post(reverse("annotations:assign", args=[self.project.pk]),
                         {"video_ids": [self.video.pk],
                          "assignee": [self.stranger.pk]})

        self.assertFalse(self.project.assignments.filter(user=self.stranger).exists())

    def test_an_annotator_can_take_an_unassigned_clip(self):
        self.as_("annotator")

        self.client.post(reverse("annotations:claim", args=[self.project.pk]),
                         {"video_ids": [self.video.pk]})

        self.assertEqual(self.project.assignments.get().user,
                         self.people["annotator"])

    def test_taking_a_clip_someone_else_holds_is_refused_and_reported(self):
        ClipAssignment.objects.create(project=self.project, video=self.video,
                                      user=self.people["reviewer"])
        self.as_("annotator")

        resp = self.client.post(reverse("annotations:claim", args=[self.project.pk]),
                                {"video_ids": [self.video.pk]}, follow=True)

        self.assertEqual(self.project.assignments.get().user,
                         self.people["reviewer"])
        self.assertIn("already taken", " ".join(
            str(m) for m in resp.context["messages"]))

    def test_a_viewer_cannot_take_clips(self):
        self.assertEqual(
            self.status("viewer", reverse("annotations:claim", args=[self.project.pk]),
                        "post", data={"video_ids": [self.video.pk]}), 404)

    def test_the_list_can_be_filtered_to_one_person(self):
        ClipAssignment.objects.create(project=self.project, video=self.video,
                                      user=self.people["annotator"])
        self.as_("owner")
        url = reverse("annotations:detail", args=[self.project.pk])

        mine = self.client.get(url, {"assignee": self.people["annotator"].pk})
        nobody = self.client.get(url, {"assignee": "none"})

        self.assertIn(f'name="video_ids" value="{self.video.pk}"',
                      mine.content.decode())
        self.assertNotIn(f'name="video_ids" value="{self.video.pk}"',
                         nobody.content.decode())


class CollaboratorViewTests(ViewAccessTestCase):
    """What a collaborator opens the project to see."""

    def html(self, who, **params):
        self.as_(who)
        return self.client.get(
            reverse("annotations:detail", args=[self.project.pk]),
            params).content.decode()

    def test_a_collaborator_sees_their_own_workload(self):
        ClipAssignment.objects.create(project=self.project, video=self.video,
                                      user=self.people["annotator"])

        html = self.html("annotator")

        self.assertIn("Your work", html)
        self.assertIn("Start annotating", html)

    def test_the_owner_does_not_get_the_strip(self):
        """It is the collaborator's view of a project they do not run."""
        self.assertNotIn("Your work", self.html("owner"))

    def test_show_mine_narrows_to_their_clips(self):
        theirs = self.video
        others = Video.objects.create(user=self.owner, title="o",
                                      storage_key="va/o.mp4", file_size_bytes=1,
                                      status=Video.Status.READY)
        self.project.videos.add(others)
        ClipAssignment.objects.create(project=self.project, video=theirs,
                                      user=self.people["annotator"])
        ClipAssignment.objects.create(project=self.project, video=others,
                                      user=self.people["reviewer"])

        html = self.html("annotator", assignee="me")

        self.assertIn(f'name="video_ids" value="{theirs.pk}"', html)
        self.assertNotIn(f'name="video_ids" value="{others.pk}"', html)

    def test_a_viewer_is_not_offered_the_pool(self):
        """A viewer cannot annotate, so taking work would be a dead end."""
        html = self.html("viewer")

        self.assertNotIn("Unassigned pool", html)

    def test_a_collaborator_does_not_get_the_gpu_controls(self):
        """A button that 404s reads as a broken page, not as a permission you
        do not have."""
        html = self.html("annotator")

        self.assertNotIn(
            reverse("annotations:pre_annotate", args=[self.project.pk]), html)
        self.assertNotIn(
            reverse("annotations:sample_frames", args=[self.project.pk]), html)

    def test_a_manager_does(self):
        html = self.html("manager")

        self.assertIn(
            reverse("annotations:pre_annotate", args=[self.project.pk]), html)


class ProjectListTests(ViewAccessTestCase):
    def html(self, who):
        self.as_(who)
        return self.client.get(reverse("annotations:list")).content.decode()

    def test_a_shared_project_says_whose_it_is_and_what_you_are(self):
        html = self.html("annotator")

        self.assertIn("shared by owner", html)
        self.assertIn("annotator", html)

    def test_your_own_project_is_not_labelled_as_shared(self):
        self.assertNotIn("shared by", self.html("owner"))

    def test_a_collaborator_is_not_offered_delete(self):
        """The row must not carry a control that would 404 — or worse, look
        like it might work."""
        delete = reverse("annotations:delete", args=[self.project.pk])

        self.assertNotIn(delete, self.html("reviewer"))
        self.assertIn(delete, self.html("owner"))

    def test_a_manager_may_still_reach_settings(self):
        settings_url = reverse("annotations:settings", args=[self.project.pk])

        self.assertIn(settings_url, self.html("manager"))
        self.assertNotIn(settings_url, self.html("annotator"))


class EditorLandingTests(ViewAccessTestCase):
    """"Annotate" and clip links land on a real frame needing labels."""

    def setUp(self):
        super().setUp()
        from apps.annotations.models import Annotation
        Annotation.objects.filter(project=self.project).delete()
        self.done = Annotation.objects.create(project=self.project, video=self.video,
                                              frame_number=120, boxes=[{"label": "bee"}])
        self.todo = Annotation.objects.create(project=self.project, video=self.video,
                                              frame_number=480, boxes=[], sampled_only=True)
        self.url = reverse("annotations:editor", args=[self.project.pk])

    def test_annotate_goes_to_the_first_frame_needing_labels(self):
        self.as_("owner")
        r = self.client.get(self.url)
        self.assertRedirects(r, f"{self.url}?video={self.video.pk}&frame=480",
                             fetch_redirect_response=False)

    def test_a_clip_link_without_a_sampled_frame_lands_on_one(self):
        self.as_("owner")
        r = self.client.get(f"{self.url}?video={self.video.pk}&frame=0")
        self.assertRedirects(r, f"{self.url}?video={self.video.pk}&frame=480",
                             fetch_redirect_response=False)

    def test_a_real_frame_and_an_explicit_jump_open_as_asked(self):
        self.as_("owner")
        self.assertEqual(self.client.get(f"{self.url}?video={self.video.pk}&frame=120").status_code, 200)
        self.assertEqual(self.client.get(f"{self.url}?video={self.video.pk}&frame=7&jump=1").status_code, 200)
