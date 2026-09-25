"""The rules, enforced by the views that matter.

test_access pins the rules as properties. These pin that the views actually
consult them — the conversion across ~20 call sites is where a mistake would
live, and its failure mode is silent.
"""

import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations.models import Annotation, AnnotationProject, ProjectShare
from apps.videos.models import Video

User = get_user_model()


class ViewAccessTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.project = AnnotationProject.objects.create(
            user=self.owner, name="P", classes=["bee"])
        self.people = {"owner": self.owner}
        for role in ("viewer", "reviewer", "manager"):
            u = User.objects.create_user(role, password="x")
            ProjectShare.objects.create(project=self.project, user=u, role=role)
            self.people[role] = u
        # A second reviewer, to hold frames the first may not touch.
        self.people["reviewer2"] = u2 = User.objects.create_user("reviewer2", password="x")
        ProjectShare.objects.create(project=self.project, user=u2, role="reviewer")
        self.stranger = User.objects.create_user("stranger", password="x")
        self.video = Video.objects.create(
            user=self.owner, title="c", storage_key="va/c.mp4",
            file_size_bytes=1, status=Video.Status.READY)
        self.project.videos.add(self.video)
        self.frame = Annotation.objects.create(project=self.project, video=self.video,
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
        self.as_("reviewer")

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
        for role in ("viewer", "reviewer", "manager"):
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

    def hold(self, who):
        self.frame.assigned_to = self.people[who]
        self.frame.save()

    def test_a_reviewer_can_fix_an_unassigned_frame(self):
        self.assertEqual(self.save_as("reviewer"), 200)

    def test_a_reviewer_can_fix_their_own_frame(self):
        self.hold("reviewer")
        self.assertEqual(self.save_as("reviewer"), 200)

    def test_a_reviewer_cannot_fix_someone_elses_frame(self):
        self.hold("reviewer2")
        self.assertEqual(self.save_as("reviewer"), 403)

    def test_a_manager_can_fix_anyones_frame(self):
        self.hold("reviewer2")
        self.assertEqual(self.save_as("manager"), 200)

    def test_saving_records_who_reviewed(self):
        self.save_as("reviewer")
        self.frame.refresh_from_db()
        self.assertEqual((self.frame.reviewed, self.frame.reviewed_by),
                         (True, self.people["reviewer"]))

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

        self.assertEqual(self.status("reviewer", url, "post"), 404)
        self.assertIn(self.status("manager", url, "post"), (200, 302))

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

        form = TrainingCreateForm(user=self.people["reviewer"])

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
                         {"who": "stranger", "role": "reviewer"})

        self.assertEqual(self.project.role_for(self.stranger), "reviewer")

    def test_the_default_role_is_reviewer(self):
        self.as_("owner")
        self.client.post(reverse("annotations:share_invite", args=[self.project.pk]),
                         {"who": "stranger"})
        self.assertEqual(self.project.role_for(self.stranger), "reviewer")

    def test_inviting_an_unknown_account_says_so_rather_than_failing_quietly(self):
        self.as_("owner")

        resp = self.client.post(
            reverse("annotations:share_invite", args=[self.project.pk]),
            {"who": "nobody@example.com"}, follow=True)

        self.assertIn("No account matches", " ".join(
            str(m) for m in resp.context["messages"]))

    def test_removing_someone_returns_their_frames_to_the_pool(self):
        """The work is the project's, not theirs — it must not vanish with them."""
        self.frame.assigned_to = self.people["reviewer"]
        self.frame.save()
        share = self.project.shares.get(user=self.people["reviewer"])
        self.as_("owner")

        self.client.post(reverse("annotations:share_update", args=[self.project.pk]),
                         {"share_id": share.pk, "remove": "1"})

        self.assertFalse(self.project.shares.filter(pk=share.pk).exists())
        self.frame.refresh_from_db()
        self.assertIsNone(self.frame.assigned_to)


class ReviewPageTests(ViewAccessTestCase):
    """The project page: three numbers, frames to review, and a reviewer's queue."""

    def html(self, who, **params):
        self.as_(who)
        return self.client.get(
            reverse("annotations:detail", args=[self.project.pk]),
            params).content.decode()

    def test_a_reviewer_sees_their_queue(self):
        self.frame.boxes = [{"x": 1, "y": 1, "w": 2, "h": 2, "class": "bee"}]
        self.frame.assigned_to = self.people["reviewer"]
        self.frame.save()

        html = self.html("reviewer")

        self.assertIn("Your review queue", html)
        self.assertIn("Continue reviewing", html)
        self.assertIn("who=me&status=review", html)

    def test_a_reviewer_with_nothing_assigned_can_take_frames(self):
        html = self.html("reviewer")
        self.assertIn("Take 100 frames", html)
        self.assertIn(reverse("annotations:take_frames", args=[self.project.pk]), html)

    def test_the_owner_does_not_get_the_queue_unless_assigned(self):
        self.assertNotIn("Your review queue", self.html("owner"))

    def test_a_viewer_is_not_offered_work(self):
        html = self.html("viewer")
        self.assertNotIn("Your review queue", html)
        self.assertNotIn("Review these", html)

    def test_only_a_manager_gets_assign_and_sampling(self):
        assign = reverse("annotations:assign_frames", args=[self.project.pk])
        sample = reverse("annotations:sample_frames", args=[self.project.pk])
        self.assertNotIn(assign, self.html("reviewer"))
        self.assertNotIn(sample, self.html("reviewer", tab="clips"))
        self.assertIn(assign, self.html("manager"))
        self.assertIn(sample, self.html("manager", tab="clips"))

    def test_the_grid_filters_to_one_person(self):
        other = Video.objects.create(user=self.owner, title="o", storage_key="va/o.mp4",
                                     file_size_bytes=1, status=Video.Status.READY)
        self.project.videos.add(other)
        Annotation.objects.create(project=self.project, video=other, frame_number=5,
                                  boxes=[], assigned_to=self.people["reviewer2"])
        self.as_("owner")
        url = reverse("annotations:detail", args=[self.project.pk])

        cards = self.client.get(url, {"who": self.people["reviewer2"].pk}).context["frame_cards"]
        self.assertEqual([c["video_pk"] for c in cards], [other.pk])
        cards = self.client.get(url, {"who": "none"}).context["frame_cards"]
        self.assertEqual([c["video_pk"] for c in cards], [self.video.pk])

    def test_only_managers_see_where_and_when(self):
        """Sharing a project does not share the footage (people.html)."""
        from datetime import datetime, timezone as dt_tz
        self.video.recorded_at = datetime(2026, 7, 14, 15, 41, tzinfo=dt_tz.utc)
        self.video.save()
        self.assertIn("14 Jul", self.html("manager"))
        self.assertNotIn("14 Jul", self.html("reviewer"))
        self.assertNotIn('name="device"', self.html("reviewer"))

    def test_the_old_review_url_lands_on_the_frames_tab(self):
        self.as_("reviewer")
        r = self.client.get(reverse("annotations:review", args=[self.project.pk]),
                            {"status": "reviewed"})
        self.assertRedirects(r, reverse("annotations:detail", args=[self.project.pk])
                             + "?status=reviewed", fetch_redirect_response=False)


class ProjectListTests(ViewAccessTestCase):
    def html(self, who):
        self.as_(who)
        return self.client.get(reverse("annotations:list")).content.decode()

    def test_a_shared_project_says_whose_it_is_and_what_you_are(self):
        html = self.html("reviewer")

        self.assertIn("shared by owner", html)
        self.assertIn("reviewer", html)

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
        self.assertNotIn(settings_url, self.html("reviewer"))


class EditorLandingTests(ViewAccessTestCase):
    """The editor lands on a real frame still to review."""

    def setUp(self):
        super().setUp()
        from apps.annotations.models import Annotation
        Annotation.objects.filter(project=self.project).delete()
        self.done = Annotation.objects.create(project=self.project, video=self.video,
                                              frame_number=120, boxes=[{"label": "bee"}],
                                              reviewed=True)
        self.todo = Annotation.objects.create(project=self.project, video=self.video,
                                              frame_number=480, boxes=[], sampled_only=True)
        self.url = reverse("annotations:editor", args=[self.project.pk])

    def test_the_editor_opens_on_the_first_frame_to_review(self):
        self.as_("owner")
        r = self.client.get(self.url)
        self.assertRedirects(r, f"{self.url}?video={self.video.pk}&frame=480",
                             fetch_redirect_response=False)

    def test_a_clip_link_without_a_sampled_frame_lands_on_one(self):
        self.as_("owner")
        r = self.client.get(f"{self.url}?video={self.video.pk}&frame=0")
        self.assertRedirects(r, f"{self.url}?video={self.video.pk}&frame=480",
                             fetch_redirect_response=False)

    def test_a_real_frame_opens_as_asked(self):
        self.as_("owner")
        self.assertEqual(self.client.get(f"{self.url}?video={self.video.pk}&frame=120").status_code, 200)

    def test_your_own_queue_comes_first(self):
        mine = Annotation.objects.create(project=self.project, video=self.video,
                                         frame_number=900, boxes=[],
                                         assigned_to=self.people["reviewer"])
        self.as_("reviewer")
        r = self.client.get(self.url)
        self.assertRedirects(r, f"{self.url}?video={self.video.pk}&frame={mine.frame_number}",
                             fetch_redirect_response=False)

    def test_prev_next_stay_inside_the_grids_filter(self):
        """Opened from the grid, the editor walks the grid's frames and keeps
        the filter on its links; a reviewed frame is out of "to review"."""
        Annotation.objects.create(project=self.project, video=self.video,
                                  frame_number=600, boxes=[])
        self.as_("owner")
        r = self.client.get(f"{self.url}?video={self.video.pk}&frame=480&status=review")
        self.assertEqual(r.context["prev_frame_url"], "")
        self.assertTrue(r.context["next_frame_url"].endswith(
            f"?video={self.video.pk}&frame=600&status=review"))
        self.assertEqual((r.context["current_frame_index"], r.context["total_project_frames"]),
                         (1, 2))
