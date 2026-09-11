"""Who may do what to a shared annotation project.

The failure mode here is silent: a viewer who can delete looks exactly like a
viewer until someone deletes. So the rules are pinned before any view is
converted to use them, and the properties that matter are asserted as
properties — not as "this one view returns 403".
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.annotations.models import (AnnotationProject, ClipAssignment,
                                     ProjectShare)
from apps.videos.models import Video

User = get_user_model()

LEVELS = ("viewer", "annotator", "reviewer", "manager", "owner")


class AccessTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.project = AnnotationProject.objects.create(user=self.owner, name="P")
        self.people = {}
        for role in ("viewer", "annotator", "reviewer", "manager"):
            u = User.objects.create_user(role, password="x")
            ProjectShare.objects.create(project=self.project, user=u, role=role,
                                        created_by=self.owner)
            self.people[role] = u
        self.people["owner"] = self.owner
        self.stranger = User.objects.create_user("stranger", password="x")
        self.n = 0

    def clip(self, assigned_to=None):
        self.n += 1
        v = Video.objects.create(user=self.owner, title=f"c{self.n}",
                                 storage_key=f"ac/{self.n}.mp4", file_size_bytes=1,
                                 status=Video.Status.READY)
        self.project.videos.add(v)
        if assigned_to:
            ClipAssignment.objects.create(project=self.project, video=v,
                                          user=assigned_to, assigned_by=self.owner)
        return v


class RoleLadderTests(AccessTestCase):
    def test_each_role_is_recognised(self):
        for role, user in self.people.items():
            self.assertEqual(self.project.role_for(user), role)

    def test_a_stranger_has_no_role(self):
        self.assertIsNone(self.project.role_for(self.stranger))

    def test_the_ladder_is_linear(self):
        """Every role permits everything the roles below it permit. This is the
        property that keeps a permission check readable — 'at least this' rather
        than a set of special cases."""
        for i, level in enumerate(LEVELS):
            for role in LEVELS[i:]:
                self.assertTrue(self.project.allows(self.people[role], level),
                                f"{role} should satisfy {level}")
            for role in LEVELS[:i]:
                self.assertFalse(self.project.allows(self.people[role], level),
                                 f"{role} should NOT satisfy {level}")

    def test_the_owner_outranks_any_share(self):
        ProjectShare.objects.create(project=self.project, user=self.stranger,
                                    role="viewer")
        self.project.shares.filter(user=self.owner).delete()

        self.assertEqual(self.project.role_for(self.owner), "owner")

    def test_an_anonymous_visitor_has_no_role(self):
        from django.contrib.auth.models import AnonymousUser

        self.assertIsNone(self.project.role_for(AnonymousUser()))
        self.assertFalse(self.project.allows(None, "viewer"))


class ScopeQuerysetTests(AccessTestCase):
    def _who(self, qs):
        return {u.username for u in self.people.values()
                if qs(u).filter(pk=self.project.pk).exists()}

    def test_read_scope_includes_every_role(self):
        self.assertEqual(self._who(AnnotationProject.accessible),
                         set(self.people))

    def test_drawing_excludes_the_viewer(self):
        self.assertEqual(self._who(AnnotationProject.annotatable),
                         {"annotator", "reviewer", "manager", "owner"})

    def test_signing_off_excludes_the_annotator(self):
        self.assertEqual(self._who(AnnotationProject.reviewable),
                         {"reviewer", "manager", "owner"})

    def test_spending_gpu_is_manager_and_above(self):
        """A labeller must not be able to spend someone else's GPU budget."""
        self.assertEqual(self._who(AnnotationProject.manageable),
                         {"manager", "owner"})

    def test_deleting_and_sharing_stay_with_the_owner(self):
        self.assertEqual(self._who(AnnotationProject.owned), {"owner"})

    def test_a_stranger_is_in_no_scope(self):
        for qs in (AnnotationProject.accessible, AnnotationProject.annotatable,
                   AnnotationProject.reviewable, AnnotationProject.manageable,
                   AnnotationProject.owned):
            self.assertFalse(qs(self.stranger).exists(), qs.__name__)

    def test_a_project_appears_once_however_it_is_reached(self):
        """Owner AND a share would otherwise duplicate the row."""
        ProjectShare.objects.create(project=self.project, user=self.owner,
                                    role="manager")

        self.assertEqual(AnnotationProject.accessible(self.owner).count(), 1)


class AssignmentTests(AccessTestCase):
    def test_an_annotator_may_draw_on_their_own_clip(self):
        v = self.clip(assigned_to=self.people["annotator"])

        self.assertTrue(self.project.may_annotate_video(
            self.people["annotator"], v.pk))

    def test_an_annotator_may_not_draw_on_someone_elses(self):
        v = self.clip(assigned_to=self.people["reviewer"])

        self.assertFalse(self.project.may_annotate_video(
            self.people["annotator"], v.pk))

    def test_an_annotator_may_not_draw_on_an_unassigned_clip(self):
        """They may claim it first — which writes an assignment, so no work
        happens off the books."""
        v = self.clip()

        self.assertFalse(self.project.may_annotate_video(
            self.people["annotator"], v.pk))

    def test_claiming_makes_it_theirs(self):
        v = self.clip()
        ClipAssignment.objects.create(project=self.project, video=v,
                                      user=self.people["annotator"])

        self.assertTrue(self.project.may_annotate_video(
            self.people["annotator"], v.pk))

    def test_a_reviewer_is_not_confined_to_their_assignments(self):
        """Checking other people's work is the job."""
        v = self.clip(assigned_to=self.people["annotator"])

        self.assertTrue(self.project.may_annotate_video(
            self.people["reviewer"], v.pk))

    def test_a_viewer_may_never_draw_even_if_assigned(self):
        """An assignment is a work queue, not a grant of permission."""
        v = self.clip(assigned_to=self.people["viewer"])

        self.assertFalse(self.project.may_annotate_video(
            self.people["viewer"], v.pk))

    def test_a_clip_has_at_most_one_assignee(self):
        from django.db.utils import IntegrityError

        v = self.clip(assigned_to=self.people["annotator"])

        with self.assertRaises(IntegrityError):
            ClipAssignment.objects.create(project=self.project, video=v,
                                          user=self.people["reviewer"])

    def test_a_persons_workload_is_their_assignments(self):
        self.clip(assigned_to=self.people["annotator"])
        self.clip(assigned_to=self.people["annotator"])
        self.clip(assigned_to=self.people["reviewer"])
        self.clip()

        self.assertEqual(
            self.project.assigned_to(self.people["annotator"]).count(), 2)

    def test_a_self_claim_is_distinguishable_from_being_given_work(self):
        v = self.clip()
        claimed = ClipAssignment.objects.create(
            project=self.project, video=v, user=self.people["annotator"])
        given = ClipAssignment.objects.get(video=self.clip(
            assigned_to=self.people["annotator"]))

        self.assertTrue(claimed.self_claimed)
        self.assertFalse(given.self_claimed)


class FootageIsolationTests(AccessTestCase):
    """Sharing a project must not widen access to the source clips.

    Annotating reads the sampled frames out of the processed bucket; the editor
    never touches raw-videos. So inviting someone to label does not hand them a
    field site's recording times or locations — and the obvious implementation
    of sharing, handing over Video.accessible, throws that away.
    """

    def test_a_collaborator_gains_no_video_access(self):
        self.clip(assigned_to=self.people["annotator"])

        for role in ("viewer", "annotator", "reviewer", "manager"):
            self.assertFalse(
                Video.accessible(self.people[role]).exists(),
                f"{role} should not reach the footage through a project share")

    def test_the_owner_still_reaches_their_own_footage(self):
        self.clip()

        self.assertTrue(Video.accessible(self.owner).exists())
