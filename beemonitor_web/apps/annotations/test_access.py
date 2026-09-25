"""Who may do what to a shared annotation project.

The failure mode here is silent: a viewer who can delete looks exactly like a
viewer until someone deletes. So the rules are pinned before any view is
converted to use them, and the properties that matter are asserted as
properties — not as "this one view returns 403".
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.annotations.models import Annotation, AnnotationProject, ProjectShare
from apps.videos.models import Video

User = get_user_model()

# No annotator: SAM 3 labels, people review (memory/39).
LEVELS = ("viewer", "reviewer", "manager", "owner")


class AccessTestCase(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.project = AnnotationProject.objects.create(user=self.owner, name="P")
        self.people = {}
        for role in ("viewer", "reviewer", "manager"):
            u = User.objects.create_user(role, password="x")
            ProjectShare.objects.create(project=self.project, user=u, role=role,
                                        created_by=self.owner)
            self.people[role] = u
        self.people["owner"] = self.owner
        self.stranger = User.objects.create_user("stranger", password="x")
        self.n = 0

    def clip(self):
        self.n += 1
        v = Video.objects.create(user=self.owner, title=f"c{self.n}",
                                 storage_key=f"ac/{self.n}.mp4", file_size_bytes=1,
                                 status=Video.Status.READY)
        self.project.videos.add(v)
        return v

    def frame(self, assigned_to=None):
        """A frame SAM 3 labelled, optionally handed to someone to review."""
        return Annotation.objects.create(
            project=self.project, video=self.clip(), frame_number=0,
            boxes=[{"x": 1, "y": 1, "w": 5, "h": 5, "class": "bee"}],
            assigned_to=assigned_to, assigned_by=self.owner if assigned_to else None)


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

    def test_reviewing_excludes_the_viewer(self):
        self.assertEqual(self._who(AnnotationProject.reviewable),
                         {"reviewer", "manager", "owner"})

    def test_spending_gpu_is_manager_and_above(self):
        """A labeller must not be able to spend someone else's GPU budget."""
        self.assertEqual(self._who(AnnotationProject.manageable),
                         {"manager", "owner"})

    def test_deleting_and_sharing_stay_with_the_owner(self):
        self.assertEqual(self._who(AnnotationProject.owned), {"owner"})

    def test_a_stranger_is_in_no_scope(self):
        for qs in (AnnotationProject.accessible,
                   AnnotationProject.reviewable, AnnotationProject.manageable,
                   AnnotationProject.owned):
            self.assertFalse(qs(self.stranger).exists(), qs.__name__)

    def test_a_project_appears_once_however_it_is_reached(self):
        """Owner AND a share would otherwise duplicate the row."""
        ProjectShare.objects.create(project=self.project, user=self.owner,
                                    role="manager")

        self.assertEqual(AnnotationProject.accessible(self.owner).count(), 1)


class FrameEditTests(AccessTestCase):
    """Who may save a frame. Assignment is per frame: a work queue that also
    keeps two reviewers off the same boxes."""

    def test_a_reviewer_may_fix_their_own_frame(self):
        f = self.frame(assigned_to=self.people["reviewer"])
        self.assertTrue(self.project.may_edit_frame(self.people["reviewer"], f))

    def test_a_reviewer_may_fix_a_frame_nobody_holds(self):
        self.assertTrue(self.project.may_edit_frame(self.people["reviewer"], self.frame()))

    def test_a_reviewer_may_not_fix_someone_elses(self):
        other = User.objects.create_user("other", password="x")
        ProjectShare.objects.create(project=self.project, user=other, role="reviewer")
        f = self.frame(assigned_to=other)
        self.assertFalse(self.project.may_edit_frame(self.people["reviewer"], f))

    def test_a_manager_may_fix_any_frame(self):
        f = self.frame(assigned_to=self.people["reviewer"])
        self.assertTrue(self.project.may_edit_frame(self.people["manager"], f))

    def test_a_reviewer_may_mark_a_frame_nobody_sampled(self):
        """"Go to frame" reaches frames with no row yet (a negative example)."""
        self.assertTrue(self.project.may_edit_frame(self.people["reviewer"], None))

    def test_a_viewer_may_never_edit_even_if_assigned(self):
        """An assignment is a work queue, not a grant of permission."""
        f = self.frame(assigned_to=self.people["viewer"])
        self.assertFalse(self.project.may_edit_frame(self.people["viewer"], f))


class FootageIsolationTests(AccessTestCase):
    """Sharing a project must not widen access to the source clips.

    Annotating reads the sampled frames out of the processed bucket; the editor
    never touches raw-videos. So inviting someone to label does not hand them a
    field site's recording times or locations — and the obvious implementation
    of sharing, handing over Video.accessible, throws that away.
    """

    def test_a_collaborator_gains_no_video_access(self):
        self.frame(assigned_to=self.people["reviewer"])

        for role in ("viewer", "reviewer", "manager"):
            self.assertFalse(
                Video.accessible(self.people[role]).exists(),
                f"{role} should not reach the footage through a project share")

    def test_the_owner_still_reaches_their_own_footage(self):
        self.clip()

        self.assertTrue(Video.accessible(self.owner).exists())
