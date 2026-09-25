"""The Clips tab describes STATE, and acts on a selection.

The loop is per clip: add a few, sample those (the GPU labels them in the same
pass), review what came back, repeat. So each clip says where it has got to,
the stages are a filter, and the one action — sampling — says what it is about
to touch. Reviewing happens on the Frames tab (test_page_split).
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations.models import Annotation, AnnotationProject
from apps.videos.models import Video

User = get_user_model()


class ProjectPageTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Summer_2026", classes=["bee", "nest hole", "box"])
        self.client.force_login(self.user)
        self.n = 0

    def clip(self, *, frames=0, labelled=0, reviewed=0):
        self.n += 1
        v = Video.objects.create(user=self.user, title=f"clip{self.n}",
                                 storage_key=f"a/{self.n}.mp4", file_size_bytes=1,
                                 status=Video.Status.READY)
        self.project.videos.add(v)
        for i in range(frames):
            Annotation.objects.create(
                project=self.project, video=v, frame_number=i,
                boxes=[{"x": 1}] if i < labelled else [],
                reviewed=i < reviewed)
        return v

    def html(self, **params):
        params.setdefault("tab", "clips")
        return self.client.get(
            reverse("annotations:detail", args=[self.project.pk]),
            params).content.decode()


class StageFilterTests(ProjectPageTests):
    def test_the_stages_are_offered_with_their_counts(self):
        self.clip()                                   # not sampled
        self.clip(frames=4, labelled=2, reviewed=1)

        html = self.html()

        self.assertIn("Not sampled &middot; 1", html)
        self.assertIn("To review &middot; 1", html)

    def test_the_clips_number_says_what_is_not_sampled(self):
        self.clip()
        self.clip(frames=1, labelled=1)
        self.assertIn("1 not sampled", self.html(tab="frames"))

    def test_a_stage_narrows_the_clip_list(self):
        sampled = self.clip(frames=3)                 # frames, no boxes
        labelled = self.clip(frames=3, labelled=3)

        html = self.html(stage="sampled")

        self.assertIn(f'name="video_ids" value="{sampled.pk}"', html)
        self.assertNotIn(f'name="video_ids" value="{labelled.pk}"', html)

    def test_not_sampled_finds_clips_with_no_frames_at_all(self):
        """These have no annotation rows, so they are invisible to the
        aggregate and have to be found by subtraction."""
        bare = self.clip()
        self.clip(frames=2)

        html = self.html(stage="new")

        self.assertIn(f'name="video_ids" value="{bare.pk}"', html)

    def test_an_unknown_stage_is_ignored_rather_than_emptying_the_page(self):
        v = self.clip(frames=1)

        self.assertIn(f'name="video_ids" value="{v.pk}"', self.html(stage="nonsense"))


class ClipActionTests(ProjectPageTests):
    def test_sampling_posts_the_selection(self):
        self.clip(frames=1)

        html = self.html()

        self.assertIn('id="clip-form"', html)
        self.assertIn(reverse("annotations:sample_frames", args=[self.project.pk]), html)

    def test_the_selection_presets_are_offered(self):
        self.clip(frames=1)

        html = self.html()

        for preset in ('data-pick="all"', 'data-pick="new"',
                       'data-pick="empty"', 'data-pick="none"'):
            self.assertIn(preset, html)

    def test_each_clip_carries_its_stage_for_the_presets_to_match(self):
        self.clip(frames=3, labelled=3, reviewed=3)

        self.assertIn('data-stage="reviewed"', self.html())

    def test_clips_are_no_longer_assigned(self):
        """Work is handed out per frame, on the Frames tab."""
        self.clip(frames=1)
        html = self.html()
        self.assertNotIn("Assigned to", html)
        self.assertNotIn('name="assignee"', html)


class ClipStateTests(ProjectPageTests):
    def test_a_clip_is_named_by_the_furthest_stage_it_reached(self):
        self.clip()
        self.clip(frames=2)
        self.clip(frames=2, labelled=2)
        self.clip(frames=2, labelled=2, reviewed=2)

        html = self.html()

        for label in ("Not sampled", "Sampled", "To review", "Reviewed"):
            self.assertIn(f">{label}</span>", html)

    def test_a_clip_shows_its_own_counts_not_just_a_total(self):
        self.clip(frames=5, labelled=3, reviewed=2)

        self.assertIn("5 frames &middot; 2 reviewed", self.html())


class SamplingFailureTests(ProjectPageTests):
    def _failed(self, v):
        from apps.annotations.models import FrameSamplingTask
        return FrameSamplingTask.objects.create(
            user=self.user, project=self.project, video=v, status=FrameSamplingTask.Status.FAILED,
            error_message="boom")

    def test_a_failed_clip_is_marked_and_filterable(self):
        v = self.clip()
        self._failed(v)

        self.assertIn('data-stage="failed"', self.html())
        self.assertIn(f'name="video_ids" value="{v.pk}"', self.html(stage="failed"))
        self.assertNotIn(f'name="video_ids" value="{v.pk}"', self.html(stage="new"))

    def test_a_later_success_clears_it(self):
        from apps.annotations.models import FrameSamplingTask
        v = self.clip()
        self._failed(v)
        FrameSamplingTask.objects.create(user=self.user, project=self.project, video=v,
                                         status=FrameSamplingTask.Status.COMPLETED)

        self.assertNotIn('data-stage="failed"', self.html())
