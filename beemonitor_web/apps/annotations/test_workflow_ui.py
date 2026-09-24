"""The project page describes STATE, and acts on a selection.

It used to be four numbered steps — Add videos, Sample frames, Annotate,
Auto-label — which read as a sequence you run once, top to bottom. The real
loop is per clip: add a few, sample those, label those, review those, repeat.
The per-clip selector that made the loop possible sat ~280 lines below the
buttons it governed, joined only by a line of grey text.

So the stages are a filter, the clip list is what you act on, and every action
says what it is about to touch.
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
        return self.client.get(
            reverse("annotations:detail", args=[self.project.pk]),
            params).content.decode()


class StageFunnelTests(ProjectPageTests):
    def test_every_stage_is_named_with_what_is_outstanding(self):
        self.clip()                                   # not sampled
        self.clip(frames=4, labelled=2, reviewed=1)

        html = self.html()

        self.assertIn("Frames sampled", html)
        self.assertIn("Frames labelled", html)
        self.assertIn("Reviewed", html)
        self.assertIn("1 not sampled", html)
        self.assertIn("2 sampled, no boxes", html)

    def test_export_stays_in_the_funnel(self):
        self.clip(frames=2, labelled=2)

        self.assertIn(reverse("annotations:export", args=[self.project.pk]),
                      self.html())

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
    def test_one_form_posts_to_both_destinations(self):
        """The checkboxes used to live in the auto-label form alone, and the
        sample form mirrored the ids across on submit. formaction removes the
        mirroring: one selection, two buttons."""
        self.clip(frames=1)

        html = self.html()

        self.assertIn('id="clip-form"', html)
        self.assertIn(reverse("annotations:sample_frames", args=[self.project.pk]), html)
        self.assertIn(reverse("annotations:pre_annotate_all", args=[self.project.pk]), html)

    def test_the_expensive_action_is_marked_as_such(self):
        self.clip(frames=1)

        self.assertIn("GPU", self.html())

    def test_the_selection_presets_are_offered(self):
        self.clip(frames=1)

        html = self.html()

        for preset in ('data-pick="all"', 'data-pick="new"',
                       'data-pick="sampled"', 'data-pick="none"'):
            self.assertIn(preset, html)

    def test_each_clip_carries_its_stage_for_the_presets_to_match(self):
        self.clip(frames=3, labelled=3, reviewed=3)

        self.assertIn('data-stage="reviewed"', self.html())

    def test_the_project_classes_ride_along_for_auto_labelling(self):
        self.clip(frames=1)

        html = self.html()

        for cls in ("bee", "nest hole", "box"):
            self.assertIn(f'name="labels" value="{cls}"', html)


class ClipStateTests(ProjectPageTests):
    def test_a_clip_is_named_by_the_furthest_stage_it_reached(self):
        self.clip()
        self.clip(frames=2)
        self.clip(frames=2, labelled=2)
        self.clip(frames=2, labelled=2, reviewed=2)

        html = self.html()

        for label in ("Not sampled", "Sampled", "Labelled", "Reviewed"):
            self.assertIn(f">{label}</span>", html)

    def test_a_clip_shows_its_own_counts_not_just_a_total(self):
        self.clip(frames=5, labelled=3)

        self.assertIn("5 frames &middot; 3 labelled", self.html())


class FailurePanelTests(ProjectPageTests):
    """Five identical lines of "Timed out — no result" is not a report."""

    def _failed(self, message="Timed out — no result", n=3):
        from apps.annotations.models import PreAnnotationTask

        tasks = []
        for _ in range(n):
            v = self.clip()
            tasks.append(PreAnnotationTask.objects.create(
                user=self.user, project=self.project, video=v,
                status=PreAnnotationTask.Status.FAILED, error_message=message))
        return tasks

    def test_the_failed_clips_are_named(self):
        tasks = self._failed(n=2)

        html = self.html()

        for t in tasks:
            self.assertIn(t.video.title, html)

    def test_retrying_just_those_clips_is_one_button(self):
        tasks = self._failed(n=2)

        html = self.html()

        self.assertIn("Retry these 2", html)
        for t in tasks:
            self.assertIn(f'name="video_ids" value="{t.video_id}"', html)

    def test_a_shared_cause_is_stated_once(self):
        self._failed(n=3)

        html = self.html()

        self.assertIn("waiting for a GPU slot", html)
        self.assertEqual(html.count("waiting for a GPU slot"), 1)

    def test_mixed_causes_are_not_summarised_into_one(self):
        """Claiming they all timed out when they didn't would send the reader
        after the wrong thing."""
        self._failed(message="Timed out — no result", n=1)
        self._failed(message="could not open video", n=1)

        self.assertNotIn("waiting for a GPU slot", self.html())

    def test_a_failed_clip_is_marked_as_such_in_the_list(self):
        self._failed(n=1)

        self.assertIn('data-stage="failed"', self.html())
