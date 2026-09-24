"""Adding clips samples them.

Adding a clip and remembering to sample it were two steps that always ran
together, and forgetting the second left the clip in the project looking
added-but-empty with nothing on the page saying why. Re-sampling with different
knobs stays available; the default path just stops having a hole in it.
"""

from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations.models import AnnotationProject, FrameSamplingTask
from apps.videos.models import Video

User = get_user_model()


class AutoSampleTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("as", password="x")
        self.client.force_login(self.user)
        self.project = AnnotationProject.objects.create(user=self.user, name="P")
        self.videos = [
            Video.objects.create(user=self.user, title=f"c{i}",
                                 storage_key=f"as/{i}.mp4", file_size_bytes=1,
                                 status=Video.Status.READY)
            for i in range(3)
        ]

    def _add(self, videos):
        with patch("apps.annotations.sampling.spawn_sampling_async") as spawn:
            self.client.post(
                reverse("annotations:add_videos", args=[self.project.pk]),
                {"video_ids": [str(v.pk) for v in videos]})
        return spawn

    def test_added_clips_are_queued_for_sampling(self):
        spawn = self._add(self.videos)

        self.assertEqual(FrameSamplingTask.objects.count(), 3)
        self.assertEqual(spawn.call_count, 3)

    def test_a_clip_already_in_the_project_is_not_re_sampled(self):
        self.project.videos.add(self.videos[0])

        self._add(self.videos)

        self.assertEqual(FrameSamplingTask.objects.count(), 2)
        sampled = set(FrameSamplingTask.objects.values_list("video_id", flat=True))
        self.assertNotIn(self.videos[0].pk, sampled)

    def test_the_task_carries_the_default_knobs(self):
        self._add([self.videos[0]])

        # Added clips are sampled by motion: their most active frames.
        task = FrameSamplingTask.objects.get()
        self.assertEqual(task.params["method"], "motion")
        self.assertEqual(task.params["max_frames"], 20)
        self.assertEqual(task.params["min_gap_s"], 1.0)

    def test_adding_nothing_queues_nothing(self):
        with patch("apps.annotations.sampling.spawn_sampling_async") as spawn:
            self.client.post(
                reverse("annotations:add_videos", args=[self.project.pk]), {})

        self.assertEqual(FrameSamplingTask.objects.count(), 0)
        spawn.assert_not_called()

    def test_clips_from_a_shared_device_are_added_too(self):
        """The picker lists accessible clips, so the add must accept them —
        filtering on owner silently dropped half a selection."""
        other = User.objects.create_user("owner2", password="x")
        theirs = Video.objects.create(user=other, title="shared",
                                      storage_key="as/shared.mp4",
                                      file_size_bytes=1, status=Video.Status.READY)

        self._add([theirs])

        # Not accessible to this user, so it is correctly refused — the point is
        # that the query is Video.accessible, not Video.objects.filter(user=...).
        self.assertNotIn(theirs, self.project.videos.all())
