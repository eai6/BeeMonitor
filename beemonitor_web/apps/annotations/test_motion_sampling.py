"""Frame sampling by motion: the most active frames are picked, spaced apart,
inside the ROI; a clip with no motion yields no frames; re-sampling replaces
only frames nobody has touched."""

import os
import shutil
import tempfile
from unittest import mock

import cv2
import numpy as np
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations import sampling
from apps.annotations.models import Annotation, AnnotationProject, FrameSamplingTask
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()
FPS = 10


def write_clip(path, n_frames=100, moving=range(40, 60), where=(0.6, 0.6)):
    """A textured still background; a dark square moves only during ``moving``,
    near ``where`` (normalized x, y)."""
    rng = np.random.default_rng(0)
    bg = rng.integers(90, 160, (240, 320, 3), dtype=np.uint8)
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (320, 240))
    for i in range(n_frames):
        frame = bg.copy()
        if i in moving:
            x = int(where[0] * 320) + (i - moving.start) * 2
            y = int(where[1] * 240)
            cv2.rectangle(frame, (x, y), (x + 14, y + 14), (10, 10, 10), -1)
        out.write(frame)
    out.release()


class ScoringTests(TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_the_busiest_frames_are_picked_and_spaced(self):
        path = os.path.join(self.dir, "c.avi")
        write_clip(path)
        scores, fps = sampling.score_motion(path)
        picks = sampling.pick_active_frames(scores, 5, min_gap_frames=int(fps * 0.5))
        self.assertEqual(len(picks), 4)          # 20 moving frames, 5 apart
        self.assertTrue(all(40 <= p < 62 for p in picks), picks)
        self.assertTrue(all(b - a >= 5 for a, b in zip(picks, picks[1:])))

    def test_a_still_clip_gives_nothing(self):
        path = os.path.join(self.dir, "still.avi")
        write_clip(path, moving=range(0))
        scores, _ = sampling.score_motion(path)
        self.assertEqual(sampling.pick_active_frames(scores, 20, 5), [])

    def test_motion_outside_the_roi_is_ignored(self):
        path = os.path.join(self.dir, "c.avi")
        write_clip(path, where=(0.6, 0.6))
        scores, _ = sampling.score_motion(path, roi=[0.0, 0.0, 0.4, 0.4])
        self.assertEqual(sampling.pick_active_frames(scores, 20, 1), [])

    def test_profile_marks_the_picked_stretch(self):
        prof = sampling.motion_profile([0.0] * 50 + [0.5] * 10 + [0.0] * 40, [55], buckets=10)
        self.assertEqual(prof["frames"], 100)
        self.assertEqual(prof["picked"], [5])
        self.assertEqual(max(prof["profile"]), 100)


class _FakeS3:
    def __init__(self, src):
        self.src, self.uploaded = src, []

    def download_file(self, bucket, key, dest):
        shutil.copy(self.src, dest)

    def upload_stream(self, bucket, key, stream, content_type=None):
        self.uploaded.append(key)


class TaskTests(TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.clip_path = os.path.join(self.dir, "c.avi")
        write_clip(self.clip_path)
        self.user = User.objects.create_user("m", password="x")
        self.dev = Device.objects.create(owner=self.user, name="Jill", key_hash="h", prefix="p")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])
        self.video = Video.objects.create(user=self.user, device=self.dev, title="c",
                                          storage_key="u/c.mp4", file_size_bytes=1,
                                          status=Video.Status.READY)
        self.project.videos.add(self.video)

    def run_task(self, **params):
        task = FrameSamplingTask.objects.create(user=self.user, project=self.project,
                                                video=self.video,
                                                params=sampling.motion_params(**params))
        fake = _FakeS3(self.clip_path)
        with mock.patch("config.storage.get_s3_client", return_value=fake):
            sampling.run_sampling_task(task.pk)
        task.refresh_from_db()
        return task, fake

    def test_writes_only_active_frames_and_the_strip(self):
        task, fake = self.run_task(max_frames=3, min_gap_s=0.5)
        self.assertEqual(task.status, FrameSamplingTask.Status.COMPLETED, task.error_message)
        frames = list(Annotation.objects.filter(project=self.project).values_list("frame_number", flat=True))
        self.assertEqual(len(frames), 3)
        self.assertTrue(all(40 <= f < 62 for f in frames), frames)
        self.assertEqual(len(fake.uploaded), 3)
        self.assertTrue(task.motion["picked"])

    def test_replace_keeps_touched_frames_and_drops_untouched_ones(self):
        untouched = Annotation.objects.create(project=self.project, video=self.video, frame_number=3,
                                              boxes=[], sampled_only=True)
        labelled = Annotation.objects.create(project=self.project, video=self.video, frame_number=5,
                                             boxes=[{"label": "bee"}], sampled_only=True)
        reviewed = Annotation.objects.create(project=self.project, video=self.video, frame_number=7,
                                             boxes=[], sampled_only=True, reviewed=True)
        self.run_task(max_frames=2)
        left = set(Annotation.objects.filter(project=self.project).values_list("pk", flat=True))
        self.assertNotIn(untouched.pk, left)
        self.assertIn(labelled.pk, left)
        self.assertIn(reviewed.pk, left)

    def test_added_clips_default_to_motion(self):
        with mock.patch("apps.annotations.sampling.spawn_sampling_async"):
            other = Video.objects.create(user=self.user, device=self.dev, title="d",
                                         storage_key="u/d.mp4", file_size_bytes=1,
                                         status=Video.Status.READY)
            self.client.force_login(self.user)
            self.client.post(reverse("annotations:add_videos", args=[self.project.pk]),
                             {"video_ids": [other.pk]})
        params = FrameSamplingTask.objects.get(video=other).params
        self.assertEqual((params["method"], params["max_frames"]), ("motion", 20))

    def test_the_panel_fields_drive_a_resample(self):
        self.client.force_login(self.user)
        with mock.patch("apps.annotations.sampling.spawn_sampling_async"):
            self.client.post(reverse("annotations:sample_frames", args=[self.project.pk]), {
                "video_ids": [self.video.pk], "max_frames": "100",   # auto-label's field
                "s_method": "motion", "s_max_frames": "10", "s_min_gap_s": "2",
                "s_roi": "frame", "s_replace": ["0", "1"]})
        params = FrameSamplingTask.objects.latest("created_at").params
        self.assertEqual((params["method"], params["max_frames"], params["min_gap_s"], params["roi"],
                          params["replace"]), ("motion", 10, 2.0, "frame", True))

    def test_old_tasks_without_a_method_still_sample_evenly(self):
        self.assertEqual(sampling.clamp_params({"sample_interval": 15})["method"], "interval")
