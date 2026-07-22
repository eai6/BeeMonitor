"""Frame sampling, decoupled from SAM 3.

Two things must hold. First, sampling reaches the editor without touching a GPU —
that is the entire point of splitting it out. Second, the placeholder rows it
creates must not leak into training as blank negative examples, while frames a
human deliberately marked empty still count as the negatives they are.
"""

from unittest.mock import patch

import numpy as np
from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.annotations import sampling
from apps.annotations.models import (
    Annotation, AnnotationProject, FrameSamplingTask, PreAnnotationTask,
)
from apps.videos.models import Video

User = get_user_model()


class FakeS3:
    """Stand-in for the storage client: records uploads, writes a real MP4."""

    def __init__(self, frames=90):
        self.frames = frames
        self.uploads = []

    def download_file(self, container, blob_path, local_path):
        import cv2

        writer = cv2.VideoWriter(
            local_path, cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
        for i in range(self.frames):
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = i % 256
            writer.write(frame)
        writer.release()
        return local_path

    def upload_stream(self, container, blob_path, stream, content_type=None):
        self.uploads.append((container, blob_path))
        return f"{container}/{blob_path}"


class SamplingTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.project = AnnotationProject.objects.create(
            user=self.user, name="P", classes=["bee"],
        )
        self.video = Video.objects.create(
            user=self.user, title="clip", storage_key="alice/clip.mp4",
            file_size_bytes=1, status=Video.Status.READY,
        )
        self.project.videos.add(self.video)

    def _task(self, **params):
        return FrameSamplingTask.objects.create(
            user=self.user, project=self.project, video=self.video,
            params={"sample_interval": 30, "max_frames": 100, **params},
        )

    def _run(self, task, s3=None):
        s3 = s3 or FakeS3()
        with patch("config.storage.get_s3_client", return_value=s3):
            sampling.run_sampling_task(task.pk)
        return s3


class ClampParamsTests(TestCase):
    def test_defaults_when_absent(self):
        self.assertEqual(sampling.clamp_params({}),
                         {"sample_interval": 30, "max_frames": 100})

    def test_values_are_clamped(self):
        got = sampling.clamp_params({"sample_interval": 99999, "max_frames": 0})
        self.assertEqual(got["sample_interval"], sampling.MAX_INTERVAL)
        self.assertEqual(got["max_frames"], sampling.MIN_FRAMES)

    def test_garbage_falls_back_to_defaults(self):
        got = sampling.clamp_params({"sample_interval": "abc", "max_frames": None})
        self.assertEqual(got, {"sample_interval": 30, "max_frames": 100})

    def test_frame_key_matches_the_gpu_workers_convention(self):
        """Diverging here means the editor can't find sampled frames."""
        self.assertEqual(
            sampling.frame_key("users/1/devices/2/clip.mp4", 42),
            "frames/users_1_devices_2_clip.mp4/f000042.jpg",
        )


class SampleFramesTests(SamplingTestCase):
    def test_writes_a_navigable_row_per_sampled_frame(self):
        task = self._task(sample_interval=30, max_frames=10)

        s3 = self._run(task)

        task.refresh_from_db()
        self.assertEqual(task.status, FrameSamplingTask.Status.COMPLETED)
        # 90 frames, every 30th -> frames 0, 30, 60.
        self.assertEqual(task.frames_written, 3)
        rows = Annotation.objects.filter(project=self.project).order_by("frame_number")
        self.assertEqual([r.frame_number for r in rows], [0, 30, 60])
        self.assertTrue(all(r.sampled_only and r.boxes == [] for r in rows))
        self.assertTrue(all(r.frame_image_path for r in rows))
        self.assertEqual(len(s3.uploads), 3)

    def test_max_frames_caps_the_output(self):
        task = self._task(sample_interval=1, max_frames=5)

        self._run(task)

        self.assertEqual(Annotation.objects.filter(project=self.project).count(), 5)

    def test_no_gpu_work_is_created(self):
        """The whole point: frames without a SAM 3 invocation."""
        self._run(self._task())

        self.assertEqual(PreAnnotationTask.objects.count(), 0)

    def test_resampling_is_idempotent(self):
        self._run(self._task())
        self._run(self._task())

        self.assertEqual(Annotation.objects.filter(project=self.project).count(), 3)

    def test_existing_boxes_are_never_clobbered(self):
        Annotation.objects.create(
            project=self.project, video=self.video, frame_number=30,
            boxes=[{"x": 1, "y": 2, "w": 3, "h": 4, "class": "bee", "class_id": 0}],
            reviewed=True, review_source=Annotation.ReviewSource.HUMAN,
        )

        self._run(self._task())

        kept = Annotation.objects.get(project=self.project, frame_number=30)
        self.assertEqual(len(kept.boxes), 1)
        self.assertFalse(kept.sampled_only)
        self.assertTrue(kept.frame_image_path)  # still gained the image

    def test_external_bucket_video_fails_with_a_clear_reason(self):
        self.video.storage_key = "s3://elsewhere/clip.mp4"
        self.video.save(update_fields=["storage_key"])
        task = self._task()

        self._run(task)

        task.refresh_from_db()
        self.assertEqual(task.status, FrameSamplingTask.Status.FAILED)
        self.assertIn("external bucket", task.error_message)

    def test_cancelled_task_is_not_run(self):
        task = self._task()
        FrameSamplingTask.objects.filter(pk=task.pk).update(
            status=FrameSamplingTask.Status.CANCELLED)

        self._run(task)

        self.assertEqual(Annotation.objects.count(), 0)
        task.refresh_from_db()
        self.assertEqual(task.status, FrameSamplingTask.Status.CANCELLED)


class SampleFramesViewTests(SamplingTestCase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)
        self.url = f"/annotations/{self.project.pk}/sample-frames/"

    @patch("apps.annotations.sampling.spawn_sampling_async")
    def test_post_creates_a_queued_task(self, spawn):
        self.client.post(self.url, {"sample_interval": "15", "max_frames": "50"})

        task = FrameSamplingTask.objects.get()
        self.assertEqual(task.status, FrameSamplingTask.Status.QUEUED)
        self.assertEqual(task.params, {"sample_interval": 15, "max_frames": 50})
        spawn.assert_called_once_with(task.pk)

    @patch("apps.annotations.sampling.spawn_sampling_async")
    def test_cancel_marks_tasks_cancelled(self, _spawn):
        self.client.post(self.url, {})
        self.client.post(f"/annotations/{self.project.pk}/sample-frames/cancel/", {})

        self.assertEqual(FrameSamplingTask.objects.get().status,
                         FrameSamplingTask.Status.CANCELLED)

    def test_another_users_project_is_not_reachable(self):
        mallory = User.objects.create_user("mallory", password="x")
        theirs = AnnotationProject.objects.create(user=mallory, name="Theirs")

        resp = self.client.post(f"/annotations/{theirs.pk}/sample-frames/", {})

        self.assertEqual(resp.status_code, 404)
        self.assertEqual(FrameSamplingTask.objects.count(), 0)


class TrainingExclusionTests(SamplingTestCase):
    """The hazard created by reusing Annotation rows for navigation."""

    def _payload_frames(self):
        from apps.training.models import TrainingJob
        from apps.training.views import _build_training_payload

        job = TrainingJob.objects.create(
            user=self.user, project=self.project, name="t",
            frame_filter=TrainingJob.FrameFilter.ALL,
        )
        _classes, video_annotations = _build_training_payload(job)
        return [f for v in video_annotations for f in v["frames"]]

    def test_unannotated_sampled_frames_are_excluded(self):
        self._run(self._task())
        # One frame actually annotated by a human.
        Annotation.objects.filter(frame_number=30).update(
            boxes=[{"x": 1, "y": 2, "w": 3, "h": 4, "class": "bee", "class_id": 0}],
            reviewed=True, review_source=Annotation.ReviewSource.HUMAN,
            sampled_only=False,
        )

        frames = self._payload_frames()

        self.assertEqual([f["frame_number"] for f in frames], [30])

    def test_human_marked_empty_frames_are_kept_as_negatives(self):
        """"Mark empty (no insect)" is a real label, not a placeholder."""
        self._run(self._task())
        Annotation.objects.filter(frame_number=0).update(
            boxes=[], reviewed=True,
            review_source=Annotation.ReviewSource.HUMAN, sampled_only=False,
        )

        frames = self._payload_frames()

        self.assertEqual([f["frame_number"] for f in frames], [0])
        self.assertEqual(frames[0]["label"], "")  # empty label file = negative

    def test_all_sampled_and_nothing_annotated_raises_rather_than_training_on_blanks(self):
        self._run(self._task())

        with self.assertRaises(ValueError):
            self._payload_frames()
