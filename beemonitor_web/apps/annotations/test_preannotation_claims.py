"""Every web process drains and polls pre-annotation: a task is invoked once,
its result recorded (and charged) once, and a claim a crash left behind is
released."""

import json
from datetime import timedelta
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.annotations import views
from apps.annotations.models import Annotation, AnnotationProject, PreAnnotationTask
from apps.videos.models import Video

User = get_user_model()


class PreAnnotationClaimTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("p", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])
        self.video = Video.objects.create(user=self.user, title="c", storage_key="u/c.mp4",
                                          file_size_bytes=1, status=Video.Status.READY)
        self.project.videos.add(self.video)

    def task(self, **kw):
        return PreAnnotationTask.objects.create(user=self.user, project=self.project,
                                                video=self.video, labeler="sam3", **kw)

    def test_two_drains_spawn_a_task_once(self):
        t = self.task()
        with mock.patch.object(views, "spawn_preannotation_async") as spawn:
            views.drain_preannotation_queue()
            views.drain_preannotation_queue()
        spawn.assert_called_once_with(t.pk)
        t.refresh_from_db()
        self.assertEqual((t.status, t.output_uri), ("processing", ""))

    def test_a_claimed_task_is_still_sendable_but_a_cancelled_one_is_not(self):
        self.assertTrue(views._unsent("processing", ""))
        self.assertTrue(views._unsent("queued", ""))
        self.assertFalse(views._unsent("processing", "s3://out/x.out"))
        self.assertFalse(views._unsent("cancelled", ""))

    def test_a_result_is_recorded_and_charged_once(self):
        t = self.task(status="processing", output_uri="s3://out/a.out", started_at=timezone.now())
        body = json.dumps({"frames": [{"frame_number": 5, "boxes": [{"x": 1, "y": 1, "w": 5, "h": 5, "class": "bee"}],
                                       "frame_image_path": "frames/k/f000005.jpg"}],
                           "execution_seconds": 20}).encode()
        s3 = mock.Mock()
        s3.get_object.return_value = {"Body": mock.Mock(read=lambda: body)}
        with mock.patch("apps.accounts.models.UserProfile.charge") as charge:
            self.assertTrue(views.finalize_preannotation_task(t, s3))
            t2 = PreAnnotationTask.objects.get(pk=t.pk)
            t2.status = "processing"          # a second process read the row before the first finished
            self.assertFalse(views.finalize_preannotation_task(t2, s3))
        charge.assert_called_once()
        self.assertEqual(Annotation.objects.count(), 1)

    def test_a_claim_that_never_got_sent_is_released(self):
        t = self.task(status="processing", started_at=timezone.now() - timedelta(minutes=16))
        with mock.patch.object(views, "drain_preannotation_queue"):
            views.poll_preannotation_tasks()
        t.refresh_from_db()
        self.assertEqual(t.status, "queued")
