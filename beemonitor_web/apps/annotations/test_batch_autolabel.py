"""Batch auto-label: N of the unlabelled sampled frames, spread across hotels,
hours and clips, each clip sent exactly its picked frames."""

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.annotations import preannotate_pick
from apps.annotations.models import Annotation, AnnotationProject, PreAnnotationTask
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()


class BatchAutoLabelTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("b", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee", "nest"])
        self.a = Device.objects.create(owner=self.user, name="A", key_hash="a", prefix="a")
        self.b = Device.objects.create(owner=self.user, name="B", key_hash="b", prefix="b")
        self.clips = []
        for i, dev in enumerate((self.a, self.a, self.b)):
            v = Video.objects.create(user=self.user, device=dev, title=f"c{i}", storage_key=f"k{i}",
                                     file_size_bytes=1, status=Video.Status.READY)
            Video.objects.filter(pk=v.pk).update(hour=10 + i % 2)
            self.project.videos.add(v)
            for f in range(0, 200, 10):          # 20 sampled, unlabelled frames each
                Annotation.objects.create(project=self.project, video=v, frame_number=f,
                                          boxes=[], sampled_only=True)
            self.clips.append(v)
        # Touched frames are never picked.
        Annotation.objects.filter(video=self.clips[0], frame_number=0).update(boxes=[{"label": "bee"}])
        Annotation.objects.filter(video=self.clips[0], frame_number=10).update(reviewed=True)
        self.client.force_login(self.user)

    def test_pick_spreads_and_skips_touched_frames(self):
        picks = preannotate_pick.pick(self.project, self.project.videos.all(), 30)
        self.assertEqual(sum(len(f) for f in picks.values()), 30)
        self.assertEqual(set(picks), {v.pk for v in self.clips})
        self.assertNotIn(0, picks[self.clips[0].pk])
        self.assertNotIn(10, picks[self.clips[0].pk])
        frames = picks[self.clips[2].pk]
        self.assertLess(min(frames), 50)
        self.assertGreater(max(frames), 150)

    def test_post_queues_one_task_per_clip_with_exact_frames(self):
        with mock.patch("apps.annotations.views.drain_preannotation_queue", return_value=0):
            r = self.client.post(reverse("annotations:pre_annotate_all", args=[self.project.pk]),
                                 {"a_frames": "12", "target_labels": ["bee"]})
        self.assertEqual(r.status_code, 302)
        tasks = list(PreAnnotationTask.objects.all())
        self.assertEqual(sum(len(t.params["frame_numbers"]) for t in tasks), 12)
        for t in tasks:
            self.assertEqual(t.params["max_frames"], len(t.params["frame_numbers"]))
            self.assertEqual(t.params["target_labels"], ["bee"])
            sampled = set(Annotation.objects.filter(video=t.video).values_list("frame_number", flat=True))
            self.assertTrue(set(t.params["frame_numbers"]) <= sampled)

    def test_the_project_page_offers_the_panel_with_counts(self):
        html = self.client.get(reverse("annotations:detail", args=[self.project.pk])).content.decode()
        self.assertIn('id="auto-panel"', html)
        self.assertIn('data-todo="18"', html)       # clip 0: 20 minus the two touched
        self.assertNotIn(reverse("annotations:pre_annotate", args=[self.project.pk]) + '"', html)
