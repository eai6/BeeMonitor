"""Name a batch; add clips to it (memory/64).

A batch shows its own name where it used to show its pipeline's, and the
launcher can add more clips to it: they run with the batch's own frozen steps
(not the pipeline as edited since) and its part of each clip, and clips
already in it are skipped.
"""

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.pipelines import engine
from apps.pipelines.aggregate import run_video_id
from apps.pipelines.models import Batch, BatchMember, Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()
STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "d", "block_type": "detect.bee", "config": {}, "inputs": {"video": "v"}},
]


class BatchBase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", password="x")
        self.maria = User.objects.create_user("maria", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="Foraging Trips", steps=STEPS)
        self.clips = [self.video(n) for n in range(4)]

    def video(self, n):
        return Video.objects.create(user=self.user, title=f"c{n}", storage_key=f"k/{n}.mp4",
                                    file_size_bytes=1, status=Video.Status.READY,
                                    duration_seconds=600, fps=30)

    def launch(self, ids, **extra):
        with mock.patch("apps.analysis.views._drain_queue"):
            return self.client.post(reverse("pipelines:run_on_videos"),
                                    {"video_ids": ids, **extra},
                                    HTTP_X_REQUESTED_WITH="XMLHttpRequest")

    def first_batch(self, **extra):
        r = self.launch([c.pk for c in self.clips[:2]], pipeline=self.pipeline.pk, **extra)
        self.assertEqual(r.status_code, 200, r.content)
        return PipelineRun.objects.first().batch_id

    def videos_in(self, batch_id):
        return sorted(run_video_id(r) for r in PipelineRun.objects.filter(batch_id=batch_id))


class NameTests(BatchBase):
    def test_a_batch_shows_its_pipelines_name_until_it_is_named(self):
        batch = self.first_batch()
        page = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertEqual(page.context["batch_name"], "Foraging Trips")
        self.client.post(reverse("pipelines:batch_rename", args=[batch]),
                         {"name": "  Foraging Trips ·  May, mendels "})
        page = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertEqual(page.context["batch_name"], "Foraging Trips · May, mendels")
        self.assertContains(page, "Foraging Trips pipeline · batch")
        runs = self.client.get(reverse("pipelines:run_list"))
        self.assertContains(runs, "Foraging Trips · May, mendels")
        # Blank goes back to the pipeline's name.
        self.client.post(reverse("pipelines:batch_rename", args=[batch]), {"name": ""})
        page = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertEqual(page.context["batch_name"], "Foraging Trips")

    def test_named_at_launch(self):
        batch = self.first_batch(batch_name="Night captures")
        self.assertEqual(Batch.objects.get(batch_id=batch).name, "Night captures")

    def test_only_the_launcher_renames(self):
        batch = self.first_batch()
        BatchMember.objects.create(batch_id=batch, user=self.maria)
        self.client.force_login(self.maria)
        r = self.client.post(reverse("pipelines:batch_rename", args=[batch]), {"name": "Mine"})
        self.assertEqual(r.status_code, 404)
        self.assertFalse(Batch.objects.exists())
        page = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertNotContains(page, "data-bn-edit")
        self.assertNotContains(page, "add_to=")


class AddClipsTests(BatchBase):
    def test_added_clips_join_the_batch_and_its_members_see_them(self):
        batch = self.first_batch()
        BatchMember.objects.create(batch_id=batch, user=self.maria)
        # One clip already in it, two new.
        r = self.launch([self.clips[1].pk, self.clips[2].pk, self.clips[3].pk], add_to=str(batch))
        self.assertEqual(r.status_code, 200, r.content)
        self.assertIn("Adding 2 clips", r.json()["message"])
        self.assertIn("1 were already in it", r.json()["message"])
        self.assertEqual(r.json()["batch_url"], reverse("pipelines:batch_detail", args=[batch]))
        self.assertEqual(self.videos_in(batch), sorted(c.pk for c in self.clips))
        self.assertEqual(PipelineRun.objects.values("batch_id").distinct().count(), 1)
        self.client.force_login(self.maria)
        page = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertEqual(len(page.context["rows"]), 4)

    def test_added_clips_use_the_batchs_version_and_part(self):
        batch = self.first_batch(part="part", part_from="2:00", part_minutes="5")
        self.pipeline.steps = STEPS + [{"id": "x", "block_type": "detect.bee", "config": {},
                                        "inputs": {"video": "v"}}]
        self.pipeline.save()
        hub = self.client.get(reverse("analysis:processing"), {"add_to": str(batch)})
        add = hub.context["add_batch"]
        self.assertTrue(add["changed"])
        self.assertEqual(add["video_ids"], {self.clips[0].pk, self.clips[1].pk})
        self.assertContains(hub, "in this batch")
        self.launch([self.clips[2].pk], add_to=str(batch), part="whole")
        run = PipelineRun.objects.filter(batch_id=batch).order_by("-started_at").first()
        self.assertEqual(run_video_id(run), self.clips[2].pk)
        self.assertEqual([s["id"] for s in run.steps], ["v", "d"])
        cfg = run.steps[0]["config"]
        self.assertEqual((cfg["start_seconds"], cfg["end_seconds"]), (120.0, 420.0))

    def test_only_the_launcher_adds_and_nothing_new_is_refused(self):
        batch = self.first_batch()
        self.assertIn("already in this batch",
                      self.launch([self.clips[0].pk], add_to=str(batch)).json()["error"])
        BatchMember.objects.create(batch_id=batch, user=self.maria)
        self.client.force_login(self.maria)
        self.assertEqual(self.launch([self.clips[3].pk], add_to=str(batch)).status_code, 404)
        self.assertIsNone(self.client.get(reverse("analysis:processing"),
                                          {"add_to": str(batch)}).context["add_batch"])
        self.assertEqual(len(self.videos_in(batch)), 2)

    def test_the_template_unbinds_the_clip(self):
        batch = self.first_batch()
        pipeline, steps, window = engine.batch_template(batch)
        self.assertEqual(pipeline, self.pipeline)
        self.assertNotIn("video_id", steps[0]["config"])
        self.assertIsNone(window)
        hub = self.client.get(reverse("analysis:processing"), {"add_to": str(batch)})
        self.assertFalse(hub.context["add_batch"]["changed"])
        self.assertContains(hub, "Add <span id=\"sel-count\">0</span> to the batch")
