"""Publishing a pipeline (memory/59): others view, run and copy it, never edit it."""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.pipelines import publishing
from apps.pipelines.models import Pipeline, PipelineRun
from apps.training.models import CustomModel

User = get_user_model()

STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "t", "block_type": "track.bee", "config": {"confidence": 0.4}, "inputs": {"video": "v"}},
    {"id": "o", "block_type": "output.table", "config": {}, "inputs": {"in": "t"}},
]


class PublishingTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user("owner", password="x")
        self.other = User.objects.create_user("other", password="x")
        self.pipeline = Pipeline.objects.create(user=self.owner, title="Visits per hour",
                                                description="Counts visits.", steps=STEPS)

    def publish(self, as_user=None, visibility="public"):
        self.client.force_login(as_user or self.owner)
        return self.client.post(reverse("pipelines:publish", args=[self.pipeline.pk]),
                                {"visibility": visibility})

    def test_publishing_lists_it_on_browse(self):
        self.publish()
        self.pipeline.refresh_from_db()
        self.assertTrue(self.pipeline.is_public)
        self.assertIsNotNone(self.pipeline.published_at)
        self.client.force_login(self.other)
        html = self.client.get(reverse("annotations:browse"), {"tab": "pipelines"}).content.decode()
        self.assertIn("Visits per hour", html)
        self.assertIn("Copy to my pipelines", html)

    def test_only_the_owner_publishes(self):
        r = self.publish(as_user=self.other)
        self.assertEqual(r.status_code, 404)
        self.pipeline.refresh_from_db()
        self.assertFalse(self.pipeline.is_public)

    def test_a_private_model_blocks_publishing_until_it_is_public(self):
        model = CustomModel.objects.create(
            user=self.owner, name="Nest holes v3", model_type="bee_tracking",
            base_model="yolov8n", storage_key="m/v3.pt", classes=["bee"],
            status=CustomModel.Status.READY, is_active=True)
        self.pipeline.steps = [dict(STEPS[0]), {**STEPS[1], "config": {"bee_model": str(model.pk)}},
                               dict(STEPS[2])]
        self.pipeline.save()

        r = self.publish()
        self.assertRedirects(r, reverse("pipelines:editor", args=[self.pipeline.pk]) + "?publish=blocked",
                             fetch_redirect_response=False)
        self.pipeline.refresh_from_db()
        self.assertFalse(self.pipeline.is_public)
        html = self.client.get(r["Location"]).content.decode()
        self.assertIn("Not published.", html)
        self.assertIn("Nest holes v3", html)

        model.visibility = "public"
        model.save()
        self.publish()
        self.pipeline.refresh_from_db()
        self.assertTrue(self.pipeline.is_public)
        self.assertEqual(publishing.summary(self.pipeline)["models"], ["Nest holes v3"])

    def test_others_see_it_read_only(self):
        self.publish()
        self.client.force_login(self.other)
        r = self.client.get(reverse("pipelines:editor", args=[self.pipeline.pk]))
        self.assertEqual(r.status_code, 200)
        html = r.content.decode()
        self.assertIn("View only.", html)
        self.assertIn("const READONLY = true", html)
        self.assertNotIn('onclick="plSave()"', html)
        self.assertNotIn(reverse("pipelines:publish", args=[self.pipeline.pk]), html)

    def test_others_cannot_save_rename_or_delete_it(self):
        self.publish()
        self.client.force_login(self.other)
        pk = self.pipeline.pk
        self.assertEqual(self.client.post(reverse("pipelines:save_graph", args=[pk]), "{}",
                                          content_type="application/json").status_code, 404)
        self.assertEqual(self.client.post(reverse("pipelines:rename", args=[pk]),
                                          {"title": "mine now"}).status_code, 404)
        self.assertEqual(self.client.post(reverse("pipelines:delete", args=[pk])).status_code, 404)
        self.pipeline.refresh_from_db()
        self.assertEqual(self.pipeline.title, "Visits per hour")

    def test_others_can_copy_it(self):
        self.publish()
        self.client.force_login(self.other)
        self.client.post(reverse("pipelines:clone", args=[self.pipeline.pk]))
        copy = Pipeline.objects.get(user=self.other)
        self.assertEqual(copy.cloned_from, self.pipeline)
        self.assertEqual([s["block_type"] for s in copy.steps], [s["block_type"] for s in STEPS])
        self.assertEqual(publishing.summary(self.pipeline)["copies"], 1)

    def test_a_private_pipeline_is_none_of_this_to_others(self):
        self.client.force_login(self.other)
        pk = self.pipeline.pk
        self.assertEqual(self.client.get(reverse("pipelines:editor", args=[pk])).status_code, 404)
        self.assertEqual(self.client.post(reverse("pipelines:clone", args=[pk])).status_code, 404)
        self.assertFalse(self.pipeline.usable_by(self.other))

    def test_others_can_pick_it_to_run(self):
        self.publish()
        self.pipeline.refresh_from_db()
        self.assertTrue(self.pipeline.usable_by(self.other))
        self.client.force_login(self.other)
        r = self.client.get(reverse("analysis:processing"))
        self.assertIn(self.pipeline, r.context["pipeline_published"])

    def test_unpublishing_keeps_copies_and_stops_new_use(self):
        self.publish()
        self.client.force_login(self.other)
        self.client.post(reverse("pipelines:clone", args=[self.pipeline.pk]))
        self.publish(visibility="private")
        self.pipeline.refresh_from_db()
        self.assertFalse(self.pipeline.is_public)
        self.assertFalse(self.pipeline.usable_by(self.other))
        self.assertTrue(Pipeline.objects.filter(user=self.other, cloned_from=self.pipeline).exists())
        self.assertNotIn(self.pipeline, Pipeline.public())

    def test_deleting_is_refused_while_others_have_runs_on_it(self):
        PipelineRun.objects.create(pipeline=self.pipeline, user=self.other)
        self.client.force_login(self.owner)
        self.client.post(reverse("pipelines:delete", args=[self.pipeline.pk]))
        self.assertTrue(Pipeline.objects.filter(pk=self.pipeline.pk).exists())

    def test_a_template_does_not_open_in_someone_elses_editor(self):
        self.pipeline.is_template = True
        self.pipeline.save()
        self.client.force_login(self.other)
        r = self.client.get(reverse("pipelines:editor", args=[self.pipeline.pk]))
        self.assertEqual(r.status_code, 404)
