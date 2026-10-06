"""Pipelines on photos (memory/45)."""

import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.pipelines import aggregate, executors
from apps.pipelines.engine import steps_with_video_steps
from apps.pipelines.models import Pipeline, PipelineRun
from apps.pipelines.registry import pipeline_input_kind, validate_steps
from apps.videos.models import Video

User = get_user_model()

PHOTO = {"width": 4000, "height": 3000, "tiles": 12, "preview_key": "1/j/photo_preview.jpg",
         "detections": [
             {"id": 1, "x": 100, "y": 100, "w": 80, "h": 60, "class": "insect", "confidence": 0.8,
              "crop_key": "1/j/photo_crops/0001.jpg", "species": "Bombus impatiens",
              "species_confidence": 0.71},
             {"id": 2, "x": 3000, "y": 2000, "w": 60, "h": 40, "class": "insect", "confidence": 0.6,
              "crop_key": "1/j/photo_crops/0002.jpg", "species": "Syrphidae",
              "species_confidence": 0.18}],
         "species_status": {"model": "bioclip", "loaded": True}}


def photo_steps(*extra):
    return [{"id": "p", "block_type": "input.photo", "config": {}},
            {"id": "d", "block_type": "detect.objects", "config": {"label": "insect"},
             "inputs": {"video": "p"}}, *extra]


class ValidationTests(TestCase):
    def test_a_photo_pipeline_validates(self):
        steps = photo_steps(
            {"id": "c", "block_type": "analyze.detection_count", "config": {"metric": "total"}, "inputs": {"detections": "d"}},
            {"id": "s", "block_type": "identify.species", "config": {}, "inputs": {"tracks": "d"}})
        self.assertEqual(validate_steps(steps), [])
        self.assertEqual(pipeline_input_kind(steps), "photo")

    def test_tracking_after_a_photo_is_refused_with_the_reason(self):
        errs = validate_steps(photo_steps(
            {"id": "m", "block_type": "track.mot", "config": {}, "inputs": {"detections": "d"}}))
        self.assertTrue(any("needs a video" in e for e in errs), errs)

    def test_video_and_photo_inputs_dont_mix(self):
        steps = photo_steps({"id": "v", "block_type": "input.video", "config": {}})
        self.assertTrue(any("not both" in e for e in validate_steps(steps)))


class RunTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("ph", password="x")
        self.photo = Video.everything.create(user=self.user, title="p", storage_key="1/p.jpg",
                                             file_size_bytes=1, kind=Video.Kind.PHOTO)
        self.pipeline = Pipeline.objects.create(user=self.user, title="photos", steps=photo_steps())

    def _run(self, steps, context=None):
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps, run.context = steps, context or {}
        return run

    def test_the_photo_is_bound_and_read(self):
        steps = steps_with_video_steps(photo_steps(), self.photo.pk)
        self.assertEqual(steps[0]["config"]["video_id"], str(self.photo.pk))
        out = executors._exec_input_photo(steps[0], self._run(steps), {}, {}, 0)
        self.assertEqual((out["artifact"], out["video_id"]), ("photo", self.photo.pk))

    def test_a_clip_is_not_a_photo(self):
        clip = Video.objects.create(user=self.user, title="c", storage_key="c.mp4", file_size_bytes=1)
        steps = steps_with_video_steps(photo_steps(), clip.pk)
        self.assertIn("error", executors._exec_input_photo(steps[0], self._run(steps), {}, {}, 0))

    def test_detect_on_a_photo_is_the_photo_task_with_species(self):
        steps = steps_with_video_steps(photo_steps(
            {"id": "s", "block_type": "identify.species", "config": {"model": "bioclip"},
             "inputs": {"tracks": "d"}}), self.photo.pk)
        ctx = {"p": {"artifact": "photo", "video_id": self.photo.pk}}
        built, err = executors.build_job_config(steps[1], self._run(steps, ctx), ctx, 1)
        self.assertIsNone(err)
        self.assertEqual(built["video_id"], self.photo.pk)
        self.assertEqual(built["config"]["task"], "detect_photo")
        self.assertEqual(built["config"]["classes"], ["insect"])
        self.assertEqual(built["config"]["species_classifier"], "bioclip")

    def test_the_gpu_job_is_created_for_a_photo(self):
        steps = steps_with_video_steps(photo_steps(), self.photo.pk)
        ctx = {"p": {"artifact": "photo", "video_id": self.photo.pk}}
        state, out = executors.submit_gpu_step(self._run(steps, ctx), steps[1], ctx, 1)
        self.assertEqual(state, "submitted", out)

    def test_counts_and_species_on_a_photo(self):
        count = executors.photo_detection_count(PHOTO)
        self.assertEqual((count["detections"], count["rows"]), (2, [{"class": "insect", "count": 2}]))
        regions = [((0.0, 0.0, 0.5, 0.5), None)]
        self.assertEqual(executors.photo_detection_count(PHOTO, regions)["regions"],
                         [{"region": 1, "count": 1}])
        sp = executors.photo_species(PHOTO, 0.25)
        self.assertEqual([r["taxon"] for r in sp["rows"]], ["Bombus impatiens", "unidentified"])
        self.assertEqual(sp["rows"][1]["best_guess"], "Syrphidae")

    def test_a_detect_node_is_the_reference_on_a_photo(self):
        flowers = {"width": 4000, "height": 3000, "detections": [
            {"x": 0, "y": 0, "w": 2000, "h": 1500, "class": "flower", "confidence": 0.9}]}
        steps = steps_with_video_steps(photo_steps(
            {"id": "f", "block_type": "detect.objects", "config": {"label": "flower"},
             "inputs": {"video": "p"}},
            {"id": "c", "block_type": "analyze.detection_count", "config": {"metric": "total"},
             "inputs": {"detections": "d", "rois": "f"}}), self.photo.pk)
        ctx = {"p": {"artifact": "photo", "video_id": self.photo.pk},
               "d": {"result": {"summary_stats": {"photo": PHOTO}}},
               "f": {"result": {"summary_stats": {"photo": flowers}}}}
        run = self._run(steps, ctx)
        ref = executors.find_reference(steps, 3, ctx, run)
        self.assertEqual(ref["regions"], [{"box": [0.0, 0.0, 0.5, 0.5]}])
        out = executors._exec_analyze_detection_count(steps[3], run, ctx, {"detections": ctx["d"]}, 3)
        self.assertEqual(out["regions"], [{"region": 1, "count": 1}])

    def test_batch_summary_and_csv_rows(self):
        run = self._run(steps_with_video_steps(photo_steps(), self.photo.pk),
                        {"d": {"result": {"summary_stats": {"photo": PHOTO}}}})
        run.save()
        rows = aggregate.photo_rows([run])
        self.assertEqual(len(rows), 2)
        summary = aggregate.photo_summary([run], 0.25)
        self.assertEqual(summary["insects"], 2)
        self.assertEqual({s["name"] for s in summary["species"]}, {"Bombus impatiens", "unidentified"})


class PhotoTabRunTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("pt", password="x")
        self.client.force_login(self.user)
        self.photo = Video.everything.create(user=self.user, title="p", storage_key="1/p.jpg",
                                             file_size_bytes=1, kind=Video.Kind.PHOTO)
        self.photo_pipeline = Pipeline.objects.create(user=self.user, title="ph", steps=photo_steps())

    def test_run_on_photos(self):
        with mock.patch("apps.pipelines.engine.launch_batch",
                        return_value=(__import__("uuid").uuid4(), [self.photo.pk], 0)) as launch, \
                mock.patch("apps.analysis.views._drain_queue"):
            r = self.client.post("/pipelines/run-on-videos/", {
                "pipeline": str(self.photo_pipeline.pk), "kind": "photo",
                "video_ids": [str(self.photo.pk)]}, HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual([v.pk for v in launch.call_args[0][1]], [self.photo.pk])

    def test_a_video_pipeline_is_refused_for_photos(self):
        video_pipeline = Pipeline.objects.create(user=self.user, title="v", steps=[
            {"id": "v", "block_type": "input.video", "config": {}}])
        r = self.client.post("/pipelines/run-on-videos/", {
            "pipeline": str(video_pipeline.pk), "kind": "photo",
            "video_ids": [str(self.photo.pk)]}, HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(r.status_code, 400)

    def test_photos_tab_offers_only_photo_pipelines(self):
        Pipeline.objects.create(user=self.user, title="clips only", steps=[
            {"id": "v", "block_type": "input.video", "config": {}}])
        r = self.client.get("/analysis/processing/?kind=photo")
        titles = [p.title for p in r.context["pipelines"]]
        self.assertEqual(titles, ["ph"])
