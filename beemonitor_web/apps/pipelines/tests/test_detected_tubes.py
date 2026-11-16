"""Tube references found by a model in each clip, numbered from the site's
layout (memory/82).

Fixing a 6,306-clip batch's tubes by hand went a day at a time; a model finds
them instead, on CPU in the recount worker. Tube numbers must mean the same
tube on every day, so the boxes are matched to the site's layout.
"""

import json
import uuid
from unittest import mock

import numpy as np
from django.test import SimpleTestCase
from django.urls import reverse

from apps.pipelines import executors, recount, references, regroup, tubes
from apps.pipelines.models import PipelineRun, TubeDayCheck
from apps.pipelines.tests.test_primitive_blocks import PrimitiveBlockTestCase
from apps.training import model_classes
from apps.videos.models import Layout, Site, TubeDetection, Video

W, H = 0.08, 0.1        # a tube's size, 0..1


def grid(rows=4, cols=6, dx=0.0, dy=0.0, scale=1.0, x0=0.1, y0=0.1):
    """A block of tubes, numbered row by row from 1."""
    out = []
    for r in range(rows):
        for c in range(cols):
            x, y = (x0 + c * 0.12) * scale + dx, (y0 + r * 0.18) * scale + dy
            out.append({"id": r * cols + c + 1, "box": [x, y, x + W * scale, y + H * scale]})
    return out


def found(nests):
    return [{"box": n["box"], "confidence": 0.9, "seen": 8} for n in nests]


class MatchTests(SimpleTestCase):
    def test_a_missed_tube_keeps_its_number_and_a_new_one_gets_the_next(self):
        template = grid()
        moved = grid(dx=0.3 * W)
        seen = [n for n in moved if n["id"] != 9] + [{"id": 0, "box": [0.85, 0.3, 0.85 + W, 0.3 + H]}]
        m = tubes.match(template, found(seen))
        by_id = {n["id"]: n for n in m["nests"]}
        self.assertEqual((m["matched"], m["kept"], m["new"], m["flag"]), (23, 1, 1, "new"))
        self.assertTrue(by_id[9]["kept"])
        self.assertAlmostEqual(by_id[9]["box"][0], moved[8]["box"][0], places=3)   # moved with the rest
        self.assertTrue(by_id[25]["new"])
        self.assertAlmostEqual(by_id[2]["box"][0], moved[1]["box"][0], places=5)

    def test_a_camera_moved_by_more_than_a_tube_still_numbers_right(self):
        """Position order would renumber everything; the shift is found first."""
        m = tubes.match(grid(), found(grid(dx=2.4 * W, dy=-0.5 * H)))
        self.assertEqual((m["matched"], m["kept"], m["new"]), (24, 0, 0))
        self.assertEqual(m["flag"], "moved")
        by_id = {n["id"]: n for n in m["nests"]}
        self.assertAlmostEqual(by_id[7]["box"][0], grid(dx=2.4 * W)[6]["box"][0], places=5)

    def test_a_zoom_is_taken_out_too(self):
        m = tubes.match(grid(), found(grid(scale=1.04, dx=0.01)))
        self.assertEqual(m["matched"], 24)
        self.assertEqual(m["flag"], "")

    def test_names_come_from_the_template(self):
        template = grid(rows=1, cols=3)
        template[1]["name"] = "Flag B"
        m = tubes.match(template, found(grid(rows=1, cols=3)))
        self.assertEqual(m["nests"][1]["name"], "Flag B")

    def test_with_no_template_tubes_are_numbered_by_position(self):
        boxes = found(grid(rows=2, cols=2))[::-1]
        m = tubes.match([], boxes)
        self.assertEqual([n["id"] for n in m["nests"]], [1, 2, 3, 4])
        self.assertLess(m["nests"][0]["box"][0], m["nests"][1]["box"][0])
        self.assertEqual(m["flag"], "no_template")

    def test_nothing_found_keeps_the_template(self):
        m = tubes.match(grid(rows=1, cols=2), [])
        self.assertEqual((m["kept"], m["flag"]), (2, "none_found"))


class ObjectsTests(SimpleTestCase):
    def test_a_passing_bee_is_not_a_tube_and_a_hidden_tube_still_is(self):
        tube_a, tube_b = (0.1, 0.1, 0.2, 0.2, 0.9), (0.5, 0.1, 0.6, 0.2, 0.8)
        frames = [[tube_a, tube_b] for _ in range(8)]
        frames[3] = [tube_a]                                  # a bee on tube b
        frames[5].append((0.8, 0.8, 0.85, 0.85, 0.6))        # a bee passing by
        out = tubes.objects(frames, 8)
        self.assertEqual(len(out), 2)
        self.assertEqual(sorted(o["seen"] for o in out), [7, 8])


class FakeNet:
    """An ONNX net's forward pass, as OpenCV gives it."""

    def __init__(self, out):
        self.out = out

    def setInput(self, blob):
        self.blob = blob

    def forward(self):
        return self.out


def e2e(boxes, w, h, cls=1, size=640):
    """An end-to-end head's output (YOLO26) for 0..1 boxes on a w x h frame."""
    r = min(size / h, size / w)
    top, left = (size - round(h * r)) // 2, (size - round(w * r)) // 2
    rows = [[b[0] * w * r + left, b[1] * h * r + top, b[2] * w * r + left, b[3] * h * r + top, 0.9, cls]
            for b in boxes]
    rows += [[0, 0, 0, 0, 0.0, 0]] * (300 - len(rows))
    return np.array([rows], dtype=np.float32)


class DecodeTests(SimpleTestCase):
    img = np.zeros((360, 640, 3), np.uint8)

    def test_an_end_to_end_head(self):
        out = e2e([[0.1, 0.2, 0.3, 0.4]], 640, 360)
        got = tubes.boxes_in(FakeNet(out), {"imgsz": 640}, self.img, {1}, 0.4)
        self.assertEqual(len(got), 1)
        np.testing.assert_allclose(got[0][:4], [64, 72, 192, 144], atol=0.5)

    def test_a_v8_head_with_nms(self):
        # (4 + 2 classes) x candidates: two overlapping boxes of class 1, one of class 0.
        cand = [[100, 300, 40, 40, 0.1, 0.9], [102, 301, 40, 40, 0.1, 0.8], [400, 300, 40, 40, 0.9, 0.1]]
        cand += [[0, 0, 0, 0, 0.0, 0.0]] * 97          # a real head has thousands of candidates
        out = np.array([np.array(cand, dtype=np.float32).T])
        got = tubes.boxes_in(FakeNet(out), {"imgsz": 640}, self.img, {1}, 0.4)
        self.assertEqual(len(got), 1)
        np.testing.assert_allclose(got[0][:4], [80, 280 - 140, 120, 320 - 140], atol=0.5)


def frame():
    import cv2

    return cv2.imencode(".jpg", np.zeros((360, 640, 3), np.uint8))[1].tobytes()


class DetectedReferenceBase(PrimitiveBlockTestCase):
    """A site with a saved block of 6 tubes and a clip recorded there."""

    def setUp(self):
        super().setUp()
        self.site = Site.objects.create(user=self.user, name="mendels", tz_name="UTC")
        Video.objects.filter(pk=self.video.pk).update(
            site=self.site, site_name="mendels", width=640, height=360, duration_seconds=60.0)
        self.video.refresh_from_db()
        self.template = grid(rows=2, cols=3)
        Layout.save_new(site=self.site, roi=[0.0, 0.0, 1.0, 1.0], polygon=None,
                        nests=self.template, user=self.user)
        self.seen = grid(rows=2, cols=3, dx=0.2 * W)

    def ref_config(self, **extra):
        return {"source": "detected", "tube_model": model_classes.BUILTIN_NEST,
                "tube_class": "nest", **extra}

    def steps(self, **extra):
        steps = self._steps("analyze.events")
        steps[3]["config"] = self.ref_config(**extra)
        return steps

    def stored(self, nests=None):
        return TubeDetection.objects.create(
            video=self.video, model_key=model_classes.BUILTIN_NEST_KEY, label="nest",
            settings_key=tubes.settings_key(tubes.DEFAULT_FRAMES, tubes.DEFAULT_CONFIDENCE),
            boxes=found(nests if nests is not None else self.seen), frames=8)

    def model(self, nests=None):
        """The model on CPU, finding these tubes on every frame."""
        out = e2e([n["box"] for n in (nests if nests is not None else self.seen)], 640, 360)
        return mock.patch.multiple(
            "apps.pipelines.tubes", _load=mock.Mock(return_value=(
                FakeNet(out), {"names": ["hotel", "nest"], "imgsz": 640})))

    def run_with(self, steps):
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user, steps=steps,
                                         status=PipelineRun.Status.COMPLETED,
                                         batch_id=uuid.uuid4())
        run.context = {"v": {"artifact": "video", "video_id": self.video.pk},
                       "m": {"artifact": "tracks", "result": {
                           "tracking_csv_path": self.tracking_csv,
                           "summary_stats": {"video_fps": 25.0}}}}
        run.save()
        return run


class DetectedReferenceTests(DetectedReferenceBase):
    def resolve(self, run=None, **extra):
        run = run or PipelineRun(pipeline=self.pipeline, user=self.user, steps=self.steps(**extra))
        context = {"v": {"artifact": "video", "video_id": self.video.pk}}
        return executors.resolve_reference(run.steps[3], run, context, 3)

    def test_the_detected_tubes_take_the_sites_numbers(self):
        self.stored()
        out = self.resolve()
        self.assertEqual(out["source"], "detected")
        self.assertEqual([n["id"] for n in out["nest_layout"]], [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(out["nest_layout"][0]["box"][0], self.seen[0]["box"][0])
        self.assertEqual(out["hotel_roi"], [0.0, 0.0, 1.0, 1.0])        # the saved ROI
        self.assertEqual((out["tubes"]["state"], out["tubes"]["matched"]), ("detected", 6))

    def test_a_clip_drawn_by_hand_keeps_its_drawing(self):
        self.stored()
        Layout.save_new(video=self.video, roi=None, polygon=None,
                        nests=[{"id": 1, "box": [0.5, 0.5, 0.6, 0.6]}], user=self.user)
        self.assertEqual(self.resolve()["tubes"]["state"], "drawn")
        self.assertEqual(self.resolve(drawn_wins="off")["tubes"]["state"], "detected")

    def test_without_a_detection_a_fresh_run_waits_for_the_worker(self):
        out = self.resolve()
        self.assertEqual(out["tubes"]["state"], "pending")
        self.assertIsNone(out["nest_layout"])
        self.assertFalse(TubeDetection.objects.exists())

    def test_the_recount_finds_the_tubes_on_cpu(self):
        run = self.run_with(self.steps())
        with self.model(), mock.patch("apps.videos.frames.frame_jpeg", return_value=frame()) as grab:
            recount.recount_run(run)
        self.assertEqual(grab.call_count, tubes.DEFAULT_FRAMES)
        det = TubeDetection.objects.get(video=self.video)
        self.assertEqual((len(det.boxes), det.frames, det.error), (6, 8, ""))
        run.refresh_from_db()
        self.assertEqual(run.context["r"]["tubes"]["matched"], 6)
        self.assertEqual({r["reference"] for r in run.context["a"]["rows"]} - {"reference_1"}, set())
        # Found once: the next recount reads it.
        with self.model() as _m, mock.patch("apps.videos.frames.frame_jpeg") as again:
            recount.recount_run(PipelineRun.objects.get(pk=run.pk))
        again.assert_not_called()

    def test_a_class_the_model_doesnt_have_says_so(self):
        run = self.run_with(self.steps(tube_class="tube"))
        with self.model(), mock.patch("apps.videos.frames.frame_jpeg", return_value=frame()):
            recount.recount_run(run)
        self.assertIn('"tube" is not one of', TubeDetection.objects.get().error)
        run.refresh_from_db()
        self.assertEqual(run.context["r"]["tubes"]["state"], "error")

    def test_a_finished_run_waiting_for_its_tubes_is_queued(self):
        from apps.pipelines import engine

        run = self.run_with(self.steps())
        run.status = PipelineRun.Status.RUNNING
        run.context["r"] = {"artifact": "roi", "tubes": {"state": "pending"}}
        with self.captureOnCommitCallbacks(execute=True):
            engine._finalize_if_terminal(run, [], {})          # no steps: still "terminal"
            run.status = PipelineRun.Status.COMPLETED
            steps = [{"id": "r"}]
            engine._finalize_if_terminal(run, steps, {"r": PipelineRun.STEP_DONE})
            run.save()
        run.refresh_from_db()
        self.assertEqual(run.counting_rules, 0)
        self.assertIsNotNone(run.recount_requested_at)


class UpdatingABatchTests(DetectedReferenceBase):
    def test_switching_to_detected_tubes_is_a_web_change_done_by_the_worker(self):
        steps = self._steps("analyze.events")
        run = self.run_with(steps)
        recount.recount_run(run)
        PipelineRun.objects.filter(pk=run.pk).update(counting_rules=recount.CURRENT_RULES)
        run.refresh_from_db()
        self.assertEqual(run.context["r"]["source"], "site")

        self.pipeline.steps = self.steps()
        self.pipeline.save()
        changes = regroup.batch_changes([run], self.pipeline)
        self.assertEqual(changes["gpu"], [])
        self.assertTrue(any(c["field"] == "Source" for c in changes["web"]))

        with mock.patch("apps.pipelines.tubes.detect") as detect:
            done = regroup.apply_pipeline(run.batch_id, self.user)
        detect.assert_not_called()                     # not inside the request
        self.assertEqual(done["queued"], 1)
        with self.model(), mock.patch("apps.videos.frames.frame_jpeg", return_value=frame()):
            recount.sweep()
        run.refresh_from_db()
        self.assertEqual(run.context["r"]["source"], "detected")
        self.assertEqual(run.counting_rules, recount.CURRENT_RULES)
        self.assertIsNone(regroup.batch_changes([run], self.pipeline))


class ReferencesPageTests(DetectedReferenceBase):
    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)
        self.run = self.run_with(self.steps())
        self.batch = self.run.batch_id
        self.page = reverse("pipelines:batch_references", args=[self.batch])

    def count(self, nests):
        TubeDetection.objects.all().delete()
        self.stored(nests)
        recount.recount_run(PipelineRun.objects.get(pk=self.run.pk))
        PipelineRun.objects.filter(pk=self.run.pk).update(counting_rules=recount.CURRENT_RULES)

    def days(self):
        from apps.pipelines import listing

        return references.days(listing.rows(PipelineRun.objects.filter(batch_id=self.batch)))

    def test_a_day_numbered_cleanly_is_detected(self):
        self.count(self.seen)
        day = self.days()[0]
        self.assertEqual((day["kind"], day["detected"], day["up_to_date"]), ("detected", 1, 1))
        self.assertContains(self.client.get(self.page), "days detected")

    def test_a_day_with_a_new_tube_is_to_look_at_until_it_is_checked(self):
        self.count(self.seen + [{"id": 0, "box": [0.8, 0.6, 0.88, 0.7]}])
        day = self.days()[0]
        self.assertEqual(day["kind"], "look")
        page = self.client.get(self.page + "?show=look")
        self.assertContains(page, "Look at this day")
        day_url = reverse("pipelines:batch_references_day", args=[self.batch, self.video.pk])
        res = self.client.get(day_url)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.context["tubes"]["new"], 1)
        self.assertEqual([s["mark"] for s in res.context["shapes"]].count("new"), 1)
        self.assertIn("tubes_of=", res.context["edit_url"])

        self.client.post(reverse("pipelines:batch_references_action", args=[self.batch]),
                         {"action": "checked", "day": self.video.pk})
        self.assertTrue(TubeDayCheck.objects.exists())
        self.assertNotContains(self.client.get(self.page + "?show=look"), "Look at this day")

    def test_a_days_detection_can_become_the_sites_numbering(self):
        extra = self.seen + [{"id": 0, "box": [0.8, 0.6, 0.88, 0.7]}]
        self.count(extra)
        self.client.post(reverse("pipelines:batch_references_action", args=[self.batch]),
                         {"action": "numbering", "day": self.video.pk, "run": self.run.pk})
        site = Layout.current(site=self.site)
        self.assertEqual(len(site.nest_layout), 7)
        self.assertEqual(site.roi_override, [0.0, 0.0, 1.0, 1.0])
        self.run.refresh_from_db()
        self.assertEqual(self.run.counting_rules, 0)          # renumbered and recounted
        recount.recount_run(PipelineRun.objects.get(pk=self.run.pk))
        PipelineRun.objects.filter(pk=self.run.pk).update(counting_rules=recount.CURRENT_RULES)
        self.assertEqual(self.days()[0]["kind"], "detected")

    def test_the_editor_opens_with_the_detected_tubes(self):
        self.count(self.seen)
        res = self.client.get(reverse("videos:layout", args=[self.video.pk])
                              + f"?scope=day&tubes_of={self.run.pk}")
        self.assertEqual(json.loads(res.context["nests_json"])[0]["box"][0], self.seen[0]["box"][0])
        self.assertContains(res, "Showing the tubes found by")


class ModelPickerTests(PrimitiveBlockTestCase):
    def test_the_detect_node_offers_the_built_in_nest_model_with_its_classes(self):
        self.client.force_login(self.user)
        res = self.client.get(reverse("pipelines:editor", args=[self.pipeline.pk]))
        models = json.loads(res.context["yolo_models_json"])
        nest = next(m for m in models if m["id"] == model_classes.BUILTIN_NEST)
        self.assertEqual(nest["classes"], ["hotel", "nest"])
        self.assertEqual(models[0]["id"], "")                   # the default bee model

    def test_picking_it_sends_the_workers_own_nest_model(self):
        config = {}
        err = executors._resolve_custom_models({"object_model": model_classes.BUILTIN_NEST},
                                               PipelineRun(user=self.user), config)
        self.assertIsNone(err)
        self.assertEqual(config["custom_bee_model_path"], "v1/nest_detection.pt")

    def test_a_trained_models_classes_are_its_projects(self):
        from apps.training.models import CustomModel

        m = CustomModel.objects.create(user=self.user, name="Tubes", storage_key="custom/1/2/best.pt",
                                       classes=["nest tube", "hotel"],
                                       status=CustomModel.Status.READY, is_active=True)
        self.assertEqual(model_classes.resolve(str(m.pk), self.user)["classes"], ["nest tube", "hotel"])


class DetectLabelTests(PrimitiveBlockTestCase):
    def test_the_models_own_name_for_its_one_class_is_not_a_gpu_change(self):
        """memory/82: "insect" saved for another model, "insects" picked from
        this one's dropdown: the same detections, not 894 GPU-hours."""
        from apps.training.models import CustomModel

        m = CustomModel.objects.create(user=self.user, name="Insects", storage_key="custom/1/10/best.pt",
                                       classes=["insects"], status=CustomModel.Status.READY,
                                       is_active=True)
        steps = self._steps("analyze.events")
        steps[1]["config"] = {"label": "insect", "object_model": str(m.pk)}
        run = PipelineRun(pipeline=self.pipeline, user=self.user, steps=steps)
        after = json.loads(json.dumps(steps))
        after[1]["config"]["label"] = "insects"
        # The editor saves every field; the clip's step predates some of them.
        after[1]["config"].update({"model_family": "yolo", "analyse": "every_frame",
                                   "confidence": "0.4"})
        self.pipeline.steps = after
        self.assertEqual(regroup.pipeline_changes(run, self.pipeline, retracks=False)["gpu"], [])
        # Another class of a two-class model is a real change.
        m.classes = ["insects", "flower"]
        m.save()
        self.pipeline.__dict__.pop("_model_classes", None)     # read once a pipeline
        after[1]["config"]["label"] = "flower"
        self.assertTrue(regroup.pipeline_changes(run, self.pipeline, retracks=False)["gpu"])


class BigBatchUpdateTests(DetectedReferenceBase):
    def test_a_big_batch_updates_in_the_background_every_clip_to_the_worker(self):
        """memory/82: Update on 5,930 clips ran longer than a request may."""
        run = self.run_with(self._steps("analyze.events"))
        recount.recount_run(run)
        PipelineRun.objects.filter(pk=run.pk).update(counting_rules=recount.CURRENT_RULES)
        self.pipeline.steps = self.steps()
        self.pipeline.save()
        self.client.force_login(self.user)
        with mock.patch.object(regroup, "BACKGROUND_FROM", 0), \
                mock.patch("apps.pipelines.recount.recount_run") as inline:
            res = self.client.post(reverse("pipelines:batch_apply_groups", args=[run.batch_id]),
                                   {"which": "all"}, follow=True)
        inline.assert_not_called()
        self.assertContains(res, "in the background")
        run.refresh_from_db()
        self.assertEqual(run.steps[3]["config"]["source"], "detected")
        self.assertEqual(run.counting_rules, 0)
        self.assertIsNotNone(run.recount_requested_at)


class NewReferenceNodeTests(DetectedReferenceBase):
    def test_a_reference_node_added_and_wired_in_joins_the_clips_on_update(self):
        """memory/82: a new Reference node wired to Events was listed as a
        change, but Update only carried settings onto nodes the clips had."""
        run = self.run_with(self._steps("analyze.events"))
        recount.recount_run(run)
        PipelineRun.objects.filter(pk=run.pk).update(counting_rules=recount.CURRENT_RULES)
        steps = self._steps("analyze.events")
        steps.append({"id": "t", "block_type": "reference.layout", "config": self.ref_config(),
                      "inputs": {"video": "v"}})
        steps[4]["inputs"] = {"tracks": "m", "rois": "t"}
        self.pipeline.steps = steps
        self.pipeline.save()
        changes = regroup.batch_changes([PipelineRun.objects.get(pk=run.pk)], self.pipeline)
        self.assertEqual(changes["gpu"], [])
        regroup.apply_pipeline(run.batch_id, self.user)
        with self.model(), mock.patch("apps.videos.frames.frame_jpeg", return_value=frame()):
            recount.sweep()
        run.refresh_from_db()
        self.assertEqual(run.steps[4]["inputs"]["rois"], "t")
        self.assertEqual(run.context["t"]["tubes"]["state"], "detected")
        self.assertEqual(run.context["a"]["reference_source"], "detect:nest")
        self.assertIsNone(regroup.batch_changes([run], self.pipeline))
