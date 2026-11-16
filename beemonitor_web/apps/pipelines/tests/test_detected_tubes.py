"""References found by a model in each clip (memory/82), numbered as the
Reference node says: a grid of rows and columns, or as found (memory/83).

Fixing a 6,306-clip batch's tubes by hand went a day at a time; a model finds
them instead, on CPU in the recount worker. Matching them to a drawing made
days to check by hand; a grid numbers them with nothing to check.
"""

import json
import uuid
from datetime import datetime, timezone as dt_timezone
from unittest import mock

import numpy as np
from django.test import SimpleTestCase
from django.urls import reverse

from apps.pipelines import executors, recount, references, regroup, tubes
from apps.pipelines.models import PipelineRun
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


GRID = {"arrangement": "grid", "rows": 4, "columns": 6}


class ArrangeTests(SimpleTestCase):
    def settings(self, **cfg):
        return tubes.tube_settings({**GRID, **cfg})

    def test_numbered_row_by_row_with_a_false_box_left_out_and_a_missed_one_placed(self):
        seen = [n for n in grid() if n["id"] != 9] + [{"id": 0, "box": [0.15, 0.85, 0.15 + W, 0.85 + H]}]
        out = tubes.arrange(found(seen), self.settings())
        self.assertTrue(out["ok"], out["reason"])
        self.assertEqual([n["id"] for n in out["nests"]], list(range(1, 25)))
        self.assertEqual((out["filled"], out["dropped"]), (1, 1))
        for want, got in zip(grid(), out["nests"]):
            self.assertAlmostEqual(got["box"][0], want["box"][0], places=3)
            self.assertAlmostEqual(got["box"][1], want["box"][1], places=3)

    def test_a_grid_that_cant_be_told_says_why(self):
        out = tubes.arrange(found(grid(rows=3)), self.settings())
        self.assertFalse(out["ok"])
        self.assertIn("3 rows of 4", out["reason"])
        self.assertEqual(out["nests"], [])

    def test_as_found_is_reading_order_with_no_count(self):
        boxes = found(grid(rows=2, cols=2))[::-1]
        out = tubes.arrange(boxes, self.settings(arrangement="found"))
        self.assertEqual([n["id"] for n in out["nests"]], [1, 2, 3, 4])
        self.assertLess(out["nests"][0]["box"][0], out["nests"][1]["box"][0])
        self.assertFalse(tubes.arrange([], self.settings(arrangement="found"))["ok"])

    def test_settings_default_to_a_six_by_ten_grid(self):
        s = tubes.tube_settings({"source": "detected"})
        self.assertEqual((s["arrangement"], s["rows"], s["columns"]), ("grid", 6, 10))
        self.assertEqual(tubes.tube_settings({"rows": "3", "columns": "0"})["columns"], 1)


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
            site=self.site, site_name="mendels", width=640, height=360, duration_seconds=60.0,
            recorded_at=datetime(2024, 5, 8, 9, tzinfo=dt_timezone.utc))
        self.video.refresh_from_db()
        self.template = grid(rows=2, cols=3)
        Layout.save_new(site=self.site, roi=[0.0, 0.0, 1.0, 1.0], polygon=None,
                        nests=self.template, user=self.user)
        self.seen = grid(rows=2, cols=3, dx=0.2 * W)

    def ref_config(self, **extra):
        return {"source": "detected", "tube_model": model_classes.BUILTIN_NEST,
                "tube_class": "nest", "arrangement": "grid", "rows": 2, "columns": 3, **extra}

    def steps(self, **extra):
        steps = self._steps("analyze.events")
        steps[3]["config"] = self.ref_config(**extra)
        return steps

    def stored(self, nests=None, video=None):
        return TubeDetection.objects.create(
            video=video or self.video, model_key=model_classes.BUILTIN_NEST_KEY, label="nest",
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

    def test_the_tubes_are_numbered_as_the_nodes_grid(self):
        self.stored(list(reversed(self.seen)))
        out = self.resolve()
        self.assertEqual(out["source"], "detected")
        self.assertEqual([n["id"] for n in out["nest_layout"]], [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(out["nest_layout"][0]["box"][0], self.seen[0]["box"][0])
        self.assertEqual(out["hotel_roi"], [0.0, 0.0, 1.0, 1.0])        # the saved ROI
        self.assertEqual((out["tubes"]["state"], out["tubes"]["filled"]), ("placed", 0))
        self.assertIn("2 × 3 grid", out["source_label"])

    def test_a_drawing_doesnt_replace_the_model(self):
        """The pipeline decides: set to a model, the model's tubes count."""
        self.stored()
        Layout.save_new(video=self.video, roi=None, polygon=None,
                        nests=[{"id": 1, "box": [0.5, 0.5, 0.6, 0.6]}], user=self.user)
        out = self.resolve()
        self.assertEqual((out["tubes"]["state"], len(out["nest_layout"])), ("placed", 6))

    def test_a_grid_that_cant_be_told_borrows_the_nearest_clips_at_its_site(self):
        from datetime import timedelta

        self.stored(self.seen[:3])                      # one row of two
        when = self.video.recorded_at
        for hours, nests in ((5, self.seen), (30, grid(rows=2, cols=3, dx=0.1))):
            other = Video.objects.create(user=self.user, site=self.site, site_name="mendels",
                                         title=f"{hours}h", storage_key=f"t/{hours}.mp4",
                                         file_size_bytes=1, status=Video.Status.READY,
                                         recorded_at=when + timedelta(hours=hours))
            self.stored(nests, video=other)
        out = self.resolve()
        self.assertEqual(out["tubes"]["state"], "borrowed")
        self.assertEqual(out["tubes"]["from_video"], Video.objects.get(title="5h").pk)
        self.assertIn("1 rows of 2", out["tubes"]["reason"])
        self.assertAlmostEqual(out["nest_layout"][5]["box"][0], self.seen[5]["box"][0])

    def test_with_no_grid_near_it_is_counted_without_tubes(self):
        self.stored(self.seen[:3])
        out = self.resolve()
        self.assertEqual(out["tubes"]["state"], "unplaced")
        self.assertIsNone(out["nest_layout"])

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
        self.assertEqual(run.context["r"]["tubes"]["state"], "placed")
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
        # The player draws the tubes the events were counted against.
        from apps.analysis.models import Job
        job = Job.objects.create(user=self.user, video=self.video, status="completed",
                                 modal_job_id=uuid.uuid4().hex)
        with mock.patch.object(executors, "_job_run", return_value=run), \
                mock.patch.object(Job, "result", create=True, new=object()):
            drawn = executors.geometry_for_job(job)
        self.assertEqual(drawn["tubes"]["state"], "placed")
        # The listing reads the new node, not the old one first in the steps.
        self.assertEqual(references.counted(run)["tubes"]["state"], "placed")
        from apps.pipelines import listing
        self.assertEqual(listing.rows(PipelineRun.objects.filter(pk=run.pk))[0]["layout"]
                         ["tubes"]["state"], "placed")


class BatchPageTests(DetectedReferenceBase):
    """No References page: one line on the batch page (memory/83)."""

    def setUp(self):
        super().setUp()
        self.client.force_login(self.user)
        self.run = self.run_with(self.steps())
        self.batch = self.run.batch_id
        self.page = reverse("pipelines:batch_detail", args=[self.batch])

    def count(self, nests):
        TubeDetection.objects.all().delete()
        self.stored(nests)
        recount.recount_run(PipelineRun.objects.get(pk=self.run.pk))
        PipelineRun.objects.filter(pk=self.run.pk).update(counting_rules=recount.CURRENT_RULES)

    def test_the_batch_page_says_how_the_clips_were_numbered(self):
        self.count(self.seen)
        page = self.client.get(self.page)
        self.assertEqual(page.context["tube_summary"]["placed"], 1)
        self.assertContains(page, "Numbered by model")
        self.assertNotContains(page, "Renumber")

    def test_clips_not_placed_or_numbered_the_old_way_are_renumbered(self):
        self.count(self.seen[:3])
        page = self.client.get(self.page)
        self.assertEqual(page.context["tube_summary"]["unplaced"], 1)
        self.assertContains(page, "Renumber 1 clip")
        # Fixed by a nearby clip at the site, then renumbered from the page.
        from datetime import timedelta
        other = Video.objects.create(user=self.user, site=self.site, site_name="mendels", title="n",
                                     storage_key="t/n.mp4", file_size_bytes=1,
                                     status=Video.Status.READY,
                                     recorded_at=self.video.recorded_at + timedelta(hours=1))
        self.stored(self.seen, video=other)
        self.client.post(reverse("pipelines:batch_tubes_recount", args=[self.batch]))
        self.run.refresh_from_db()
        self.assertLess(self.run.counting_rules, recount.CURRENT_RULES)
        self.assertEqual(self.client.get(reverse("pipelines:batch_recount_progress",
                                                 args=[self.batch])).json()["pending"], 1)
        recount.sweep()
        self.assertEqual(self.client.get(self.page).context["tube_summary"]["borrowed"], 1)

    def test_only_the_launcher_renumbers(self):
        other = type(self.user).objects.create_user("someone", password="x")
        self.client.force_login(other)
        res = self.client.post(reverse("pipelines:batch_tubes_recount", args=[self.batch]))
        self.assertEqual(res.status_code, 404)

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
        # Not recounted yet: the player still draws the layout it was counted with.
        waiting = PipelineRun.objects.get(pk=run.pk)
        from apps.analysis.models import Job
        job0 = Job.objects.create(user=self.user, video=self.video, status="completed",
                                  modal_job_id=uuid.uuid4().hex)
        with mock.patch.object(executors, "_job_run", return_value=waiting), \
                mock.patch.object(Job, "result", create=True, new=object()):
            self.assertEqual(executors.geometry_for_job(job0)["source"], "site")
        with self.model(), mock.patch("apps.videos.frames.frame_jpeg", return_value=frame()):
            recount.sweep()
        run.refresh_from_db()
        self.assertEqual(run.steps[4]["inputs"]["rois"], "t")
        self.assertEqual(run.context["t"]["tubes"]["state"], "placed")
        self.assertEqual(run.context["a"]["reference_source"], "detect:nest")
        self.assertIsNone(regroup.batch_changes([run], self.pipeline))
