"""Left-out days and the YOLO pre-label choice (memory/57).

Left-out days are a test set: clips on them are never sampled, pre-labeled or
trained on. A YOLO model pre-labels through a map from its classes onto the
project's, so a bee model can label "insect"."""

import json
from datetime import date, datetime, timezone as dt_tz
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.annotations import labelers, leftout, sampling_remote as sr
from apps.annotations.models import Annotation, AnnotationProject, FrameSamplingTask, SamplingBatch
from apps.annotations.test_sampling_remote import GPU, FakeAWS
from apps.training.models import CustomModel, TrainingJob
from apps.videos.models import Site, Video

User = get_user_model()
BIG = 50_000_000


def _when(day, hour=10):
    return datetime(day.year, day.month, day.day, hour, tzinfo=dt_tz.utc)


class ParseTests(TestCase):
    def test_sites_bare_dates_comments_and_bad_dates(self):
        rules, errors = leftout.parse(
            "Mendels: 2024-04-21, 2024-04-30\n\n natalies: 2024-05-04 2024-05-08 # rainy\n"
            "2024-06-05\nmendels: 2024-13-45, soon\n")
        self.assertEqual(rules["mendels"], {date(2024, 4, 21), date(2024, 4, 30)})
        self.assertEqual(rules["natalies"], {date(2024, 5, 4), date(2024, 5, 8)})
        self.assertEqual(rules[""], {date(2024, 6, 5)})
        self.assertEqual(len(errors), 2)
        self.assertIn("2024-13-45", errors[0])
        self.assertIn("'soon'", errors[1])

    def test_nothing_in_the_text_raises(self):
        for text in ("", None, ":", "::::", "a:b:c", "\x00", "2024-01-01:"):
            leftout.parse(text)

    def test_describe_is_json_friendly(self):
        rules, _ = leftout.parse("mendels: 2024-04-30, 2024-04-21\n2024-06-05")
        self.assertEqual(leftout.describe(rules),
                         {"*": ["2024-06-05"], "mendels": ["2024-04-21", "2024-04-30"]})


class LeftOutTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("u", password="x")
        self.site = Site.objects.create(user=self.user, name="mendels")
        self.project = AnnotationProject.objects.create(
            user=self.user, name="P", classes=["insect"],
            left_out_days="mendels: 2024-04-30\nnatalies: 2024-05-04\n2024-06-05")

    def clip(self, title, day, site=None, site_name="", size=BIG):
        v = Video.objects.create(user=self.user, title=title, storage_key=f"u/{title}",
                                 file_size_bytes=size, status=Video.Status.READY,
                                 recorded_at=_when(day), site=site, site_name=site_name)
        self.project.videos.add(v)
        return v

    def test_days_match_by_site_fk_or_name_and_bare_dates_match_every_site(self):
        a = self.clip("m-test", date(2024, 4, 30), site=self.site)
        b = self.clip("m-train", date(2024, 5, 4), site=self.site)
        c = self.clip("n-test", date(2024, 5, 4), site_name="Natalies")
        d = self.clip("n-train", date(2024, 4, 30), site_name="natalies")
        e = self.clip("any-site", date(2024, 6, 5), site_name="elsewhere")
        self.assertEqual(leftout.left_out_ids(self.project), {a.pk, c.pk, e.pk})
        self.assertNotIn(b.pk, leftout.left_out_ids(self.project))
        self.assertNotIn(d.pk, leftout.left_out_ids(self.project))

    def test_a_clip_with_no_time_is_never_left_out(self):
        v = self.clip("x", date(2024, 4, 30), site=self.site)
        Video.objects.filter(pk=v.pk).update(recorded_at=None)
        self.assertEqual(leftout.left_out_ids(self.project), set())

    def test_split_for_sampling_skips_left_out_days_and_empty_files(self):
        keep = self.clip("keep", date(2024, 5, 1), site=self.site)
        out = self.clip("out", date(2024, 4, 30), site=self.site)
        empty = self.clip("empty", date(2024, 5, 2), site=self.site, size=2048)
        k, l, e = leftout.split_for_sampling(self.project, self.project.videos.all())
        self.assertEqual(([v.pk for v in k], [v.pk for v in l], [v.pk for v in e]),
                         ([keep.pk], [out.pk], [empty.pk]))
        self.assertIn("1 clip(s) on left-out days and 1 clip(s) with no video",
                      leftout.skipped_note(l, e))

    def test_training_never_sees_left_out_frames_and_records_them(self):
        from apps.training.views import _build_training_payload

        train = self.clip("mendels_2024-05-01_10_00_00.mp4", date(2024, 5, 1), site=self.site)
        test = self.clip("mendels_2024-04-30_10_00_00.mp4", date(2024, 4, 30), site=self.site)
        for v in (train, test):
            Annotation.objects.create(project=self.project, video=v, frame_number=5,
                                      boxes=[{"x": 1, "y": 1, "w": 5, "h": 5, "class": "insect",
                                              "class_id": 0}], reviewed=True)
        job = TrainingJob.objects.create(user=self.user, project=self.project, name="j")
        _classes, groups = _build_training_payload(job)
        self.assertEqual([g["video_blob_path"] for g in groups], [train.storage_key])
        ds = TrainingJob.objects.get(pk=job.pk).metrics["dataset"]
        self.assertEqual(ds["left_out_clips"], 1)
        self.assertEqual(ds["left_out_frames"], 1)
        self.assertEqual(ds["training_clips"], [train.title])
        self.assertEqual(ds["left_out_days"]["mendels"], ["2024-04-30"])

    def test_settings_save_the_days_and_report_problems(self):
        self.clip("m", date(2024, 4, 30), site=self.site)
        self.client.force_login(self.user)
        resp = self.client.post(reverse("annotations:settings", args=[self.project.pk]), {
            "name": "P", "description": "", "class_name": ["insect"], "class_orig": ["0"],
            "left_out_days": "mendels: 2024-04-30\nnowhere: 2024-05-01\nmendels: 2024-99-01",
        }, follow=True)
        self.project.refresh_from_db()
        self.assertIn("nowhere: 2024-05-01", self.project.left_out_days)
        text = " ".join(str(m) for m in resp.context["messages"])
        self.assertIn("2024-99-01", text)
        self.assertIn("site 'nowhere'", text)
        self.assertIn("1 clip(s) are on left-out days", text)


@override_settings(**GPU)
class YoloLabelerTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("y", password="x")
        self.other = User.objects.create_user("o", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P",
                                                        classes=["insect", "nest"])
        self.mine = CustomModel.objects.create(
            user=self.user, name="Bee v4", model_type="custom", base_model="yolo11s",
            storage_key="custom/1/4/best.pt", classes=["bee", "wasp"],
            status=CustomModel.Status.READY, is_active=True)
        self.private = CustomModel.objects.create(
            user=self.other, name="Theirs", model_type="custom", base_model="yolo11s",
            storage_key="custom/2/9/best.pt", classes=["bee"],
            status=CustomModel.Status.READY, is_active=True)
        self.videos = [Video.objects.create(user=self.user, title=f"c{i}", storage_key=f"u/c{i}.mp4",
                                            file_size_bytes=BIG, status=Video.Status.READY)
                       for i in range(2)]
        self.project.videos.add(*self.videos)
        self.aws = FakeAWS()
        patch = mock.patch("boto3.client", side_effect=self.aws.client)
        patch.start()
        self.addCleanup(patch.stop)

    def test_default_map_same_name_else_the_only_class_else_skip(self):
        self.assertEqual(labelers.default_map(["Bee", "wasp"], ["bee", "nest"]),
                         {"Bee": "bee", "wasp": ""})
        self.assertEqual(labelers.default_map(["bee"], ["insect"]), {"bee": "insect"})

    def test_choices_list_own_and_public_models_only(self):
        values = [c["value"] for c in labelers.choices(self.user)]
        self.assertEqual(values[:2], ["sam3", "builtin"])
        self.assertIn(f"cm:{self.mine.pk}", values)
        self.assertNotIn(f"cm:{self.private.pk}", values)

    def test_from_post_maps_and_refuses_bad_picks(self):
        post = {"s_labeler": f"cm:{self.mine.pk}",
                f"s_map:cm:{self.mine.pk}:bee": "insect", f"s_map:cm:{self.mine.pk}:wasp": "insect"}
        spec = labelers.from_post(post, self.user, self.project)
        self.assertEqual(spec["class_map"], {"bee": "insect", "wasp": "insect"})
        self.assertEqual(spec["model_key"], "custom/1/4/best.pt")
        with self.assertRaises(labelers.LabelerError):
            labelers.from_post({"s_labeler": f"cm:{self.private.pk}"}, self.user, self.project)
        with self.assertRaises(labelers.LabelerError):          # everything skipped
            labelers.from_post({"s_labeler": f"cm:{self.mine.pk}"}, self.user, self.project)
        with self.assertRaises(labelers.LabelerError):          # not a project class
            labelers.from_post({"s_labeler": "builtin", "s_map:builtin:bee": "spider"},
                               self.user, self.project)
        self.assertEqual(labelers.from_post({}, self.user, self.project), {"labeler": "sam3"})

    def test_map_boxes_renames_and_drops_unmapped(self):
        boxes = [{"class": "Bee", "x": 1}, {"class": "wasp", "x": 2}]
        self.assertEqual(labelers.map_boxes(boxes, {"bee": "insect"}), [{"class": "insect", "x": 1}])
        self.assertEqual(labelers.map_boxes(boxes, {}), boxes)

    def test_sampling_with_a_yolo_model_sends_its_classes_and_writes_the_projects(self):
        self.client.force_login(self.user)
        self.client.post(reverse("annotations:sample_frames", args=[self.project.pk]), {
            "video_ids": [self.videos[0].pk], "s_method": "sample_label",
            "s_labeler": "builtin", "s_map:builtin:bee": "insect", "s_max_frames": "10"})
        task = FrameSamplingTask.objects.get()
        self.assertEqual(task.params["class_map"], {"bee": "insect"})
        self.assertEqual(task.params["classes"], ["insect"])
        sr.dispatch()
        batch = SamplingBatch.objects.get()
        p = self.aws.payload(batch)
        self.assertEqual((p["detector_kind"], p["classes"], p["confidence"]), ("yolo", ["bee"], 0.25))
        self.assertNotIn("custom_bee_model_path", p)                # built-in model
        self.assertEqual(self.aws.invokes[0]["EndpointName"], "sam3")
        self.aws.finish(batch, [{"task_id": task.pk, "frames": [{
            "n": 7, "key": "frames/k/f000007.jpg", "w": 1920, "h": 1080,
            "boxes": [{"x": 1, "y": 2, "w": 3, "h": 4, "class": "bee"},
                      {"x": 5, "y": 6, "w": 7, "h": 8, "class": "wasp"}]}]}])
        with mock.patch("apps.accounts.quota.charge"):
            sr.collect()
        a = Annotation.objects.get()
        self.assertEqual([(b["class"], b["class_id"]) for b in a.boxes], [("insect", 0)])

    def test_a_custom_model_rides_along_and_sam3_is_unchanged(self):
        self.client.force_login(self.user)
        url = reverse("annotations:sample_frames", args=[self.project.pk])
        self.client.post(url, {"video_ids": [self.videos[0].pk], "s_method": "sample_label",
                               "s_labeler": f"cm:{self.mine.pk}",
                               f"s_map:cm:{self.mine.pk}:wasp": "insect"})
        self.client.post(url, {"video_ids": [self.videos[1].pk], "s_method": "sample_label",
                               "s_classes": ["insect"]})
        sr.dispatch()
        payloads = [self.aws.payload(b) for b in SamplingBatch.objects.order_by("pk")]
        self.assertEqual(payloads[0]["custom_bee_model_path"], "custom/1/4/best.pt")
        self.assertEqual(payloads[0]["classes"], ["wasp"])
        self.assertNotIn("detector_kind", payloads[1])
        self.assertEqual((payloads[1]["classes"], payloads[1]["confidence"]), (["insect"], 0.3))

    def test_the_sampling_forms_offer_the_models_and_their_class_maps(self):
        self.client.force_login(self.user)
        for url in (reverse("annotations:detail", args=[self.project.pk]) + "?tab=clips",
                    reverse("annotations:add_videos_page", args=[self.project.pk])):
            html = self.client.get(url).content.decode()
            self.assertIn('name="s_labeler"', html, url)
            self.assertIn(f'value="cm:{self.mine.pk}"', html, url)
            self.assertIn(f'name="s_map:cm:{self.mine.pk}:bee"', html, url)
            self.assertNotIn(f"cm:{self.private.pk}", html, url)

    def test_a_refused_pick_samples_nothing(self):
        self.client.force_login(self.user)
        self.client.post(reverse("annotations:sample_frames", args=[self.project.pk]), {
            "video_ids": [self.videos[0].pk], "s_method": "sample_label",
            "s_labeler": f"cm:{self.private.pk}"})
        self.assertFalse(FrameSamplingTask.objects.exists())

    def test_single_clip_pre_annotation_maps_classes_on_finalize(self):
        from apps.annotations.views import _create_preannotation_task, finalize_preannotation_task

        req = mock.Mock(user=self.user)
        req.POST = mock.MagicMock()
        req.POST.get = {"labeler": "builtin", "map:builtin:bee": "insect"}.get
        req.POST.getlist = lambda k: []
        task = _create_preannotation_task(req, self.project, self.videos[0])
        self.assertEqual((task.labeler, task.params["class_map"], task.params["target_labels"]),
                         ("yolo", {"bee": "insect"}, ["insect"]))
        task.status = task.Status.PROCESSING
        task.output_uri = "s3://out/x.out"
        task.save()
        result = {"frames": [{"frame_number": 3, "boxes": [{"x": 1, "y": 1, "w": 2, "h": 2,
                                                            "class": "bee"}]}]}
        self.aws.objects[("out", "x.out")] = json.dumps(result).encode()
        with mock.patch("apps.accounts.quota.charge"):
            self.assertTrue(finalize_preannotation_task(task, self.aws))
        a = Annotation.objects.get()
        self.assertEqual([(b["class"], b["class_id"]) for b in a.boxes], [("insect", 0)])


class ExportReviewedTests(TestCase):
    def test_reviewed_1_exports_only_checked_frames(self):
        user = User.objects.create_user("e", password="x")
        project = AnnotationProject.objects.create(user=user, name="T", classes=["insect"])
        v = Video.objects.create(user=user, title="t", storage_key="u/t.mp4", file_size_bytes=BIG,
                                 status=Video.Status.READY)
        project.videos.add(v)
        for n, checked in ((1, True), (2, False)):
            Annotation.objects.create(project=project, video=v, frame_number=n, reviewed=checked,
                                      boxes=[{"x": 1, "y": 1, "w": 2, "h": 2, "class": "insect", "class_id": 0}])
        self.client.force_login(user)
        import io
        import zipfile
        with mock.patch("apps.annotations.views.ThreadPoolExecutor") as pool:
            pool.return_value.__enter__.return_value.map = lambda f, xs: [None for _ in xs]
            resp = self.client.get(reverse("annotations:export", args=[project.pk]) + "?reviewed=1")
        names = zipfile.ZipFile(io.BytesIO(resp.content)).namelist()
        self.assertEqual(sorted(n for n in names if n.startswith("labels/")),
                         ["labels/val/t_f000001.txt"])


@override_settings(**GPU, SAMPLING_SCAN_SECONDS=60)
class ScanWindowAndRoiTests(TestCase):
    """Motion is scored on one minute per clip, inside the clip's ROI; a site
    with no ROI is named so it can be drawn."""

    def setUp(self):
        from apps.videos.models import Layout
        self.user = User.objects.create_user("w", password="x")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])
        self.drawn = Site.objects.create(user=self.user, name="mendels")
        self.bare = Site.objects.create(user=self.user, name="natalies")
        Layout.objects.create(site=self.drawn, number=1, roi_override=[0.2, 0.1, 0.8, 0.9])
        self.videos = [Video.objects.create(user=self.user, title=f"c{i}", storage_key=f"u/c{i}.mp4",
                                            file_size_bytes=BIG, status=Video.Status.READY, site=s)
                       for i, s in enumerate((self.drawn, self.bare, self.bare))]
        self.project.videos.add(*self.videos)
        self.aws = FakeAWS()
        patch = mock.patch("boto3.client", side_effect=self.aws.client)
        patch.start()
        self.addCleanup(patch.stop)

    def test_the_gpu_scans_a_minute_and_sites_without_roi_are_named(self):
        self.client.force_login(self.user)
        resp = self.client.post(reverse("annotations:sample_frames", args=[self.project.pk]), {
            "s_method": "sample_label", "s_classes": ["bee"], "s_roi": "device"}, follow=True)
        text = " ".join(str(m) for m in resp.context["messages"])
        self.assertIn("No ROI drawn for natalies:", text)
        self.assertNotIn("mendels", text.split("No ROI drawn for")[1].split(":")[0])
        sr.dispatch()
        payloads = [self.aws.payload(b) for b in SamplingBatch.objects.all()]
        self.assertEqual({p["scan_seconds"] for p in payloads}, {60})
        rois = {c["task_id"]: c["roi"] for p in payloads for c in p["clips"]}
        drawn_task = FrameSamplingTask.objects.get(video=self.videos[0])
        self.assertEqual(rois[drawn_task.pk], [0.2, 0.1, 0.8, 0.9])
