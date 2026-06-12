"""Tests for the activity pages, frame-ingest wiring, and the BioCLIP pipeline.

All network/S3 is stubbed — these run with no AWS access.
"""

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.devices.models import Device

from . import pipeline
from .models import Activity, ActivityFrame, Detection, Observation, Taxon


class ActivityPageTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        User = get_user_model()
        cls.user = User.objects.create_user(username="alice", password="pw12345!")
        cls.other = User.objects.create_user(username="bob", password="pw12345!")
        cls.device, _ = Device.create_with_key(owner=cls.user, name="hive-1")
        cls.activity = Activity.objects.create(
            device=cls.device, activity_uid="evt-1", started_at=timezone.now(),
            lat=40.8, lon=-77.8, peak_motion=1234,
        )
        # storage_key points nowhere real; the view presigns best-effort (None on fail).
        ActivityFrame.objects.create(activity=cls.activity, storage_key="x/y/z.jpg",
                                     kind="crop", motion_score=999)

    def test_list_requires_login(self):
        resp = self.client.get(reverse("monitor:activity_list"))
        self.assertEqual(resp.status_code, 302)  # redirect to login

    def test_list_and_detail_render(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("monitor:activity_list"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "hive-1")
        resp = self.client.get(reverse("monitor:activity_detail", args=[self.activity.pk]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Analysis pending")  # no detections yet

    def test_detail_access_scoped_to_device(self):
        self.client.force_login(self.other)  # no access to alice's device
        resp = self.client.get(reverse("monitor:activity_detail", args=[self.activity.pk]))
        self.assertEqual(resp.status_code, 404)

    def test_device_filter(self):
        self.client.force_login(self.user)
        resp = self.client.get(reverse("monitor:activity_list"), {"device": self.device.pk})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(list(resp.context["activities"]), [self.activity])


class FrameIngestTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = get_user_model().objects.create_user(username="carol", password="pw12345!")
        self.device, self.raw_key = Device.create_with_key(owner=self.user, name="hive-2")

    def test_requires_device_auth(self):
        resp = self.client.post("/api/v1/devices/frames", {})
        self.assertEqual(resp.status_code, 401)

    def test_missing_activity_uid_is_400(self):
        resp = self.client.post(
            "/api/v1/devices/frames", {"meta": "{}"},
            HTTP_AUTHORIZATION=f"Bearer {self.raw_key}",
        )
        self.assertEqual(resp.status_code, 400)


_BUMBLEBEE = {
    "score": 0.9,
    "common_name": "common eastern bumble bee",
    "ranks": {"kingdom": "Animalia", "phylum": "Arthropoda", "class": "Insecta",
              "order": "Hymenoptera", "family": "Apidae", "genus": "Bombus",
              "species": "Bombus impatiens"},
}


@override_settings(SAGEMAKER_BIOCLIP_ENDPOINT_NAME="test-ep",
                   MONITOR_BIOCLIP_MIN_CONFIDENCE=0.2)
class PipelineTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(username="dan", password="pw12345!")
        self.device, _ = Device.create_with_key(owner=self.user, name="hive-3")
        self.activity = Activity.objects.create(
            device=self.device, activity_uid="evt-9", started_at=timezone.now())
        self.frames = [
            ActivityFrame.objects.create(activity=self.activity, storage_key=f"f{i}.jpg")
            for i in range(2)
        ]

    def test_resolve_taxon_builds_chain_with_parents(self):
        sp = pipeline._resolve_taxon(_BUMBLEBEE["ranks"])
        self.assertEqual(sp.rank, "species")
        self.assertEqual(sp.name, "Bombus impatiens")
        self.assertEqual(sp.parent.name, "Bombus")          # genus
        self.assertEqual(sp.parent.parent.name, "Apidae")    # family
        self.assertEqual(Taxon.objects.count(), 7)           # one per rank

    def test_enabled_guard_noops_without_endpoint(self):
        with override_settings(SAGEMAKER_BIOCLIP_ENDPOINT_NAME=""):
            self.assertFalse(pipeline.enabled())
            pipeline.classify_frame_async(self.frames[0].id)  # must not raise/queue
        self.assertEqual(Detection.objects.count(), 0)

    def _run_with_preds(self, preds):
        # Stub S3 read + endpoint; no-op the connection.close (keeps the test txn).
        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "_invoke_bioclip", return_value=preds), \
                mock.patch.object(pipeline, "connection"):
            for fr in self.frames:
                pipeline.classify_frame(fr.id)

    def test_full_classify_marks_analyzed(self):
        self._run_with_preds([_BUMBLEBEE])
        self.assertEqual(Detection.objects.count(), 2)
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.ANALYZED)
        self.assertEqual(self.activity.best_taxon.name, "Bombus impatiens")
        self.assertAlmostEqual(self.activity.best_confidence, 0.9)
        obs = Observation.objects.get(activity=self.activity)
        self.assertEqual(obs.taxon.name, "Bombus impatiens")
        self.assertEqual(obs.individual_count, 1)

    def test_low_confidence_is_no_detection(self):
        weak = {**_BUMBLEBEE, "score": 0.05}
        self._run_with_preds([weak])
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.NO_DETECTION)
        self.assertFalse(Observation.objects.filter(activity=self.activity).exists())

    def test_partial_classification_waits(self):
        # Only one of two frames classified -> activity stays pending, no rollup.
        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "_invoke_bioclip", return_value=[_BUMBLEBEE]), \
                mock.patch.object(pipeline, "connection"):
            pipeline.classify_frame(self.frames[0].id)
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.PENDING)
        self.assertFalse(Observation.objects.filter(activity=self.activity).exists())
