"""Tests for the activity pages, frame-ingest wiring, and the BioCLIP pipeline.

All network/S3 is stubbed — these run with no AWS access.
"""

import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.devices.models import Device

from . import pipeline, priors
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

    def test_one_frame_error_still_aggregates_from_the_other(self):
        # A transient failure on one frame must NOT strand the whole activity.
        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "_invoke_bioclip",
                                  side_effect=[RuntimeError("boom"), [_BUMBLEBEE]]), \
                mock.patch.object(pipeline, "connection"):
            pipeline.classify_frame(self.frames[0].id)  # errors
            pipeline.classify_frame(self.frames[1].id)  # succeeds
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.ANALYZED)
        self.assertEqual(self.activity.best_taxon.name, "Bombus impatiens")
        # The errored frame still recorded a (failure) detection so the gate completes.
        self.assertEqual(
            Detection.objects.filter(frame=self.frames[0], taxon__isnull=True).count(), 1)

    def test_all_frames_error_marks_failed(self):
        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "_invoke_bioclip", side_effect=RuntimeError("boom")), \
                mock.patch.object(pipeline, "connection"):
            for fr in self.frames:
                pipeline.classify_frame(fr.id)
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.FAILED)

    def test_reclassify_is_idempotent(self):
        # Re-running must not duplicate Detections or Observations (unique constraints).
        self._run_with_preds([_BUMBLEBEE])
        self._run_with_preds([_BUMBLEBEE])
        self.assertEqual(Detection.objects.count(), 2)  # one per frame, not 4
        self.assertEqual(
            Observation.objects.filter(activity=self.activity).count(), 1)

    def test_location_prior_candidates_passed_to_endpoint(self):
        captured = []

        def fake_invoke(jpeg, candidate_taxa=None):
            captured.append(candidate_taxa)
            return [_BUMBLEBEE]

        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "region_taxa",
                                  return_value=["Bombus impatiens", "Apis mellifera"]), \
                mock.patch.object(pipeline, "_invoke_bioclip", side_effect=fake_invoke), \
                mock.patch.object(pipeline, "connection"):
            pipeline.classify_frame(self.frames[0].id)
        self.assertEqual(captured[0], ["Bombus impatiens", "Apis mellifera"])

    def test_two_confident_taxa_yield_two_observations(self):
        # Frames that confidently disagree -> one Observation per taxon.
        apis = {"score": 0.85, "common_name": "honey bee",
                "ranks": {"kingdom": "Animalia", "class": "Insecta",
                          "order": "Hymenoptera", "family": "Apidae",
                          "genus": "Apis", "species": "Apis mellifera"}}
        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "region_taxa", return_value=[]), \
                mock.patch.object(pipeline, "_invoke_bioclip",
                                  side_effect=[[_BUMBLEBEE], [apis]]), \
                mock.patch.object(pipeline, "connection"):
            pipeline.classify_frame(self.frames[0].id)
            pipeline.classify_frame(self.frames[1].id)
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.ANALYZED)
        obs = Observation.objects.filter(activity=self.activity)
        self.assertEqual(obs.count(), 2)
        self.assertEqual(set(obs.values_list("taxon__name", flat=True)),
                         {"Bombus impatiens", "Apis mellifera"})
        self.assertEqual(self.activity.best_taxon.name, "Bombus impatiens")  # 0.9 > 0.85

    def test_weak_constrained_falls_back_to_unconstrained(self):
        def fake_invoke(jpeg, candidate_taxa=None):
            if candidate_taxa:                       # constrained -> weak
                return [{**_BUMBLEBEE, "score": 0.05}]
            return [_BUMBLEBEE]                       # unconstrained -> strong

        with mock.patch.object(pipeline, "_read_crop_bytes", return_value=b"jpeg"), \
                mock.patch.object(pipeline, "region_taxa", return_value=["Bombus impatiens"]), \
                mock.patch.object(pipeline, "_invoke_bioclip", side_effect=fake_invoke), \
                mock.patch.object(pipeline, "connection"):
            for fr in self.frames:
                pipeline.classify_frame(fr.id)
        self.activity.refresh_from_db()
        self.assertEqual(self.activity.status, Activity.Status.ANALYZED)
        self.assertAlmostEqual(self.activity.best_confidence, 0.9)  # fallback won


class _FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class PriorsTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_inat_parsed_and_cached(self):
        payload = {"results": [{"taxon": {"name": "Bombus impatiens"}},
                               {"taxon": {"name": "Apis mellifera"}}]}
        with mock.patch("apps.monitor.priors.urllib.request.urlopen",
                        return_value=_FakeResp(payload)) as m:
            taxa = priors.region_taxa(40.80, -77.86, month=6)
            self.assertEqual(taxa, ["Bombus impatiens", "Apis mellifera"])
            priors.region_taxa(40.80, -77.86, month=6)  # served from cache
        self.assertEqual(m.call_count, 1)

    def test_no_coords_returns_empty(self):
        self.assertEqual(priors.region_taxa(None, None), [])

    def test_both_sources_fail_returns_empty(self):
        with mock.patch("apps.monitor.priors.urllib.request.urlopen",
                        side_effect=OSError("net")):
            self.assertEqual(priors.region_taxa(1.0, 2.0), [])
