"""The review workspace: multi-hotel filtering, triage, day grouping, estimate.

What these guard is that the page still answers the question it exists for —
which clips are worth analysing — as the filters get richer.
"""

from datetime import datetime, timezone as dt_tz

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analysis import pricing
from apps.analysis.models import Job
from apps.analysis.views import apply_video_filters
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()


def _dt(day, hour):
    return datetime(2026, 8, day, hour, 30, tzinfo=dt_tz.utc)


class ReviewHubTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.hotel_a = Device.objects.create(owner=self.user, name="beemonitor3",
                                             key_hash="h1", prefix="bmk_1")
        self.hotel_b = Device.objects.create(owner=self.user, name="beemonitor1",
                                             key_hash="h2", prefix="bmk_2")
        self.hotel_c = Device.objects.create(owner=self.user, name="hotel-north",
                                             key_hash="h3", prefix="bmk_3")
        self.a1 = self._video(self.hotel_a, _dt(3, 13), confirmed=True)
        self.a2 = self._video(self.hotel_a, _dt(3, 12))
        self.b1 = self._video(self.hotel_b, _dt(4, 13))
        self.client.force_login(self.user)

    def _video(self, device, when, confirmed=False):
        # A real upload writes both: `bee_confirmed` (flat, for filters) and
        # `bee` (rich, for the badge) — apps/api/uploads.py:205.
        meta = {"bee_confirmed": bool(confirmed),
                "bee": {"status": "confirmed" if confirmed else "unconfirmed",
                        "confidence": 0.54 if confirmed else 0.0}}
        return Video.objects.create(
            user=self.user, device=device, title=f"clip-{when:%d%H}",
            storage_key=f"alice/{when:%d%H}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=when, metadata=meta,
        )


class MultiDeviceFilterTests(ReviewHubTestCase):
    def test_several_hotels_can_be_reviewed_at_once(self):
        r = self.client.get(reverse("analysis:processing"),
                            {"device": [str(self.hotel_a.id), str(self.hotel_b.id)]})

        html = r.content.decode()
        for v in (self.a1, self.a2, self.b1):
            self.assertIn(f'data-vid="{v.pk}"', html)

    def test_one_hotel_still_narrows_to_it(self):
        r = self.client.get(reverse("analysis:processing"), {"device": str(self.hotel_b.id)})

        html = r.content.decode()
        self.assertIn(f'data-vid="{self.b1.pk}"', html)
        self.assertNotIn(f'data-vid="{self.a1.pk}"', html)

    def test_a_plain_dict_with_one_device_still_works(self):
        """The per-device scheduler passes a plain dict, not a QueryDict."""
        qs = apply_video_filters(Video.objects.all(), {"device": str(self.hotel_a.id)})

        self.assertEqual(set(qs.values_list("pk", flat=True)), {self.a1.pk, self.a2.pk})

    def test_no_device_means_every_hotel(self):
        qs = apply_video_filters(Video.objects.all(), {})

        self.assertEqual(qs.count(), 3)


class HotelCountTests(ReviewHubTestCase):
    def test_counts_ignore_the_hotel_selection_itself(self):
        """A count answers 'what would ticking this add', so selecting one
        hotel must not zero the others."""
        r = self.client.get(reverse("analysis:processing"), {"device": str(self.hotel_a.id)})

        counts = {row["obj"].name: row["count"] for row in r.context["device_rows"]}
        self.assertEqual(counts["beemonitor3"], 2)
        self.assertEqual(counts["beemonitor1"], 1)

    def test_counts_do_respect_the_other_filters(self):
        r = self.client.get(reverse("analysis:processing"), {"confirmed": "yes"})

        counts = {row["obj"].name: row["count"] for row in r.context["device_rows"]}
        self.assertEqual(counts["beemonitor3"], 1)
        self.assertEqual(counts["beemonitor1"], 0)

    def test_an_empty_hotel_is_listed_with_zero_not_hidden(self):
        r = self.client.get(reverse("analysis:processing"))

        names = [row["obj"].name for row in r.context["device_rows"]]
        counts = {row["obj"].name: row["count"] for row in r.context["device_rows"]}
        self.assertIn("hotel-north", names)
        self.assertEqual(counts["hotel-north"], 0)

    def test_each_hotel_gets_a_distinct_dot(self):
        r = self.client.get(reverse("analysis:processing"))

        dots = [row["dot"] for row in r.context["device_rows"]]
        self.assertEqual(len(dots), len(set(dots)))


class TriageTests(ReviewHubTestCase):
    def test_counts_split_confirmed_from_unconfirmed(self):
        r = self.client.get(reverse("analysis:processing"))

        self.assertEqual(r.context["triage"]["confirmed"], 1)
        self.assertEqual(r.context["triage"]["unconfirmed"], 2)

    def test_never_analyzed_counts_clips_without_a_completed_job(self):
        Job.objects.create(user=self.user, video=self.a1, status="completed",
                           modal_job_id="j1")

        r = self.client.get(reverse("analysis:processing"))

        self.assertEqual(r.context["triage"]["unanalyzed"], 2)

    def test_the_never_analyzed_filter_excludes_finished_clips(self):
        Job.objects.create(user=self.user, video=self.a1, status="completed",
                           modal_job_id="j1")

        qs = apply_video_filters(Video.objects.all(), {"analysis": "never"})

        self.assertNotIn(self.a1.pk, set(qs.values_list("pk", flat=True)))
        self.assertEqual(qs.count(), 2)

    def test_a_failed_job_still_counts_as_never_analyzed(self):
        Job.objects.create(user=self.user, video=self.a1, status="failed",
                           modal_job_id="j1")

        qs = apply_video_filters(Video.objects.all(), {"analysis": "never"})

        self.assertIn(self.a1.pk, set(qs.values_list("pk", flat=True)))


class DayGroupingTests(ReviewHubTestCase):
    def test_clips_are_grouped_by_recorded_day_newest_first(self):
        r = self.client.get(reverse("analysis:processing"))

        days = [g["day"].day for g in r.context["video_days"]]
        self.assertEqual(days, [4, 3])

    def test_a_group_knows_how_many_hotels_it_spans(self):
        extra = self._video(self.hotel_b, _dt(3, 14))

        r = self.client.get(reverse("analysis:processing"))

        aug3 = [g for g in r.context["video_days"] if g["day"].day == 3][0]
        self.assertEqual(len(aug3["hotels"]), 2)
        self.assertIn(extra, aug3["videos"])

    def test_every_clip_carries_its_hotel_dot(self):
        r = self.client.get(reverse("analysis:processing"))

        for group in r.context["video_days"]:
            for video in group["videos"]:
                self.assertTrue(video.dot.startswith("#"))


class EstimateTests(ReviewHubTestCase):
    """What a run will cost, estimated from COMPARABLE finished runs.

    The first version of this averaged everything in the Job table and priced it
    all on the T4, which produced ~$0.001 a clip: pre-annotation tasks finish in
    seconds and dragged the median down, and SAM 3 runs on a 1.9x dearer
    instance were priced as if they were YOLO.
    """

    def _job(self, **kw):
        kw.setdefault("user", self.user)
        kw.setdefault("video", self.a1)
        kw.setdefault("status", "completed")
        kw.setdefault("modal_job_id", f"j{Job.objects.count()}")
        kw.setdefault("config", {})
        return Job.objects.create(**kw)

    def test_no_history_reports_no_sample_rather_than_a_made_up_number(self):
        r = self.client.get(reverse("analysis:processing"))

        self.assertEqual(r.context["estimate"]["sample"], 0)
        self.assertEqual(r.context["estimate"]["cost"], 0.0)

    def test_the_estimate_is_the_median_of_real_runs(self):
        for secs in (60, 90, 120):
            self._job(execution_seconds=secs)

        est = pricing.estimate_per_video(self.user)

        self.assertEqual(est["sample"], 3)
        self.assertEqual(est["seconds"], 90.0)
        self.assertLess(est["cost_low"], est["cost"])
        self.assertGreater(est["cost_high"], 0)

    def test_unfinished_runs_do_not_skew_it(self):
        self._job(execution_seconds=90)
        self._job(video=self.a2, status="processing", execution_seconds=0)

        self.assertEqual(pricing.estimate_per_video(self.user)["sample"], 1)

    def test_pre_annotation_tasks_are_excluded(self):
        """They sample a handful of frames — seconds, not minutes — and were
        what pulled the figure down to a tenth of a cent."""
        self._job(execution_seconds=300)
        for _ in range(8):
            self._job(execution_seconds=4, config={"task": "pre_annotate"})

        est = pricing.estimate_per_video(self.user)

        self.assertEqual(est["sample"], 1)
        self.assertEqual(est["seconds"], 300.0)

    def test_annotated_video_renders_are_excluded_too(self):
        self._job(execution_seconds=300)
        self._job(execution_seconds=6, config={"task": "annotate_video"})

        self.assertEqual(pricing.estimate_per_video(self.user)["sample"], 1)

    def test_sam3_and_yolo_history_do_not_contaminate_each_other(self):
        self._job(execution_seconds=100, config={"detector_kind": "yolo"})
        self._job(execution_seconds=900, config={"detector_kind": "sam3"})

        yolo = pricing.estimate_per_video(self.user, detector_kind="yolo")
        sam3 = pricing.estimate_per_video(self.user, detector_kind="sam3")

        self.assertEqual(yolo["seconds"], 100.0)
        self.assertEqual(sam3["seconds"], 900.0)

    def test_a_missing_detector_kind_counts_as_yolo(self):
        self._job(execution_seconds=100)

        self.assertEqual(pricing.estimate_per_video(self.user, detector_kind="yolo")["sample"], 1)
        self.assertEqual(pricing.estimate_per_video(self.user, detector_kind="sam3")["sample"], 0)

    def test_sam3_is_priced_on_the_g5_not_the_t4(self):
        """Routing sends SAM 3 to the g5; pricing it on the T4 understates by
        the ratio of the two rates."""
        self._job(execution_seconds=100, config={"detector_kind": "sam3"})
        self._job(execution_seconds=100, config={"detector_kind": "yolo"})

        sam3 = pricing.estimate_per_video(self.user, detector_kind="sam3")
        yolo = pricing.estimate_per_video(self.user, detector_kind="yolo")

        self.assertEqual(sam3["instance"], "ml.g5.xlarge")
        self.assertEqual(yolo["instance"], "ml.g4dn.xlarge")
        self.assertGreater(sam3["cost"], yolo["cost"])
        self.assertAlmostEqual(sam3["cost"] / yolo["cost"], 1.408 / 0.7364, places=2)


class PipelineDetectorTests(TestCase):
    def test_a_sam3_detect_step_makes_the_pipeline_sam3(self):
        from apps.pipelines.models import Pipeline

        pl = Pipeline(steps=[{"block_type": "detect.objects",
                              "config": {"model_family": "sam3"}}])

        self.assertEqual(pricing.pipeline_detector_kind(pl), "sam3")

    def test_the_older_detector_key_is_honoured(self):
        from apps.pipelines.models import Pipeline

        pl = Pipeline(steps=[{"block_type": "detect.objects",
                              "config": {"detector": "sam3"}}])

        self.assertEqual(pricing.pipeline_detector_kind(pl), "sam3")

    def test_anything_else_is_yolo(self):
        from apps.pipelines.models import Pipeline

        for steps in ([], [{"block_type": "input.video", "config": {}}],
                      [{"block_type": "detect.objects", "config": {}}]):
            self.assertEqual(pricing.pipeline_detector_kind(Pipeline(steps=steps)), "yolo")


class PerPipelineEstimateTests(ReviewHubTestCase):
    def test_each_pipeline_carries_its_own_estimate(self):
        from apps.pipelines.models import Pipeline

        Job.objects.create(user=self.user, video=self.a1, status="completed",
                           execution_seconds=100, modal_job_id="jy", config={})
        Job.objects.create(user=self.user, video=self.a1, status="completed",
                           execution_seconds=900, modal_job_id="js",
                           config={"detector_kind": "sam3"})
        yolo_pl = Pipeline.objects.create(user=self.user, title="Y", steps=[
            {"block_type": "detect.objects", "config": {"model_family": "yolo"}}])
        sam3_pl = Pipeline.objects.create(user=self.user, title="S", steps=[
            {"block_type": "detect.objects", "config": {"model_family": "sam3"}}])

        est = self.client.get(reverse("analysis:processing")).context["pipeline_estimates"]

        self.assertEqual(est[str(yolo_pl.pk)]["seconds"], 100.0)
        self.assertEqual(est[str(sam3_pl.pk)]["seconds"], 900.0)
        self.assertGreater(est[str(sam3_pl.pk)]["cost"], est[str(yolo_pl.pk)]["cost"])


class StatusPillTests(ReviewHubTestCase):
    def test_running_is_not_the_same_colour_as_analyzed(self):
        """The palette rename made amber resolve to green, so the old running
        badge was pixel-identical to the finished one."""
        Job.objects.create(user=self.user, video=self.a1, status="completed",
                           modal_job_id="j1")
        Job.objects.create(user=self.user, video=self.b1, status="processing",
                           modal_job_id="j2")

        html = self.client.get(reverse("analysis:processing")).content.decode()

        self.assertIn("bg-green-100 text-green-700", html)
        self.assertIn("bg-blue-100 text-blue-700", html)
        self.assertNotIn("bg-amber-100 text-amber-700", html)


class RunControlsInTheRailTests(ReviewHubTestCase):
    """The pipeline picker and Run live in the rail, bound across the DOM.

    HTML forms cannot nest, so the controls sit inside the rail's markup while
    belonging to the POST form in <main> via form="run-form". If that attribute
    is ever dropped, the button silently submits the GET filter form instead —
    the page would look fine and running would do nothing.
    """

    def test_the_pipeline_select_belongs_to_the_run_form(self):
        html = self.client.get(reverse("analysis:processing")).content.decode()

        self.assertIn('name="pipeline" form="run-form"', html)

    def test_the_run_button_belongs_to_the_run_form(self):
        html = self.client.get(reverse("analysis:processing")).content.decode()

        self.assertIn('type="submit" form="run-form" id="run-btn"', html)

    def test_the_filter_form_keeps_its_own_apply(self):
        html = self.client.get(reverse("analysis:processing")).content.decode()

        self.assertIn('<form method="get" id="filter-form"', html)
        self.assertIn(">Apply</button>", html)

    def test_a_viewer_gets_the_filters_but_no_run_controls(self):
        viewer = User.objects.create_user("carol", password="x")
        from apps.devices.models import DeviceShare
        DeviceShare.objects.create(device=self.hotel_a, user=viewer,
                                   role=DeviceShare.Role.VIEWER)
        self.client.force_login(viewer)

        html = self.client.get(reverse("analysis:processing")).content.decode()

        self.assertIn('id="filter-form"', html)
        self.assertNotIn('id="run-btn"', html)


class TrackingEndpointRoutingTests(TestCase):
    """SAM 3 must not silently land on the T4.

    _tracking_endpoint used to fall back to the default endpoint when the SAM 3
    one was unconfigured — which is exactly what happened in production: the
    variable was never set on App Runner, so every SAM 3 run went to the g4dn
    and died deep inside inference with an error that said nothing about
    routing. Refusing at submit names the actual problem.
    """

    def test_yolo_uses_the_default_endpoint(self):
        from apps.analysis.views import _tracking_endpoint

        with self.settings(SAGEMAKER_ENDPOINT_NAME="main", SAGEMAKER_SAM3_ENDPOINT_NAME="sam3"):
            self.assertEqual(_tracking_endpoint("yolo"), "main")

    def test_sam3_uses_its_own_endpoint(self):
        from apps.analysis.views import _tracking_endpoint

        with self.settings(SAGEMAKER_ENDPOINT_NAME="main", SAGEMAKER_SAM3_ENDPOINT_NAME="sam3"):
            self.assertEqual(_tracking_endpoint("sam3"), "sam3")

    def test_sam3_without_its_endpoint_refuses_instead_of_using_the_t4(self):
        from apps.analysis.views import _tracking_endpoint

        with self.settings(SAGEMAKER_ENDPOINT_NAME="main", SAGEMAKER_SAM3_ENDPOINT_NAME=""):
            with self.assertRaises(RuntimeError) as caught:
                _tracking_endpoint("sam3")

        self.assertIn("SAGEMAKER_SAM3_ENDPOINT_NAME", str(caught.exception))

    def test_yolo_is_unaffected_by_a_missing_sam3_endpoint(self):
        from apps.analysis.views import _tracking_endpoint

        with self.settings(SAGEMAKER_ENDPOINT_NAME="main", SAGEMAKER_SAM3_ENDPOINT_NAME=""):
            self.assertEqual(_tracking_endpoint("yolo"), "main")


class ViewerNavigationTests(ReviewHubTestCase):
    """Stepping through the viewer moves through TIME, not through indices.

    Cards are ordered newest-first, so "previous" by position walks toward newer
    footage — the opposite of stepping back through a day. The controls are
    named after time so the mapping cannot be read the wrong way round.
    """

    def _html(self):
        return self.client.get(reverse("analysis:processing")).content.decode()

    def test_the_grid_is_newest_first(self):
        """The premise the navigation depends on."""
        videos = self.client.get(reverse("analysis:processing")).context["videos"]
        times = [v.recorded_at for v in videos]

        self.assertEqual(times, sorted(times, reverse=True))

    def test_the_controls_are_labelled_by_time(self):
        html = self._html()

        self.assertIn("← Earlier", html)
        self.assertIn("Later →", html)
        self.assertNotIn("← Prev", html)

    def test_earlier_moves_down_the_newest_first_grid(self):
        html = self._html()

        self.assertIn("function earlier() { open(at + 1); }", html)
        self.assertIn("function later() { open(at - 1); }", html)

    def test_left_arrow_goes_back_in_time(self):
        html = self._html()

        self.assertIn("e.key === 'ArrowLeft' || e.key === 'j'", html)
        self.assertIn("e.key === 'ArrowRight' || e.key === 'k'", html)

    def test_both_buttons_are_wired_to_the_named_moves(self):
        html = self._html()

        self.assertIn("getElementById('v-prev').addEventListener('click', earlier)", html)
        self.assertIn("getElementById('v-next').addEventListener('click', later)", html)
