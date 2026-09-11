"""The review workspace: multi-hotel filtering, triage, day grouping.

What these guard is that the page still answers the question it exists for —
which clips are worth analysing — as the filters get richer.
"""

from datetime import datetime, timezone as dt_tz

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

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
        self.a1 = self._video(self.hotel_a, _dt(3, 13))
        self.a2 = self._video(self.hotel_a, _dt(3, 12))
        self.b1 = self._video(self.hotel_b, _dt(4, 13))
        self.client.force_login(self.user)

    def _video(self, device, when):
        return Video.objects.create(
            user=self.user, device=device, title=f"clip-{when:%d%H}",
            storage_key=f"alice/{when:%d%H}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=when,
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
        r = self.client.get(reverse("analysis:processing"), {"q": "clip-03"})

        counts = {row["obj"].name: row["count"] for row in r.context["device_rows"]}
        self.assertEqual(counts["beemonitor3"], 2)
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

    def test_the_run_controls_follow_the_filters_in_one_scroll_flow(self):
        """Narrow to a set of clips, then act on it — the two read as one
        column, rather than the run block being pushed to the foot of the rail
        with whatever gap is left between them."""
        html = self.client.get(reverse("analysis:processing")).content.decode()

        apply_at = html.index(">Apply</button>")
        picker_at = html.index('name="pipeline" form="run-form"')
        grid_at = html.index('id="grid-scroll"')

        self.assertLess(apply_at, picker_at, "run controls must come after Apply")
        self.assertLess(picker_at, grid_at, "run controls must stay in the rail")

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


class NoPreRunEstimateTests(ReviewHubTestCase):
    """Nothing forecasts a cost before a run.

    The figure this replaced was a hardcoded 349 credits/clip that had been
    wrong for months, because nothing ever checked it against reality. A
    projection shown up front reads as a promise; what a run took is reported
    afterwards, as time.
    """

    def test_the_run_bar_shows_a_count_not_a_price(self):
        html = self.client.get(reverse("analysis:processing")).content.decode()

        self.assertIn('id="cred-line"', html)
        self.assertNotIn("/clip on", html)
        self.assertNotIn("past run", html)

    def test_pipeline_options_carry_no_cost_data(self):
        from apps.pipelines.models import Pipeline
        Pipeline.objects.create(user=self.user, title="P", steps=[])

        html = self.client.get(reverse("analysis:processing")).content.decode()

        for attr in ("data-cost", "data-low", "data-high", "data-sample"):
            self.assertNotIn(attr, html)

    def test_the_page_no_longer_computes_an_estimate(self):
        ctx = self.client.get(reverse("analysis:processing")).context

        self.assertNotIn("estimate", ctx)
        self.assertNotIn("pipeline_estimates", ctx)
