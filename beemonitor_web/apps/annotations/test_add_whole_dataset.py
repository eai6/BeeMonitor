"""Adding clips from a device's whole history, not just the newest page:
a date range that shows where footage is, paging back through older clips,
even spreads across dates, and a capped, bulk add."""

from datetime import datetime, timedelta, timezone as dt_tz
from unittest import mock

from django.contrib.auth import get_user_model
from django.http import QueryDict
from django.test import TestCase
from django.urls import reverse

from apps.annotations import coverage, views as ann_views
from apps.annotations.models import AnnotationProject, FrameSamplingTask
from apps.devices.models import Device
from apps.videos import workspace
from apps.videos.models import Video

User = get_user_model()


class WholeDatasetTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("whole", password="x")
        self.dev = Device.objects.create(owner=self.user, name="Jill", key_hash="hj", prefix="bmk_j")
        self.project = AnnotationProject.objects.create(user=self.user, name="P", classes=["bee"])
        self.client.force_login(self.user)
        self.n = 0

    def clip(self, when, device=None):
        self.n += 1
        return Video.objects.create(
            user=self.user, device=device or self.dev, title=f"c{self.n}",
            storage_key=f"w/{self.n}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=when)

    def day(self, m, d, h=12):
        return datetime(2026, m, d, h, tzinfo=dt_tz.utc)


class DateRangeTests(WholeDatasetTestCase):
    def test_a_to_date_includes_the_whole_day(self):
        last = self.clip(self.day(8, 31, 18))
        self.clip(self.day(9, 1, 1))
        qs = workspace.apply_video_filters(Video.objects.all(),
                                           QueryDict("from=2026-08-01&to=2026-08-31"))
        self.assertEqual(list(qs), [last])

    def test_overview_spans_all_footage_with_month_chips(self):
        for m, d in ((6, 2), (6, 20), (8, 5), (8, 5)):
            self.clip(self.day(m, d))
        ov = workspace.date_overview(Video.objects.all())
        self.assertEqual((ov["first"], ov["last"], ov["total"]), ("2026-06-02", "2026-08-05", 4))
        self.assertEqual([(m["label"], m["n"]) for m in ov["months"]], [("Jun", 2), ("Aug", 2)])
        self.assertEqual(ov["months"][1]["to"], "2026-08-31")
        self.assertEqual(sum(b["n"] for b in ov["bars"]), 4)

    def test_the_page_shows_the_range_control(self):
        self.clip(self.day(6, 2))
        html = self.client.get(reverse("annotations:add_videos_page", args=[self.project.pk])).content.decode()
        self.assertIn("Recorded between", html)
        self.assertIn('data-from="2026-06-01" data-to="2026-06-30"', html)


class SpreadTests(WholeDatasetTestCase):
    def test_draft_spreads_a_cell_across_its_dates(self):
        clips = [self.clip(self.day(6, 1) + timedelta(days=i)) for i in range(30)]
        picks = coverage.draft(Video.accessible(self.user), self.project.videos.all(), per_cell=2)
        dates = sorted(p["date"] for p in picks)
        self.assertEqual(len(picks), 2)
        self.assertLess(dates[0], "2026-06-15")
        self.assertGreater(dates[1], "2026-06-15")
        self.assertTrue({"id", "device", "hour", "date"} <= set(picks[0]))
        del clips

    def test_even_spread_hits_the_target_and_skips_the_project(self):
        other = Device.objects.create(owner=self.user, name="Dan", key_hash="hd", prefix="bmk_d")
        for i in range(40):
            self.clip(self.day(6, 1, 8) + timedelta(days=i))
            self.clip(self.day(6, 1, 15) + timedelta(days=i), device=other)
        held = self.clip(self.day(9, 1, 8))
        self.project.videos.add(held)
        picks = coverage.even_spread(Video.accessible(self.user), self.project.videos.all(), 20)
        self.assertEqual(len(picks), 20)
        self.assertNotIn(held.pk, [p["id"] for p in picks])
        self.assertEqual({p["device"] for p in picks}, {self.dev.pk, other.pk})

    def test_even_spread_takes_everything_when_under_target(self):
        for i in range(5):
            self.clip(self.day(6, 1) + timedelta(days=i))
        self.assertEqual(len(coverage.even_spread(Video.accessible(self.user),
                                                  self.project.videos.all(), 1000)), 5)


class PagingTests(WholeDatasetTestCase):
    def test_older_pages_load_until_the_history_ends(self):
        for i in range(5):
            self.clip(self.day(6, 1) + timedelta(days=i))
        url = reverse("annotations:add_videos_grid", args=[self.project.pk])
        with mock.patch.object(ann_views, "ADD_PAGE_SIZE", 2):
            first = self.client.get(url + "?offset=2").json()
            last = self.client.get(url + "?offset=4").json()
        self.assertEqual((first["count"], first["next_offset"], first["has_more"]), (2, 4, True))
        self.assertEqual((last["count"], last["has_more"]), (1, False))
        self.assertIn("vid-card", first["html"])


class AddTests(WholeDatasetTestCase):
    def post(self, **data):
        with mock.patch("apps.annotations.sampling.spawn_sampling_async"):
            return self.client.post(reverse("annotations:add_videos", args=[self.project.pk]), data)

    def test_on_screen_and_off_screen_picks_are_added_and_sampled_once(self):
        a, b, c = (self.clip(self.day(6, d)) for d in (1, 2, 3))
        self.project.videos.add(c)
        self.post(video_ids=[a.pk], extra_ids=f"{b.pk},{c.pk}")
        self.assertEqual(set(self.project.videos.values_list("pk", flat=True)), {a.pk, b.pk, c.pk})
        self.assertEqual(set(FrameSamplingTask.objects.values_list("video_id", flat=True)), {a.pk, b.pk})

    def test_more_than_the_cap_is_refused(self):
        a = self.clip(self.day(6, 1))
        with mock.patch.object(ann_views, "ADD_CAP", 1):
            r = self.post(video_ids=[a.pk], extra_ids="999999")
        self.assertEqual(r.status_code, 302)
        self.assertEqual(self.project.videos.count(), 0)
