"""Filtering the project's own video list.

A flat list of every video stops being usable past a few dozen — this project
already has 45. These pin the filter dimensions the steps above act on, and the
query-count fix, which used to be one COUNT per video.
"""

from datetime import timedelta

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.utils import timezone

from apps.annotations.models import Annotation, AnnotationProject
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()


class VideoFilterTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.jill = Device.objects.create(owner=self.user, name="Jill",
                                          key_hash="h1", prefix="p1")
        self.dani = Device.objects.create(owner=self.user, name="Danniella",
                                          key_hash="h2", prefix="p2")
        self.project = AnnotationProject.objects.create(
            user=self.user, name="Summer_2026", classes=["bee"])
        base = timezone.now().replace(year=2026, month=7, day=8, hour=18,
                                      minute=0, second=0, microsecond=0)
        self.videos = {}
        for name, device, site, hour, day in (
            ("jill_evening", self.jill, "Location1", 18, 8),
            ("jill_morning", self.jill, "Location1", 9, 8),
            ("jill_other_day", self.jill, "Location1", 18, 3),
            ("dani_evening", self.dani, "Location2", 18, 8),
        ):
            v = Video.objects.create(
                user=self.user, device=device, title=name, site_name=site,
                storage_key=f"a/{name}.mp4", file_size_bytes=1,
                status=Video.Status.READY,
                recorded_at=base.replace(day=day, hour=hour),
                year=2026, month=7, day=day, hour=hour)
            self.project.videos.add(v)
            self.videos[name] = v
        # One video has frames; the rest don't.
        Annotation.objects.create(project=self.project,
                                  video=self.videos["jill_evening"],
                                  frame_number=0, boxes=[])
        self.client.force_login(self.user)

    def _titles(self, **params):
        qs = {f"v_{k}": v for k, v in params.items()}
        resp = self.client.get(f"/annotations/{self.project.pk}/", qs)
        return [d["video"].title for d in resp.context["video_data"]]

    def test_unfiltered_shows_everything(self):
        self.assertEqual(len(self._titles()), 4)

    def test_filter_by_device(self):
        self.assertEqual(sorted(self._titles(device=str(self.dani.pk))),
                         ["dani_evening"])

    def test_filter_by_location(self):
        self.assertEqual(sorted(self._titles(site="Location2")), ["dani_evening"])

    def test_filter_by_day(self):
        self.assertEqual(sorted(self._titles(day="3")), ["jill_other_day"])

    def test_filter_by_hour_window(self):
        """The workflow driver: 'everything recorded in the evening'."""
        self.assertEqual(sorted(self._titles(hfrom="17", hto="20")),
                         ["dani_evening", "jill_evening", "jill_other_day"])

    def test_hour_window_wraps_past_midnight(self):
        self.assertEqual(sorted(self._titles(hfrom="17", hto="10")),
                         ["dani_evening", "jill_evening", "jill_morning",
                          "jill_other_day"])

    def test_filter_by_annotation_state(self):
        self.assertEqual(self._titles(state="annotated"), ["jill_evening"])
        self.assertEqual(len(self._titles(state="unannotated")), 3)

    def test_filters_combine(self):
        self.assertEqual(sorted(self._titles(device=str(self.jill.pk),
                                             hfrom="17", hto="20", day="8")),
                         ["jill_evening"])

    def test_search_by_title(self):
        self.assertEqual(sorted(self._titles(q="morning")), ["jill_morning"])

    def test_no_matches_keeps_the_card_so_you_can_reset(self):
        resp = self.client.get(f"/annotations/{self.project.pk}/", {"v_q": "nope"})
        self.assertEqual(resp.context["video_data"], [])
        self.assertTrue(resp.context["video_filter_on"])
        self.assertIn("No videos match this filter", resp.content.decode())
        self.assertIn("Reset", resp.content.decode())

    def test_counts_reported_for_the_header(self):
        resp = self.client.get(f"/annotations/{self.project.pk}/", {"v_state": "annotated"})
        self.assertEqual(resp.context["video_count"], 4)          # project total
        self.assertEqual(resp.context["video_annotated_count"], 1)
        self.assertEqual(resp.context["video_filtered_count"], 1)

    def test_dropdowns_only_offer_values_present_in_the_project(self):
        resp = self.client.get(f"/annotations/{self.project.pk}/")
        opts = resp.context["video_filter_opts"]
        self.assertEqual(sorted(n for _pk, n in opts["devices"]), ["Danniella", "Jill"])
        self.assertEqual(opts["sites"], ["Location1", "Location2"])
        self.assertEqual(opts["years"], [2026])
        self.assertEqual(sorted(opts["days"]), [3, 8])

    def test_frame_counts_do_not_scale_queries_with_video_count(self):
        """Was one COUNT per video inside a loop — 45 queries for 45 videos.

        Asserts the shape, not an exact number: adding 20 more videos must not
        add ~20 more queries.
        """
        from django.test.utils import CaptureQueriesContext
        from django.db import connection

        url = f"/annotations/{self.project.pk}/"
        with CaptureQueriesContext(connection) as small:
            self.client.get(url)

        for i in range(20):
            v = Video.objects.create(user=self.user, device=self.jill,
                                     title=f"extra{i}", storage_key=f"a/x{i}.mp4",
                                     file_size_bytes=1, status=Video.Status.READY)
            self.project.videos.add(v)

        with CaptureQueriesContext(connection) as large:
            resp = self.client.get(url)

        self.assertEqual(len(resp.context["video_data"]), 24)
        # 6x the videos; allow a couple of extra queries for unrelated widgets,
        # but nothing like one-per-video.
        self.assertLess(len(large), len(small) + 5,
                        f"query count grew {len(small)} -> {len(large)} with 20 more videos")
