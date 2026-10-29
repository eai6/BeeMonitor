"""Analyse part of a clip (memory/53): a 34-minute clip, events from 5 minutes.

The run panel's "from m:ss for N minutes" reaches the GPU job as
start_seconds/end_seconds, the worker gets the matching frames (split into
pieces inside the window when it's long), credits count only the part, and
the result says which part it covers.
"""

import os
from decimal import Decimal
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.accounts import quota
from apps.accounts.models import UserProfile
from apps.analysis import views as analysis_views
from apps.analysis.models import Job
from apps.pipelines.models import Pipeline, PipelineRun
from apps.pipelines.views import part_of_clip
from apps.videos.models import Video
from apps.videos.timecode import fmt_clock, parse_clock, part_label

User = get_user_model()
STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "d", "block_type": "detect.bee", "config": {}, "inputs": {"video": "v"}},
]


class ClockTests(TestCase):
    def test_times_people_write(self):
        self.assertEqual([parse_clock(t) for t in ("0:00", "2:30", "90", "1:02:05")], [0, 150, 90, 3725])
        self.assertEqual([fmt_clock(s) for s in (0, 150, 3725)], ["0:00", "2:30", "1:02:05"])
        with self.assertRaises(ValueError):
            parse_clock("two minutes")

    def test_the_run_panel_choice(self):
        self.assertIsNone(part_of_clip({"part": "whole"}))
        self.assertEqual(part_of_clip({"part": "part", "part_from": "2:00", "part_minutes": "5"}), (120, 300))
        self.assertEqual(part_of_clip({"part": "part", "part_from": "10:00", "part_minutes": "end"}), (600, None))
        self.assertIsNone(part_of_clip({"part": "part", "part_from": "0:00", "part_minutes": "end"}))


class LaunchTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="p", steps=STEPS)
        self.clip = self.video(34 * 60 + 32)

    def video(self, seconds, n=0):
        return Video.objects.create(user=self.user, title=f"c{n}", storage_key=f"k/{n}.mp4",
                                    file_size_bytes=1, status=Video.Status.READY,
                                    duration_seconds=seconds, fps=30)

    def launch(self, ids, **part):
        with mock.patch("apps.analysis.views._drain_queue"):
            return self.client.post(reverse("pipelines:run_on_videos"),
                                    {"pipeline": self.pipeline.pk, "video_ids": ids, **part},
                                    HTTP_X_REQUESTED_WITH="XMLHttpRequest")

    def test_five_minutes_reach_the_gpu_job(self):
        r = self.launch([self.clip.pk], part="part", part_from="2:00", part_minutes="5")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertIn("2:00–7:00", r.json()["message"])
        job = Job.objects.get()
        self.assertEqual((job.config["start_seconds"], job.config["end_seconds"]), (120.0, 420.0))
        run = PipelineRun.objects.get()
        self.client.force_login(self.user)
        page = self.client.get(reverse("pipelines:run_detail", args=[self.pipeline.pk, run.pk]))
        self.assertContains(page, "Analysed 2:00–7:00 of 34:32")

    def test_the_whole_clip_has_no_window(self):
        self.launch([self.clip.pk], part="whole")
        self.assertNotIn("start_seconds", Job.objects.get().config)

    def test_a_window_past_the_end_stops_at_the_end(self):
        self.launch([self.clip.pk], part="part", part_from="32:00", part_minutes="5")
        cfg = Job.objects.get().config
        self.assertEqual(cfg["start_seconds"], 1920.0)
        self.assertNotIn("end_seconds", cfg)

    def test_clips_that_end_before_the_start_are_left_out(self):
        short = self.video(60, n=1)
        r = self.launch([self.clip.pk, short.pk], part="part", part_from="10:00", part_minutes="1")
        self.assertIn("Left out 1", r.json()["message"])
        self.assertEqual(list(Job.objects.values_list("video_id", flat=True)), [self.clip.pk])

    def test_credits_count_only_the_part(self):
        # 25 credits: 34 minutes fits, 100 hours doesn't — but 5 minutes of it does.
        big = self.video(100 * 3600, n=2)
        self.assertEqual(self.launch([big.pk], part="whole").status_code, 402)
        self.assertEqual(self.launch([big.pk], part="part", part_from="0:00", part_minutes="5").status_code, 200)
        self.assertLess(quota.pending(self.user), Decimal("0.2"))

    def test_a_bad_time_is_refused(self):
        r = self.launch([self.clip.pk], part="part", part_from="soon", part_minutes="5")
        self.assertEqual(r.status_code, 400)
        self.assertFalse(Job.objects.exists())

    def test_the_api_takes_seconds(self):
        from rest_framework.test import APIClient
        api = APIClient()
        api.force_authenticate(self.user)
        r = api.post(f"/api/v1/pipelines/{self.pipeline.pk}/run/",
                     {"video_ids": [self.clip.pk], "start_seconds": 60, "end_seconds": 120}, format="json")
        self.assertEqual(r.status_code, 202, r.content)
        cfg = Job.objects.get().config
        self.assertEqual((cfg["start_seconds"], cfg["end_seconds"]), (60.0, 120.0))


class WorkerPayloadTests(TestCase):
    def setUp(self):
        user = User.objects.create_user("bee")
        self.clip = Video.objects.create(user=user, title="c", storage_key="k/c.mp4", file_size_bytes=1,
                                         duration_seconds=34 * 60, fps=30)
        self.sent = []

    def launch(self, payload):
        def put(job_id, p):
            self.sent.append(p)
            return f"s3://in/{job_id}"
        with mock.patch.object(analysis_views, "_put_inference_payload", side_effect=put), \
                mock.patch.object(analysis_views, "_invoke_endpoint_async", return_value=("s3://out", "")):
            return analysis_views._launch_gpu({"job_id": "j", **payload}, self.clip, "j")

    def test_one_invocation_gets_the_frames(self):
        with mock.patch.dict(os.environ, {"BEEMONITOR_CHUNK_TRACKING": ""}):
            self.launch({"start_seconds": 120, "end_seconds": 420})
        self.assertEqual((self.sent[0]["start_frame"], self.sent[0]["end_frame"]), (3600, 12600))

    def test_a_long_window_is_split_inside_the_window(self):
        with mock.patch.dict(os.environ, {"BEEMONITOR_CHUNK_TRACKING": "1"}):
            _, cfg = self.launch({"start_seconds": 60, "end_seconds": 60 + 1500})  # 25 min > 20-min limit
        frames = [(p["start_frame"], p.get("end_frame")) for p in self.sent]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0][0], 1800)          # starts at 1:00
        self.assertEqual(frames[0][1], frames[1][0])  # no gap, no overlap
        self.assertEqual(frames[1][1], 46800)         # stops at 26:00, not the end of the clip
        self.assertEqual(len(cfg["chunks"]), 2)

    def test_the_whole_clip_sends_no_frames(self):
        with mock.patch.dict(os.environ, {"BEEMONITOR_CHUNK_TRACKING": ""}):
            self.launch({})
        self.assertNotIn("start_frame", self.sent[0])

    def test_the_label(self):
        self.assertEqual(part_label({"start_seconds": 0, "end_seconds": 300}, self.clip), "Analysed 0:00–5:00 of 34:00")
        self.assertEqual(part_label({}, self.clip), "")


class PicksAcrossFiltersTests(LaunchTests):
    """The Videos page keeps picks across filter changes and sends them with a
    run; "Select all" under a filter runs that filter's clips and the picks."""

    def test_select_all_runs_the_filter_and_picks_from_other_filters(self):
        mendel = [self.video(60, n) for n in (1, 2)]
        for v in mendel:
            v.title = f"mendels {v.pk}"
            v.save()
        other = self.video(60, 9)   # picked earlier, under another filter
        with mock.patch("apps.analysis.views._drain_queue"):
            r = self.client.post(reverse("pipelines:run_on_videos"),
                                 {"pipeline": self.pipeline.pk, "all_filtered": "1",
                                  "q": "mendels", "video_ids": [other.pk]},
                                 HTTP_X_REQUESTED_WITH="XMLHttpRequest")
        self.assertEqual(r.status_code, 200, r.content)
        self.assertEqual(sorted(r.json()["video_ids"]), sorted([v.pk for v in mendel] + [other.pk]))

    def test_picks_alone_run_just_those(self):
        a, b = self.video(60, 3), self.video(60, 4)
        r = self.launch([a.pk, b.pk], part="whole")
        self.assertEqual(sorted(r.json()["video_ids"]), sorted([a.pk, b.pk]))

    def test_the_page_keeps_picks_for_the_tab(self):
        page = self.client.get(reverse("analysis:processing")).content.decode()
        self.assertIn("bm-hub-picks-", page)
        self.assertIn("rail-elsewhere", page)
