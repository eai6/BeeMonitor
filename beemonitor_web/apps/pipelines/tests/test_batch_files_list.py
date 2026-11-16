"""The Files list: every clip in a batch, with its times and what became of it
(memory/80).

The other downloads only carry clips that produced rows, so a batch's failed
and cancelled clips, and their recording times, could be seen nowhere but the
page.
"""
import csv
import io
from datetime import datetime, timezone as dt_timezone

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.analysis.models import Job
from apps.pipelines import files_list, review
from apps.pipelines.models import ClipReview, EventReview, Pipeline, PipelineRun
from apps.videos.models import Site, Video

User = get_user_model()
BATCH = "91dc8dc3-0000-4000-8000-0000000000ff"
STEPS = [
    {"id": "v", "block_type": "input.video", "config": {}},
    {"id": "t", "block_type": "track.mot", "config": {}},
]


class FilesListTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("fl", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="Trips")
        self.site = Site.objects.create(user=self.user, name="mendels", tz_name="America/New_York")

    def _run(self, title, when, status, context, source="filename", gpu_seconds=None):
        # A file-name time is stored as the name's wall-clock, labelled UTC; the
        # save places it on the site's zone (memory/60).
        video = Video.objects.create(
            user=self.user, title=title, storage_key=f"fl/{title}.mp4",
            file_size_bytes=150_000_000, status=Video.Status.READY, recorded_at=when,
            duration_seconds=600, fps=30, site=self.site,
            metadata={"recorded_at_source": source})
        steps = [dict(s) for s in STEPS]
        steps[0]["config"] = {"video_id": str(video.pk)}
        if gpu_seconds is not None:
            job = Job.objects.create(user=self.user, video=video, status="completed",
                                     execution_seconds=gpu_seconds)
            context = {**context, "t": {**context.get("t", {}), "job_id": job.pk}}
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH, status=status,
            steps=steps, step_status={"v": "done", "t": "done" if status == "completed" else "failed"},
            context={"v": {"artifact": "video", "video_id": video.pk}, **context})

    def _download(self):
        r = self.client.get(reverse("pipelines:batch_combined_csv", args=[BATCH, "files"]))
        self.assertEqual(r.status_code, 200)
        return list(csv.DictReader(io.StringIO(b"".join(r.streaming_content).decode())))

    def test_every_clip_is_listed_with_why_it_did_not_finish(self):
        self._run("mendels_2024-05-08_15_00_00", datetime(2024, 5, 8, 15, 0, tzinfo=dt_timezone.utc),
                  "completed", {}, gpu_seconds=1800)
        self._run("mendels_2024-04-17_19_00_00", datetime(2024, 4, 17, 19, 0, tzinfo=dt_timezone.utc),
                  "failed", {"t": {"error": "Cannot open video: /tmp/x.mp4"}})
        self._run("mendels_2024-04-20_10_30_00", datetime(2024, 4, 20, 10, 30, tzinfo=dt_timezone.utc),
                  "failed", {"t": {"error": "Cancelled by user."}})

        rows = self._download()

        self.assertEqual([r["video_title"] for r in rows], [
            "mendels_2024-04-17_19_00_00", "mendels_2024-04-20_10_30_00",
            "mendels_2024-05-08_15_00_00"])          # recording order
        broken, cancelled, done = rows
        self.assertEqual(broken["run_status"], "failed")
        self.assertEqual(broken["failure_cause"], "The clip's file can't be played")
        self.assertIn("empty", broken["failure_detail"])
        self.assertIn("will not help", broken["next_step"])
        self.assertEqual(broken["error"], "Cannot open video: /tmp/x.mp4")
        self.assertEqual(cancelled["run_status"], "cancelled")
        self.assertEqual(cancelled["failure_cause"], "Cancelled by you")
        self.assertEqual(done["run_status"], "completed")
        self.assertEqual((done["failure_cause"], done["failure_detail"], done["error"]), ("", "", ""))
        self.assertEqual(done["gpu_hours"], "0.5")

    def test_times_are_the_sites_local_clock_and_say_where_they_came_from(self):
        self._run("mendels_2024-05-08_15_00_00", datetime(2024, 5, 8, 15, 0, tzinfo=dt_timezone.utc),
                  "completed", {})

        row = self._download()[0]

        self.assertEqual((row["date"], row["start_time"], row["end_time"]),
                         ("2024-05-08", "15:00:00", "15:10:00"))
        self.assertEqual(row["time_zone"], "America/New_York")
        self.assertEqual(row["clip_start"], "2024-05-08T15:00:00.000-04:00")
        self.assertEqual(row["time_source"], "filename")
        self.assertEqual(row["filename_time"], "2024-05-08 15:00:00")
        self.assertEqual((row["duration_min"], row["size_mb"]), ("10.0", "150.0"))

    def test_a_start_with_no_zone_is_a_plain_wall_clock(self):
        self.site.tz_name = ""
        self.site.save()
        self._run("mendels_2024-05-08_15_00_00", datetime(2024, 5, 8, 15, 0, tzinfo=dt_timezone.utc),
                  "completed", {})

        row = self._download()[0]

        self.assertEqual(row["time_zone"], "")
        self.assertNotIn("+", row["clip_start"])

    def test_the_batch_page_offers_it(self):
        self._run("mendels_2024-05-08_15_00_00", datetime(2024, 5, 8, 15, 0, tzinfo=dt_timezone.utc),
                  "completed", {})

        r = self.client.get(reverse("pipelines:batch_detail", args=[BATCH]))

        self.assertContains(r, reverse("pipelines:batch_combined_csv", args=[BATCH, "files"]))
        self.assertContains(r, "Files list")

    def test_file_name_times(self):
        self.assertEqual(files_list.filename_time("natalies_2024-05-22_19_50_00"),
                         "2024-05-22 19:50:00")
        self.assertEqual(files_list.filename_time("Jill · 3df2b768c516"), "")

    def test_it_says_which_clips_were_reviewed_and_how(self):
        # memory/84: the clips the metrics are scored over, even one with no
        # events, which no other download lists.
        when = datetime(2024, 5, 8, 10, 0, tzinfo=dt_timezone.utc)
        rows = [{"action": "exit", "reference": "reference_1", "track_id": 1, "frame": 10},
                {"action": "enter", "reference": "reference_1", "track_id": 2, "frame": 90}]
        busy = self._run("natalies_2024-05-08_10_00_00", when, "completed",
                         {"e": {"table_kind": "events", "rows": rows}})
        empty = self._run("natalies_2024-05-08_10_10_00", when.replace(minute=10), "completed",
                          {"e": {"table_kind": "events", "rows": []}})
        self._run("natalies_2024-05-08_10_20_00", when.replace(minute=20), "completed",
                  {"e": {"table_kind": "events", "rows": rows}})
        maria = User.objects.create_user("maria", password="x")
        EventReview.objects.create(run=busy, event_key=review.event_key(rows[0]), user=self.user,
                                   verdict="accepted")
        EventReview.objects.create(run=busy, event_key=review.event_key(rows[1]), user=maria,
                                   verdict="rejected")
        ClipReview.objects.create(run=busy, user=self.user)
        ClipReview.objects.create(run=empty, user=maria)

        busy_row, empty_row, untouched = self._download()

        self.assertEqual(
            [busy_row[k] for k in ("clip_reviewed", "reviewed_by", "events", "events_reviewed",
                                   "events_accepted", "events_rejected", "events_added")],
            ["true", "fl;maria", "2", "2", "1", "1", "0"])
        self.assertEqual((empty_row["clip_reviewed"], empty_row["reviewed_by"], empty_row["events"]),
                         ("true", "maria", "0"))
        self.assertEqual((untouched["clip_reviewed"], untouched["reviewed_by"],
                          untouched["events_reviewed"]), ("false", "", "0"))
