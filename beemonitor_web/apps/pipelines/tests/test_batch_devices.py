"""Which unit a clip came from, on the page and in the download.

The batch header printed rows.0.video.device.name — the FIRST clip's device,
labelled as if it were the batch's. A batch is a set of clips and nothing stops
them coming from several units, so that was a guess: right whenever a batch
happened to be single-device, silently wrong when it was not. Which unit a
result came from is the first thing you need before trusting it.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.devices.models import Device
from apps.pipelines import aggregate
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()

BATCH = "cccc3333-0000-4000-8000-0000000000aa"


class BatchDeviceTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bd", password="x")
        self.client.force_login(self.user)
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")
        self.a = Device.objects.create(owner=self.user, name="BeeMonitor4",
                                       key_hash="kbd1", prefix="bmk_a",
                                       location="north hedgerow")
        self.b = Device.objects.create(owner=self.user, name="BeeMonitor7",
                                       key_hash="kbd2", prefix="bmk_b")

    def _clip(self, device, title):
        video = Video.objects.create(
            user=self.user, device=device, title=title,
            storage_key=f"bd/{title}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now())
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH,
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(video.pk)}}],
            context={"v": {"artifact": "video", "video_id": video.pk},
                     "m": {"artifact": "tracks",
                           "result": {"tracking_csv_path": "t.csv"}}})

    # ── the list itself ─────────────────────────────────────────────────

    def test_one_device_is_listed_once(self):
        rows = aggregate.batch_rows([self._clip(self.a, "one"),
                                     self._clip(self.a, "two")])

        devices = aggregate.batch_devices(rows)

        self.assertEqual([d["name"] for d in devices], ["BeeMonitor4"])
        self.assertEqual(devices[0]["location"], "north hedgerow")

    def test_a_batch_spanning_units_names_all_of_them(self):
        rows = aggregate.batch_rows([self._clip(self.a, "one"),
                                     self._clip(self.b, "two")])

        self.assertEqual(sorted(d["name"] for d in aggregate.batch_devices(rows)),
                         ["BeeMonitor4", "BeeMonitor7"])

    def test_a_clip_with_no_device_is_skipped_not_blank(self):
        rows = aggregate.batch_rows([self._clip(None, "orphan"),
                                     self._clip(self.a, "one")])

        self.assertEqual([d["name"] for d in aggregate.batch_devices(rows)],
                         ["BeeMonitor4"])

    def test_no_devices_at_all_is_an_empty_list(self):
        rows = aggregate.batch_rows([self._clip(None, "orphan")])

        self.assertEqual(aggregate.batch_devices(rows), [])

    # ── the page ────────────────────────────────────────────────────────

    def test_the_page_names_every_device_in_the_batch(self):
        self._clip(self.a, "one")
        self._clip(self.b, "two")

        html = self.client.get(reverse("pipelines:batch_detail",
                                       kwargs={"batch_id": BATCH})).content.decode()

        self.assertIn("BeeMonitor4", html)
        self.assertIn("BeeMonitor7", html)

    def test_each_row_carries_its_own_device(self):
        """The header says which units; a row says which one is THIS clip."""
        self._clip(self.a, "one")
        self._clip(self.b, "two")

        html = self.client.get(reverse("pipelines:batch_detail",
                                       kwargs={"batch_id": BATCH})).content.decode()

        # Once in the header, once on the row, for each device.
        self.assertGreaterEqual(html.count("BeeMonitor4"), 2)
        self.assertGreaterEqual(html.count("BeeMonitor7"), 2)


class DownloadProvenanceTests(TestCase):
    """The CSV must name the device that actually recorded each clip."""

    def setUp(self):
        self.user = User.objects.create_user("dp", password="x")
        self.a = Device.objects.create(owner=self.user, name="BeeMonitor4",
                                       key_hash="kdp1", prefix="bmk_p",
                                       location="north hedgerow")
        self.b = Device.objects.create(owner=self.user, name="BeeMonitor7",
                                       key_hash="kdp2", prefix="bmk_q",
                                       location="south field")

    def _src(self, device, title, site=""):
        video = Video.objects.create(
            user=self.user, device=device, title=title,
            storage_key=f"dp/{title}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now())
        if site:
            video.site_name = site
            video.save(update_fields=["site_name"])
        return {"video": video, "title": title, "recorded_at": video.recorded_at}

    def test_each_row_names_its_own_clips_device(self):
        one = aggregate._provenance(self._src(self.a, "one"))
        two = aggregate._provenance(self._src(self.b, "two"))

        self.assertEqual(one["device_name"], "BeeMonitor4")
        self.assertEqual(one["device_id"], self.a.pk)
        self.assertEqual(two["device_name"], "BeeMonitor7")
        self.assertEqual(two["device_id"], self.b.pk)

    def test_the_id_is_carried_so_names_can_be_renamed_safely(self):
        """A name is a label a person edits; the id is what joins."""
        prov = aggregate._provenance(self._src(self.a, "one"))
        self.a.name = "Renamed"
        self.a.save()

        self.assertEqual(prov["device_id"], self.a.pk)

    def test_a_clip_with_no_device_reports_blank_not_a_crash(self):
        prov = aggregate._provenance(self._src(None, "orphan"))

        self.assertEqual(prov["device_name"], "")
        self.assertEqual(prov["device_id"], "")

    def test_the_clips_own_site_wins_over_the_devices_current_one(self):
        """A device can be moved; the clip records where it actually was."""
        prov = aggregate._provenance(self._src(self.a, "one", site="old orchard"))

        self.assertEqual(prov["site_name"], "old orchard")
        self.assertEqual(prov["device_name"], "BeeMonitor4")


class ExportedRowsCarryTheirDeviceTests(TestCase):
    """End to end: two devices in one batch, one CSV, each row on its own unit.

    _provenance being right is necessary but not sufficient — what matters is
    that the join survives the export, so a row about BeeMonitor7's clip cannot
    come out labelled BeeMonitor4.
    """

    def setUp(self):
        self.user = User.objects.create_user("ex", password="x")
        self.pipeline = Pipeline.objects.create(user=self.user, title="P")
        self.a = Device.objects.create(owner=self.user, name="BeeMonitor4",
                                       key_hash="kex1", prefix="bmk_x")
        self.b = Device.objects.create(owner=self.user, name="BeeMonitor7",
                                       key_hash="kex2", prefix="bmk_y")

    def _run_with_interactions(self, device, title, partner):
        video = Video.objects.create(
            user=self.user, device=device, title=title,
            storage_key=f"ex/{title}.mp4", file_size_bytes=1,
            status=Video.Status.READY, recorded_at=timezone.now())
        rows = [{"start_frame": 0, "end_frame": 5, "start_sec": 0.0, "end_sec": 0.2,
                 "duration_sec": 0.2, "a": partner, "a_kind": "organism",
                 "b": "nest_1", "b_kind": "reference", "relation": "inside",
                 "source": "derived"}]
        return PipelineRun.objects.create(
            pipeline=self.pipeline, user=self.user, batch_id=BATCH,
            status="completed",
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(video.pk)}}],
            context={"v": {"artifact": "video", "video_id": video.pk},
                     "a": {"artifact": "table", "table_kind": "interactions",
                           "interaction_count": len(rows), "rows": rows}})

    def test_every_exported_row_names_the_unit_that_recorded_it(self):
        runs = [self._run_with_interactions(self.a, "from-a", "track-a"),
                self._run_with_interactions(self.b, "from-b", "track-b")]

        fields, rows = aggregate.primitive_csv(runs, "interactions")

        self.assertIn("device_name", fields)
        by_track = {r["a"]: r for r in rows}
        self.assertEqual(by_track["track-a"]["device_name"], "BeeMonitor4")
        self.assertEqual(by_track["track-b"]["device_name"], "BeeMonitor7")
        self.assertEqual(by_track["track-a"]["video_title"], "from-a")
        self.assertEqual(by_track["track-b"]["video_title"], "from-b")

    def test_device_columns_come_first_so_a_spreadsheet_sees_them(self):
        runs = [self._run_with_interactions(self.a, "from-a", "track-a")]

        fields, _rows = aggregate.primitive_csv(runs, "interactions")

        self.assertLess(fields.index("device_name"), fields.index("a"))
