"""The rendered video is the player's picture (memory/61): the overlay's live
part carries events, interactions, references and the drawing style, and a
render sends the GPU a spec built from exactly that — never the worker's own
nest boxes or the raw tracking CSV."""

import gzip
import json
from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from apps.analysis import overlay
from apps.analysis.models import Job, JobResult
from apps.analysis.overlay_spec import DEFAULT_LAYERS, STYLE, layers_from, track_rgb
from apps.analysis.test_job_primitives import AGREE_TRACKING, _csv
from apps.devices.models import Device
from apps.videos.models import Video

User = get_user_model()

BOXES = {"v": overlay.VERSION, "fps": 25.0, "tracks": {"3": {"species": "Osmia"}},
         "rows": [0, 3, 10, 10, 20, 20, 0], "row_width": 7}


class RenderVideoTests(TestCase):
    def setUp(self):
        from apps.pipelines.models import Pipeline, PipelineRun
        from apps.pipelines.recount import recount_run

        self.user = User.objects.create_user("render", password="x")
        self.client.force_login(self.user)
        self.device = Device.objects.create(
            owner=self.user, name="Hotel", key_hash="hrv", prefix="bmk_rv",
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}], rotate_180=True)
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip", storage_key="rv/c.mp4",
            file_size_bytes=1, status=Video.Status.READY, recorded_at=timezone.now(),
            fps=25.0, width=1920, height=1080, duration_seconds=4.0)
        self.job = Job.objects.create(user=self.user, video=self.video,
                                      status="completed", modal_job_id="rv-1")
        path = _csv(AGREE_TRACKING, "render.csv")
        JobResult.objects.create(job=self.job, tracking_csv_path=path,
                                 summary_stats={"video_fps": 25.0,
                                                "nest_bboxes": {"9": [0, 0, 5, 5]}})
        pipeline = Pipeline.objects.create(user=self.user, title="P")
        run = PipelineRun.objects.create(
            pipeline=pipeline, user=self.user, status="completed",
            started_at=self.job.created_at,
            steps=[{"id": "v", "block_type": "input.video",
                    "config": {"video_id": str(self.video.pk)}},
                   {"id": "r", "block_type": "reference.layout",
                    "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
                   {"id": "e", "block_type": "analyze.events", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}},
                   {"id": "i", "block_type": "analyze.interactions", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}}],
            context={"v": {"artifact": "video", "video_id": self.video.pk},
                     "m": {"artifact": "tracks", "job_id": self.job.pk,
                           "result": {"tracking_csv_path": path,
                                      "summary_stats": {"video_fps": 25.0}}}})
        recount_run(run)

    def live(self):
        return json.loads(self.client.get(
            reverse("analysis:track_overlay", args=[self.job.pk]), {"part": "events"}).content)

    def test_the_live_part_carries_events_interactions_references_and_the_style(self):
        live = self.live()
        self.assertEqual(sorted(e[2] for e in live["events"]), ["enter", "exit"])
        self.assertTrue(live["interactions"])
        self.assertTrue(all(r[4] == "in_reference" for r in live["interactions"]))
        self.assertEqual([r["label"] for r in live["regions"]], ["reference_1"])
        self.assertEqual(live["style"], STYLE)

    @mock.patch("apps.accounts.quota.check")
    @mock.patch("apps.analysis.views._invoke_endpoint_async", return_value=("s3://out", "s3://fail"))
    @mock.patch("apps.analysis.views._put_inference_payload", return_value="s3://in")
    @mock.patch("apps.analysis.views.stored_overlay", return_value=overlay.encode(BOXES))
    @mock.patch("apps.analysis.views.get_s3_client")
    def test_a_render_sends_the_players_picture_with_the_chosen_layers(
            self, s3, _stored, put, _invoke, _quota):
        uploaded = {}
        s3.return_value.upload_stream.side_effect = (
            lambda bucket, key, stream, **kw: uploaded.update({key: stream.read()}))
        self.client.post(reverse("analysis:annotate_video", args=[self.job.pk]),
                         {"layers": "1", "boxes": "on", "events": "on", "clock": "on"})

        payload = put.call_args[0][1]
        self.assertEqual(payload["task"], "annotate_video")
        self.assertNotIn("tracking_csv_path", payload)
        self.assertNotIn("nest_bboxes", payload)
        spec = json.loads(gzip.decompress(uploaded[payload["render_spec_path"]]))
        self.assertEqual(spec["rows"], BOXES["rows"])
        self.assertEqual(spec["style"], STYLE)
        self.assertTrue(spec["rotate_180"])
        self.assertEqual(sorted(e[2] for e in spec["events"]), ["enter", "exit"])
        self.assertEqual([r["label"] for r in spec["regions"]], ["reference_1"])
        # Only the layers ticked in the dialog.
        self.assertEqual({k for k, v in spec["layers"].items() if v}, {"boxes", "events", "clock"})
        self.assertTrue(spec["clip_start"])

    def test_the_render_dialog_offers_every_layer_and_its_cost_in_gpu_hours(self):
        with mock.patch("apps.analysis.views._generate_presigned_url", return_value="https://x"):
            html = self.client.get(reverse("analysis:results", args=[self.job.pk])).content.decode()
        for name in ("boxes", "species", "trails", "lost", "regions", "events", "interactions", "clock"):
            self.assertIn(f'data-ov-render-layer="{name}"', html)
        self.assertIn("GPU-hours", html)
        self.assertIn('data-ov-opt="events"', html)
        self.assertIn('data-ov-opt="interactions"', html)


class SpecTests(TestCase):
    def test_layers_default_and_follow_the_form(self):
        self.assertEqual(layers_from({}), DEFAULT_LAYERS)
        # The render dialog names only what's ticked: the rest is off.
        got = layers_from({"layers": "1", "boxes": "on", "regions": "on"})
        self.assertEqual({k for k, v in got.items() if v}, {"boxes", "regions"})

    def test_track_colour_is_the_hsl_rule(self):
        import colorsys

        r, g, b = colorsys.hls_to_rgb(((12 * 137.508) % 360) / 360, 0.58, 0.85)
        self.assertEqual(track_rgb(12), (round(r * 255), round(g * 255), round(b * 255)))
