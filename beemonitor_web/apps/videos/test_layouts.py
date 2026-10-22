"""ROI and reference objects for sites and single clips (memory/52).

A clip without a unit (a hand upload of ground-nesting footage) can now be
analysed against drawn nest holes and flags: drawn on the clip or on its site,
looked up clip → site → unit, named, and never mixed with the bee-hotel model.
"""

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.devices.layouts import layout_for_video
from apps.devices.models import Device
from apps.pipelines import ops
from apps.videos.models import Layout, Site, Video

User = get_user_model()
HOLE = {"id": 1, "box": [0.4, 0.5, 0.5, 0.6], "name": "Hole 1"}
FLAG = {"id": 2, "box": [0.1, 0.4, 0.15, 0.48], "name": "Flag C"}


class Base(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("bee", password="pw-12345678")
        self.site = Site.objects.create(user=self.user, name="North bank")
        self.clip = Video.objects.create(user=self.user, site=self.site, site_name="North bank",
                                         title="ground_nest.mp4", storage_key="1/a/c.mp4",
                                         file_size_bytes=1, duration_seconds=2072, width=1920, height=1080)
        self.client.force_login(self.user)


class LookupTests(Base):
    def test_clip_then_site_then_unit(self):
        self.assertIsNone(layout_for_video(self.clip)["source"])
        Layout.save_new(site=self.site, nests=[HOLE], roi=[0, 0, 0.8, 1])
        got = layout_for_video(self.clip)
        self.assertEqual((got["source"], got["source_label"]), ("site", "site North bank (v1)"))
        self.assertEqual(got["nest_layout"], [HOLE])
        Layout.save_new(video=self.clip, nests=[FLAG])
        self.assertEqual(layout_for_video(self.clip)["source"], "video")
        other = Video.objects.create(user=self.user, site=self.site, title="b", storage_key="k", file_size_bytes=1)
        self.assertEqual(layout_for_video(other)["source"], "site")

    def test_a_unit_layout_still_works(self):
        unit = Device.objects.create(owner=self.user, name="Amber", key_hash="h", prefix="p",
                                     nest_layout=[{"id": 3, "box": [0, 0, 0.2, 0.2]}])
        clip = Video.objects.create(user=self.user, device=unit, title="u", storage_key="k2", file_size_bytes=1)
        self.assertEqual(layout_for_video(clip)["source"], "device")

    def test_saving_the_same_layout_twice_keeps_one_version(self):
        Layout.save_new(site=self.site, nests=[HOLE])
        Layout.save_new(site=self.site, nests=[HOLE])
        Layout.save_new(site=self.site, nests=[HOLE, FLAG])
        self.assertEqual(list(Layout.objects.filter(site=self.site).values_list("number", flat=True)), [2, 1])

    def test_names_reach_the_results(self):
        refs = ops.roi_references({"nest_layout": [HOLE, FLAG]})
        self.assertEqual([r["label"] for r in refs], ["Hole 1", "Flag C"])


class EditorTests(Base):
    def save(self, url, **body):
        import json
        return self.client.post(url, json.dumps({"roi": {"box": [0, 0, 0.8, 1]},
                                                 "nests": [HOLE, FLAG], **body}),
                                content_type="application/json")

    def test_the_clip_editor_draws_on_a_frame_of_the_clip(self):
        r = self.client.get(reverse("videos:layout", args=[self.clip.pk]))
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, reverse("videos:frame", args=[self.clip.pk]))
        self.assertContains(r, "Every clip at <strong>North bank</strong>")

    def test_saving_from_a_clip_goes_to_its_site_by_default(self):
        r = self.save(reverse("videos:layout", args=[self.clip.pk]))
        self.assertIn("every clip at North bank", r.json()["message"])
        site_layout = Layout.current(site=self.site)
        self.assertEqual([n["name"] for n in site_layout.nest_layout], ["Hole 1", "Flag C"])
        self.assertIsNone(Layout.current(video=self.clip))

    def test_or_to_the_clip_alone(self):
        self.save(reverse("videos:layout", args=[self.clip.pk]), scope="clip")
        self.assertIsNotNone(Layout.current(video=self.clip))
        self.assertIsNone(Layout.current(site=self.site))

    def test_a_site_that_was_only_a_name_becomes_a_site(self):
        clip = Video.objects.create(user=self.user, site_name="Orchard row 3", title="o",
                                    storage_key="k3", file_size_bytes=1)
        twin = Video.objects.create(user=self.user, site_name="Orchard row 3", title="o2",
                                    storage_key="k4", file_size_bytes=1)
        self.save(reverse("videos:layout", args=[clip.pk]))
        site = Site.objects.get(user=self.user, name="Orchard row 3")
        self.assertEqual(layout_for_video(Video.objects.get(pk=twin.pk))["source"], "site")
        self.assertEqual(Layout.current(site=site).number, 1)

    def test_the_site_editor(self):
        r = self.client.get(reverse("videos:site_layout", args=[self.site.pk]))
        self.assertContains(r, reverse("videos:frame", args=[self.clip.pk]))
        self.save(reverse("videos:site_layout", args=[self.site.pk]))
        self.assertEqual(len(Layout.current(site=self.site).nest_layout), 2)

    def test_other_users_cannot_edit(self):
        self.client.force_login(User.objects.create_user("x"))
        self.assertEqual(self.client.get(reverse("videos:layout", args=[self.clip.pk])).status_code, 404)
        self.assertEqual(self.save(reverse("videos:site_layout", args=[self.site.pk])).status_code, 404)
        self.assertFalse(Layout.objects.exists())

    def test_a_platform_admin_can_draw_on_anyones_clip_and_it_is_audited(self):
        from apps.accounts.models import AdminAction
        admin = User.objects.create_superuser("boss", "b@x.org", "pw-12345678")
        self.client.force_login(admin)
        self.assertContains(self.client.get(reverse("videos:detail", args=[self.clip.pk])),
                            reverse("videos:layout", args=[self.clip.pk]))
        self.assertEqual(self.client.get(reverse("videos:layout", args=[self.clip.pk])).status_code, 200)
        self.save(reverse("videos:layout", args=[self.clip.pk]), scope="clip")
        self.assertEqual(Layout.current(video=self.clip).saved_by, admin)
        log = AdminAction.objects.get()
        self.assertEqual((log.actor, log.target, log.action), (admin, self.user, "edit_layout"))

    def test_the_unit_editor_keeps_names_too(self):
        import json
        unit = Device.objects.create(owner=self.user, name="Amber", key_hash="h2", prefix="p")
        self.client.post(reverse("devices:roi_editor", args=[unit.pk]), json.dumps({"nests": [FLAG]}),
                         content_type="application/json")
        unit.refresh_from_db()
        self.assertEqual(unit.nest_layout[0]["name"], "Flag C")

    def test_the_clip_page_says_where_its_layout_comes_from(self):
        r = self.client.get(reverse("videos:detail", args=[self.clip.pk]))
        self.assertContains(r, "visits can't be counted")
        Layout.save_new(site=self.site, nests=[HOLE])
        r = self.client.get(reverse("videos:detail", args=[self.clip.pk]))
        self.assertContains(r, "from site North bank (v1)")


class NewSiteTests(Base):
    """A new site has no clips: pick one of yours to draw on; it joins the site."""

    def test_pick_a_clip_draw_and_it_joins_the_site(self):
        import json
        loose = Video.objects.create(user=self.user, title="loose.mp4", storage_key="k9", file_size_bytes=1,
                                     duration_seconds=600)
        new = Site.objects.create(user=self.user, name="site1")
        r = self.client.get(reverse("videos:site_layout", args=[new.pk]))
        self.assertContains(r, "Pick a clip to draw on")
        self.assertContains(r, "loose.mp4")
        r = self.client.get(reverse("videos:site_layout", args=[new.pk]), {"clip": loose.pk})
        self.assertContains(r, reverse("videos:frame", args=[loose.pk]))
        self.client.post(reverse("videos:site_layout", args=[new.pk]),
                         json.dumps({"nests": [HOLE], "clip": loose.pk}), content_type="application/json")
        loose.refresh_from_db()
        self.assertEqual(loose.site, new)
        self.assertEqual(layout_for_video(loose)["source"], "site")

    def test_another_users_clip_cannot_be_picked(self):
        other = User.objects.create_user("x")
        theirs = Video.objects.create(user=other, title="theirs.mp4", storage_key="k8", file_size_bytes=1)
        new = Site.objects.create(user=self.user, name="site1")
        r = self.client.get(reverse("videos:site_layout", args=[new.pk]), {"clip": theirs.pk})
        self.assertNotContains(r, reverse("videos:frame", args=[theirs.pk]))


class SitesPageTests(Base):
    def test_list_and_add(self):
        r = self.client.get(reverse("videos:sites_page"))
        self.assertContains(r, "North bank")
        self.assertContains(r, "None yet")
        r = self.client.post(reverse("videos:sites_page"), {"name": "Campus garden"})
        site = Site.objects.get(name="Campus garden")
        self.assertRedirects(r, reverse("videos:site_layout", args=[site.pk]), fetch_redirect_response=False)


class FrameTests(Base):
    def test_a_frame_at_a_time(self):
        done = mock.Mock(returncode=0, stdout=b"\xff\xd8jpeg")
        with mock.patch("config.storage.presigned_get", return_value="https://s3/x"), \
                mock.patch("subprocess.run", return_value=done) as run:
            r = self.client.get(reverse("videos:frame", args=[self.clip.pk]), {"t": "125"})
        self.assertEqual((r.status_code, r["Content-Type"]), (200, "image/jpeg"))
        self.assertIn("125.000", run.call_args[0][0])

    def test_no_frame_once_the_raw_footage_is_gone(self):
        from django.utils import timezone
        Video.objects.filter(pk=self.clip.pk).update(raw_deleted_at=timezone.now())
        self.assertEqual(self.client.get(reverse("videos:frame", args=[self.clip.pk])).status_code, 404)


class AnalysisTests(Base):
    def test_a_site_layout_reaches_the_job_and_turns_off_the_hotel_model(self):
        from apps.analysis.views import _video_job_config
        Layout.save_new(site=self.site, nests=[HOLE], roi=[0, 0, 0.8, 1])
        cfg = _video_job_config({}, self.clip, use_device_roi=True)
        self.assertEqual((cfg["nest_layout"], cfg["hotel_roi"], cfg["nest_fallback"]), ([HOLE], [0, 0, 0.8, 1], False))

    def test_a_unit_layout_keeps_the_hotel_model(self):
        from apps.analysis.views import _video_job_config
        unit = Device.objects.create(owner=self.user, name="Amber", key_hash="h3", prefix="p",
                                     roi_override=[0, 0, 1, 1])
        clip = Video.objects.create(user=self.user, device=unit, title="u", storage_key="k5", file_size_bytes=1)
        self.assertNotIn("nest_fallback", _video_job_config({}, clip, use_device_roi=True))

    def test_pipeline_gpu_config_uses_the_site_layout(self):
        from apps.pipelines.executors import build_detect_and_track_config
        from apps.pipelines.models import Pipeline, PipelineRun
        Layout.save_new(site=self.site, nests=[HOLE])
        steps = [{"id": "v", "block_type": "input.video", "config": {"video_id": str(self.clip.pk)}},
                 {"id": "d", "block_type": "detect.objects", "config": {}, "inputs": {"video": "v"}}]
        pipeline = Pipeline.objects.create(user=self.user, title="p", steps=steps)
        run = PipelineRun.objects.create(pipeline=pipeline, user=self.user, steps=steps)
        context = {"v": {"artifact": "video", "video_id": self.clip.pk}}
        built, err = build_detect_and_track_config(steps[1], run, context, 1)
        self.assertIsNone(err)
        self.assertEqual((built["config"]["nest_layout"], built["config"]["nest_fallback"]), ([HOLE], False))

    def test_sampled_detections_become_references_with_stable_ids(self):
        from apps.pipelines.executors import detected_reference
        from apps.pipelines.models import Pipeline, PipelineRun
        steps = [{"id": "v", "block_type": "input.video", "config": {"video_id": str(self.clip.pk)}},
                 {"id": "d", "block_type": "detect.objects", "config": {"label": "flag"}, "inputs": {"video": "v"}}]
        pipeline = Pipeline.objects.create(user=self.user, title="p", steps=steps)
        run = PipelineRun.objects.create(pipeline=pipeline, user=self.user, steps=steps)

        def box(x, y):
            return {"x": x, "y": y, "w": 96, "h": 108, "class": "flag", "confidence": 0.9}
        # The same two flags in three frames, jittering a few pixels.
        frames = [{"frame_number": n, "boxes": [box(1600 + n, 810), box(180, 410 + n)]} for n in (0, 2, 4)]
        context = {"v": {"artifact": "video", "video_id": self.clip.pk},
                   "d": {"result": {"summary_stats": {"sampled_frames": frames}}}}
        out = detected_reference(steps[1], run, context, 1)
        refs = out["nest_layout"]
        self.assertEqual([r["id"] for r in refs], [1, 2])
        self.assertAlmostEqual(refs[0]["box"][0], 180 / 1920, places=3)   # the higher flag is 1
        self.assertAlmostEqual(refs[1]["box"][0], (1600 + 0) / 1920, delta=0.003)
