"""Every pipeline duration is set in seconds and becomes frames per clip."""
from importlib import import_module

from django.apps import apps as django_apps
from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.pipelines import executors, registry
from apps.pipelines.models import Pipeline


def step(**config):
    return {"block_type": "analyze.interactions", "config": config}


class GapToleranceTests(SimpleTestCase):
    def test_seconds_become_this_clips_frames(self):
        self.assertEqual(executors._gap_frames(step(gap_seconds=0.6), 25), 15)
        self.assertEqual(executors._gap_frames(step(gap_seconds=0.6), 30), 18)

    def test_default_is_0_6_seconds(self):
        self.assertEqual(executors._gap_frames(step(), 25), 15)
        self.assertEqual(executors._gap_frames(step(), 50), 30)

    def test_a_run_frozen_with_frames_keeps_them(self):
        self.assertEqual(executors._gap_frames(step(gap_frames=40), 30), 40)

    def test_no_field_is_in_frames_any_more(self):
        labels = [f["label"] for b in registry.BLOCK_REGISTRY.values()
                  for f in b.get("config_fields", [])] + [f["label"] for f in registry.TRACKER_FIELDS]
        self.assertFalse([label for label in labels if "(frames)" in label], labels)

    def test_external_tracker_durations_are_always_sent_in_seconds(self):
        out = executors.tracker_settings({"tracker": "ocsort"})
        self.assertEqual(out["max_age_seconds"], 1.2)
        self.assertEqual(out["min_hits_seconds"], 0.12)
        self.assertEqual(executors.tracker_settings({"tracker": "bytetrack"})["track_buffer_seconds"], 1.2)


class MinimumLengthTests(SimpleTestCase):
    def test_min_seconds_reads_the_node(self):
        self.assertEqual(executors._min_seconds(step(min_seconds="0.5")), 0.5)
        self.assertEqual(executors._min_seconds(step()), 0.0)
        self.assertEqual(executors._min_seconds(step(min_seconds="x")), 0.0)

    def test_the_interactions_node_offers_it(self):
        names = [f["name"] for f in registry.BLOCK_REGISTRY["analyze.interactions"]["config_fields"]]
        self.assertIn("min_seconds", names)


class MigrationTests(TestCase):
    def test_saved_frames_become_seconds_at_25_fps(self):
        user = get_user_model().objects.create_user("m", password="x")
        p = Pipeline.objects.create(user=user, title="P", steps=[
            {"id": "a", "block_type": "analyze.interactions", "config": {"gap_frames": "15"}},
            {"id": "e", "block_type": "analyze.events", "config": {"gap_frames": 50}},
            {"id": "t", "block_type": "track.mot", "config": {"tracker": "ocsort", "ocsort_max_age": "30",
                                                               "ocsort_min_hits": 3}},
            {"id": "d", "block_type": "detect.objects", "config": {"sample_interval": "30"}},
        ])
        import_module("apps.pipelines.migrations.0007_durations_in_seconds").forwards(django_apps, None)
        p.refresh_from_db()
        cfg = {s["id"]: s["config"] for s in p.steps}
        self.assertEqual(cfg["a"], {"gap_seconds": "0.6"})
        self.assertEqual(cfg["e"], {"gap_seconds": "2"})
        self.assertEqual(cfg["t"], {"tracker": "ocsort", "ocsort_max_age_seconds": "1.2",
                                    "ocsort_min_hits_seconds": "0.12"})
        self.assertEqual(cfg["d"], {"sample_seconds": "1.2"})


class OneLostTrackSettingTests(TestCase):
    def test_keep_and_revive_add_up_into_keep(self):
        user = get_user_model().objects.create_user("k", password="x")
        p = Pipeline.objects.create(user=user, title="P", steps=[
            {"id": "t", "block_type": "track.mot", "config": {
                "tracker": "beetrack", "beetrack_max_age_seconds": "2.0",
                "beetrack_max_resurrection_seconds": "1.0"}},
            {"id": "u", "block_type": "track.mot", "config": {
                "tracker": "beetrack", "beetrack_max_age_seconds": "0.5",
                "beetrack_max_resurrection_seconds": "0.3"}},
        ])
        import_module("apps.pipelines.migrations.0008_one_lost_track_setting").forwards(django_apps, None)
        p.refresh_from_db()
        cfg = {s["id"]: s["config"] for s in p.steps}
        self.assertEqual(cfg["t"], {"tracker": "beetrack", "beetrack_max_age_seconds": "3"})
        self.assertEqual(cfg["u"]["beetrack_max_age_seconds"], "0.8")
        self.assertNotIn("beetrack_max_resurrection_seconds", cfg["u"])

    def test_the_editor_has_no_revive_setting_and_explains_the_rest(self):
        names = {f["name"]: f for f in registry.TRACKER_FIELDS}
        self.assertNotIn("beetrack_max_resurrection_seconds", names)
        self.assertTrue(names["beetrack_max_age_seconds"].get("help"))
        self.assertTrue(names["beetrack_min_hits_seconds"].get("help"))


class ConfirmDefaultMigrationTests(TestCase):
    def test_untouched_zero_moves_to_0_2_and_a_choice_is_kept(self):
        user = get_user_model().objects.create_user("c", password="x")
        p = Pipeline.objects.create(user=user, title="P", steps=[
            {"id": "a", "block_type": "track.mot",
             "config": {"tracker": "beetrack", "beetrack_min_hits_seconds": "0"}},
            {"id": "b", "block_type": "track.mot",
             "config": {"tracker": "beetrack", "beetrack_min_hits_seconds": "0.5"}},
        ])
        import_module("apps.pipelines.migrations.0009_confirm_tracks_after_0_2s").forwards(django_apps, None)
        p.refresh_from_db()
        cfg = {s["id"]: s["config"] for s in p.steps}
        self.assertEqual(cfg["a"]["beetrack_min_hits_seconds"], "0.2")
        self.assertEqual(cfg["b"]["beetrack_min_hits_seconds"], "0.5")
