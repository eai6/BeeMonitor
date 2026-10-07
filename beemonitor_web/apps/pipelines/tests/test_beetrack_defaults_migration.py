"""Saved BeeTrack MOT nodes on 0.5 / 0.3 move to 2.0 / 1.0; chosen values stay."""

import importlib

from django.apps import apps as django_apps
from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.pipelines.models import Pipeline

migration = importlib.import_module("apps.pipelines.migrations.0006_beetrack_longer_track_memory")


def _mot(**config):
    return [{"id": "m", "block_type": "track.mot", "config": config}]


class MigrationTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user("bt", password="x")

    def _migrated(self, steps):
        p = Pipeline.objects.create(user=self.user, title="p", steps=steps)
        migration.forwards(django_apps, None)
        p.refresh_from_db()
        return p.steps[0]["config"]

    def test_old_defaults_move(self):
        cfg = self._migrated(_mot(tracker="beetrack", beetrack_max_age_seconds="0.5",
                                  beetrack_max_resurrection_seconds="0.3"))
        self.assertEqual((cfg["beetrack_max_age_seconds"], cfg["beetrack_max_resurrection_seconds"]),
                         ("2.0", "1.0"))

    def test_a_chosen_value_stays(self):
        cfg = self._migrated(_mot(tracker="beetrack", beetrack_max_age_seconds="1.2"))
        self.assertEqual(cfg["beetrack_max_age_seconds"], "1.2")

    def test_other_trackers_untouched(self):
        cfg = self._migrated(_mot(tracker="bytetrack", beetrack_max_age_seconds="0.5"))
        self.assertEqual(cfg["beetrack_max_age_seconds"], "0.5")
