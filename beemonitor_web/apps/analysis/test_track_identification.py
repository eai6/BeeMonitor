"""BioCLIP gets the region's species; the species step says why it has none."""

from unittest import mock

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.analysis.views import _candidate_taxa
from apps.devices.models import Device
from apps.pipelines.executors import _species_note
from apps.videos.models import Video

User = get_user_model()


class CandidateTaxaTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("ct", password="x")

    def _video(self, **device):
        dev = Device.objects.create(owner=self.user, name="u", **device)
        return Video.objects.create(user=self.user, device=dev, title="c",
                                    storage_key="ct/c.mp4", file_size_bytes=1)

    def test_located_device_asks_for_its_region(self):
        video = self._video(lat=40.8, lon=-77.9)
        with mock.patch("apps.monitor.priors.region_taxa",
                        return_value=["Osmia lignaria"]) as rt:
            self.assertEqual(_candidate_taxa(video), ["Osmia lignaria"])
        self.assertEqual(rt.call_args[0][:2], (40.8, -77.9))

    def test_no_location_means_tree_of_life(self):
        self.assertEqual(_candidate_taxa(self._video()), [])

    def test_lookup_failure_never_blocks_the_job(self):
        video = self._video(lat=1.0, lon=2.0)
        with mock.patch("apps.monitor.priors.region_taxa", side_effect=RuntimeError):
            self.assertEqual(_candidate_taxa(video), [])


class SpeciesNoteTests(TestCase):
    def _note(self, species):
        return _species_note({"summary_stats": {"identification": {"species": species}}})

    def test_model_failure_is_named(self):
        note = self._note({"model": "bioclip", "loaded": False, "error": "no weights"})
        self.assertIn("BioCLIP could not be loaded: no weights", note)

    def test_no_crops(self):
        self.assertIn("no track crops", self._note({"model": "beemachine", "loaded": True,
                                                    "tracks": 0}))

    def test_old_runs_are_told_to_re_run(self):
        self.assertIn("re-run", _species_note({"summary_stats": {}}))
