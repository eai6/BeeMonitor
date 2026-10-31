"""Tracks filtered by insect group before counting (memory/65).

Ants walking into the tubes were counted as bee events. Identify Species now
passes tracks on: each crop's species name is looked up to its family, the
crops vote on a group (bees, wasps, ants…), and the analyzers wired after it
count only the groups it keeps. Tracks with no clear group are always counted.
The grouping lives in the pipeline and can be changed while reviewing; all of
it is web-side, so changing it is a recount, never a GPU run.
"""

import json
from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase
from django.urls import reverse

from apps.analysis.models import Job
from apps.pipelines import executors, recount, taxon_groups
from apps.pipelines.models import PipelineRun

from .test_primitive_blocks import PrimitiveBlockTestCase, _write_csv

LINEAGE = {
    "Camponotus pennsylvanicus": {"class": "Insecta", "order": "Hymenoptera", "family": "Formicidae", "genus": "Camponotus"},
    "Lasius niger": {"class": "Insecta", "order": "Hymenoptera", "family": "Formicidae", "genus": "Lasius"},
    "Periplaneta americana": {"class": "Insecta", "order": "Blattodea", "family": "Blattidae", "genus": "Periplaneta"},
    "Bombus impatiens": {"class": "Insecta", "order": "Hymenoptera", "family": "Apidae", "genus": "Bombus"},
    "Vespula maculifrons": {"class": "Insecta", "order": "Hymenoptera", "family": "Vespidae", "genus": "Vespula"},
    "Ectemnius continuus": {"class": "Insecta", "order": "Hymenoptera", "family": "Crabronidae", "genus": "Ectemnius"},
    "Myotis lucifugus": {"class": "Mammalia", "order": "Chiroptera", "family": "Vespertilionidae", "genus": "Myotis"},
}


def fake_gbif(name):
    return dict(LINEAGE[name]) if name in LINEAGE else {}


def cfg(**config):
    return taxon_groups.settings(config)


class GroupingTests(SimpleTestCase):
    def setUp(self):
        cache.clear()
        patcher = mock.patch.object(taxon_groups, "_gbif", side_effect=fake_gbif)
        self.gbif = patcher.start()
        self.addCleanup(patcher.stop)

    def test_families_fall_in_groups(self):
        lin = taxon_groups.lineages(list(LINEAGE) + ["Wasp", "Syrphidae"])
        group = {n: taxon_groups.default_group(l) for n, l in lin.items()}
        self.assertEqual(group["Camponotus pennsylvanicus"], "Ants")
        self.assertEqual(group["Bombus impatiens"], "Bees")
        self.assertEqual(group["Vespula maculifrons"], "Wasps")
        self.assertEqual(group["Periplaneta americana"], "Other crawlers")
        self.assertEqual(group["Myotis lucifugus"], "Unclear")
        # BeeMachine's coarse classes need no lookup.
        self.assertEqual(group["Wasp"], "Wasps")
        self.assertEqual(group["Syrphidae"], "Flies")

    def test_an_ant_read_as_several_species_and_a_cockroach_is_an_ant(self):
        crops = (["Periplaneta americana"] * 3 + ["Camponotus pennsylvanicus"] * 2
                 + ["Lasius niger"] * 2)
        g = taxon_groups.track_groups({7: crops}, cfg())[7]
        # No single ant species beats the cockroach, but the ants as a group do.
        self.assertEqual(g["top"], "Periplaneta americana")
        self.assertEqual(g["group"], "Ants")
        self.assertAlmostEqual(g["share"], 0.571, places=3)
        self.assertFalse(g["counted"])

    def test_a_split_vote_or_too_few_crops_is_unclear_and_counted(self):
        groups = taxon_groups.track_groups({
            1: ["Camponotus pennsylvanicus"] * 2 + ["Bombus impatiens"] * 2 + ["Periplaneta americana"],
            2: ["Camponotus pennsylvanicus"] * 2,
        }, cfg())
        self.assertEqual(groups[1]["group"], "Unclear")
        self.assertTrue(groups[1]["counted"])
        self.assertEqual(groups[2]["group"], "Unclear")
        self.assertEqual(groups[2]["why"], "only 2 crops")
        self.assertTrue(groups[2]["counted"])

    def test_a_name_gbif_cant_place_is_unclear(self):
        g = taxon_groups.track_groups({1: ["Not a real thing"] * 5}, cfg())[1]
        self.assertEqual(g["group"], "Unclear")
        self.assertTrue(g["counted"])

    def test_a_failed_lookup_is_unclear_and_retried_later(self):
        self.gbif.side_effect = lambda name: None
        g = taxon_groups.track_groups({1: ["Bombus impatiens"] * 5}, cfg())[1]
        self.assertEqual(g["group"], "Unclear")
        self.gbif.side_effect = fake_gbif
        g = taxon_groups.track_groups({1: ["Bombus impatiens"] * 5}, cfg())[1]
        self.assertEqual(g["group"], "Bees")

    def test_lookups_are_cached(self):
        taxon_groups.lineages(["Bombus impatiens"])
        taxon_groups.lineages(["Bombus impatiens"])
        self.assertEqual(self.gbif.call_count, 1)

    def test_rules_win_species_over_genus_over_family_and_a_track_over_all(self):
        rules = {"groups": ["Small dark bees"], "rules": [
            {"kind": "family", "value": "Crabronidae", "group": "Bees"},
            {"kind": "genus", "value": "Ectemnius", "group": "Wasps"},
            {"kind": "name", "value": "Ectemnius continuus", "group": "Small dark bees"},
            {"kind": "track", "job": 9, "track": 2, "group": "Ants"},
        ]}
        groups = taxon_groups.track_groups(
            {1: ["Ectemnius continuus"] * 4, 2: ["Bombus impatiens"] * 4},
            cfg(group_rules=json.dumps(rules)), job_id=9)
        self.assertEqual(groups[1]["group"], "Small dark bees")
        self.assertTrue(groups[1]["counted"])        # not in leave_out
        self.assertEqual(groups[2]["group"], "Ants")
        self.assertEqual(groups[2]["why"], "set by hand")
        self.assertFalse(groups[2]["counted"])

    def test_a_rule_regroups_every_crop_with_that_name(self):
        rules = {"rules": [{"kind": "name", "value": "Periplaneta americana", "group": "Ants"}]}
        g = taxon_groups.track_groups(
            {1: ["Periplaneta americana"] * 3 + ["Bombus impatiens"] * 2},
            cfg(group_rules=json.dumps(rules)))[1]
        self.assertEqual(g["group"], "Ants")

    def test_off_names_groups_but_counts_every_track(self):
        g = taxon_groups.track_groups({1: ["Camponotus pennsylvanicus"] * 5},
                                      cfg(group_filter="off"))[1]
        self.assertEqual(g["group"], "Ants")
        self.assertTrue(g["counted"])

    def test_you_choose_which_groups_to_leave_out(self):
        moths_only = cfg(leave_out=json.dumps(["Moths & butterflies"]))
        g = taxon_groups.track_groups({1: ["Camponotus pennsylvanicus"] * 5}, moths_only)[1]
        self.assertTrue(g["counted"])

    def test_settings_defaults_and_a_percentage(self):
        s = cfg()
        self.assertTrue(s["on"])
        self.assertEqual(s["leave_out"], taxon_groups.DEFAULT_LEAVE_OUT)
        self.assertNotIn("Bees", s["leave_out"])
        self.assertEqual(s["share"], 0.5)
        self.assertEqual(cfg(agree_share="70")["share"], 0.7)
        # Unclear can never be left out.
        self.assertEqual(cfg(leave_out='["Unclear", "Ants"]')["leave_out"], ["Ants"])

    def test_with_rule_replaces_removes_and_adds_new_groups(self):
        c = taxon_groups.with_rule({}, {"kind": "name", "value": "Lasius niger", "group": "Ants"})
        c = taxon_groups.with_rule(c, {"kind": "name", "value": "lasius niger", "group": "Tiny things"})
        r = json.loads(c["group_rules"])
        self.assertEqual(r["rules"], [{"kind": "name", "value": "lasius niger", "group": "Tiny things"}])
        self.assertEqual(r["groups"], ["Tiny things"])
        c = taxon_groups.with_rule(c, {"kind": "name", "value": "Lasius niger", "group": ""})
        self.assertEqual(json.loads(c["group_rules"])["rules"], [])


CROPS = ([{"track_id": 1, "frame": f, "crop": f"frame_{f}.jpg", "species": "Camponotus pennsylvanicus"}
          for f in range(5)]
         + [{"track_id": 2, "frame": f, "crop": f"frame_{f}.jpg", "species": "Bombus impatiens"}
            for f in range(5)])


class FilterBase(PrimitiveBlockTestCase):
    """Identify Species in the line: MOT → Identify Species → Events."""

    def setUp(self):
        super().setUp()
        cache.clear()
        patcher = mock.patch.object(taxon_groups, "_gbif", side_effect=fake_gbif)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.votes_csv = _write_csv(CROPS, "track_votes.csv")

    def _steps_with_species(self, config=None, in_line=True):
        steps = self._steps("analyze.events")
        species = {"id": "s", "block_type": "identify.species",
                   "config": {"model": "bioclip", **(config or {})}, "inputs": {"tracks": "m"}}
        if in_line:
            steps[4]["inputs"] = {"tracks": "s", "rois": "r"}
        # Saved after the analyzer, as the canvas can.
        return steps + [species]

    def _run(self, steps):
        import uuid
        job = Job.objects.create(user=self.user, video=self.video, status=Job.Status.COMPLETED,
                                 modal_job_id=uuid.uuid4().hex)
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user, steps=steps,
                                         status=PipelineRun.Status.COMPLETED)
        run.context = {
            "v": {"artifact": "video", "video_id": self.video.pk},
            "m": {"artifact": "tracks", "job_id": job.pk, "label": "bee", "result": {
                "tracking_csv_path": self.tracking_csv,
                "summary_stats": {"video_fps": 25.0, "track_votes_csv": self.votes_csv}}},
            "s": {"artifact": "table", "table_kind": "species", "rows": []},
            "a": {"artifact": "events", "table_kind": "events", "rows": []},
        }
        run.save()
        recount.recount_run(run)
        run.refresh_from_db()
        return run, job


class FilterTests(FilterBase):
    def test_events_skip_ant_tracks(self):
        run, _job = self._run(self._steps_with_species())

        species = run.context["s"]
        self.assertEqual(species["artifact"], "tracks")
        self.assertEqual(species["dropped_tracks"], [1])
        self.assertEqual(species["track_groups"]["1"]["group"], "Ants")
        self.assertEqual(species["track_groups"]["2"]["group"], "Bees")
        events = run.context["a"]
        # Track 1 makes both events (test_recount); it's an ant, so none count.
        self.assertEqual(events["rows"], [])
        self.assertEqual(events["filtered_out"]["tracks"], 1)
        self.assertEqual(events["filtered_out"]["events"], 2)
        self.assertEqual(events["filtered_out"]["by_group"][0]["group"], "Ants")
        self.assertEqual(len(events["filtered_out"]["event_keys"]), 2)

    def test_with_the_filter_off_every_track_counts(self):
        run, _job = self._run(self._steps_with_species({"group_filter": "off"}))
        self.assertEqual(run.context["s"]["dropped_tracks"], [])
        self.assertEqual(len(run.context["a"]["rows"]), 2)
        self.assertNotIn("filtered_out", run.context["a"])

    def test_a_node_wired_off_to_the_side_filters_nothing(self):
        run, _job = self._run(self._steps_with_species(in_line=False))
        self.assertEqual(run.context["s"]["dropped_tracks"], [1])
        self.assertEqual(len(run.context["a"]["rows"]), 2)

    def test_the_groups_dont_change_the_gpu_job(self):
        steps = self._steps_with_species({"leave_out": '["Bees"]', "group_rules": '{"rules": []}'})
        self.assertEqual(executors._pipeline_species(steps[2], steps), "bioclip")

    def test_the_tracks_csv_has_the_group(self):
        run, job = self._run(self._steps_with_species())
        from apps.pipelines import tracks
        by_track = tracks.with_groups({}, run)
        self.assertEqual(tracks.identity(by_track["1"])["group"], "Ants")
        self.assertEqual(tracks.identity(by_track["1"])["counted"], "no")
        self.assertEqual(tracks.identity(by_track["2"])["counted"], "yes")

    def test_the_player_gets_each_tracks_group(self):
        from apps.analysis.views import _overlay_groups
        run, _job = self._run(self._steps_with_species())
        live = _overlay_groups(run, self.user)
        self.assertEqual(live["groups"]["1"][:2], ["Ants", False])
        self.assertTrue(live["can_regroup"])
        self.assertIn("Bees", live["group_options"])


class RegroupTests(FilterBase):
    """Changing a track's group while reviewing."""

    def _post(self, job, run, body, user=None):
        self.client.force_login(user or self.user)
        return self.client.post(reverse("analysis:track_group", args=[job.pk]) + f"?run={run.pk}",
                                json.dumps(body), content_type="application/json")

    def test_this_track_only_is_counted_again_at_once(self):
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()
        run, job = self._run(self._steps_with_species())

        res = self._post(job, run, {"track": 1, "scope": "track", "group": "Bees"})

        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(res.json()["track"]["group"], "Bees")
        run.refresh_from_db()
        self.assertEqual(len(run.context["a"]["rows"]), 2)
        self.assertEqual(run.context["s"]["track_groups"]["1"]["why"], "set by hand")
        # Saved to the pipeline too, for the next clips.
        self.pipeline.refresh_from_db()
        rules = json.loads(self.pipeline.steps[-1]["config"]["group_rules"])["rules"]
        self.assertEqual(rules[0]["kind"], "track")

    def test_a_species_rule_queues_the_batchs_other_clips(self):
        import uuid
        batch = uuid.uuid4()
        run, job = self._run(self._steps_with_species())
        other, _ = self._run(self._steps_with_species())
        PipelineRun.objects.filter(pk__in=[run.pk, other.pk]).update(batch_id=batch)

        res = self._post(job, run, {"track": 1, "scope": "name", "group": "Bees"})

        self.assertEqual(res.status_code, 200, res.content)
        self.assertEqual(res.json()["rule"],
                         {"kind": "name", "value": "Camponotus pennsylvanicus", "group": "Bees"})
        self.assertEqual(res.json()["queued"], 1)
        other.refresh_from_db()
        self.assertEqual(other.counting_rules, 0)
        self.assertIn("Camponotus pennsylvanicus", other.steps[-1]["config"]["group_rules"])

    def test_a_family_rule_and_undoing_it(self):
        run, job = self._run(self._steps_with_species())
        self._post(job, run, {"track": 1, "scope": "family", "group": "Bees"})
        run.refresh_from_db()
        self.assertEqual(run.context["s"]["dropped_tracks"], [])
        self._post(job, run, {"track": 1, "scope": "family", "group": ""})
        run.refresh_from_db()
        self.assertEqual(run.context["s"]["dropped_tracks"], [1])

    def test_only_the_launcher_can_change_groups(self):
        from django.contrib.auth.models import User
        run, job = self._run(self._steps_with_species())
        stranger = User.objects.create_user("stranger", password="x")
        res = self._post(job, run, {"track": 1, "scope": "track", "group": "Bees"}, user=stranger)
        self.assertEqual(res.status_code, 404)

    def test_the_batch_page_says_what_was_left_out(self):
        from apps.pipelines import review
        run, _job = self._run(self._steps_with_species())
        summary = review.batch_group_summary([run], {})
        self.assertEqual(summary["tracks"], 1)
        self.assertEqual(summary["events"], 2)
        self.assertEqual(summary["by_group"][0]["group"], "Ants")
        self.assertIn("Bees", summary["kept"])


class ApplyToBatchTests(FilterBase):
    """Apply the pipeline's groups to a batch that already ran, in place."""

    def _batch(self, n=2, steps=None):
        import uuid
        batch = uuid.uuid4()
        runs = []
        for _ in range(n):
            run, _job = self._run(steps or self._steps_with_species(in_line=False))
            runs.append(run)
        PipelineRun.objects.filter(pk__in=[r.pk for r in runs]).update(batch_id=batch)
        return batch, runs

    def _post(self, batch):
        self.client.force_login(self.user)
        return self.client.post(reverse("pipelines:batch_apply_groups", args=[batch]))

    def test_the_batch_is_recounted_in_place_with_the_filter(self):
        batch, runs = self._batch()
        self.assertEqual(len(runs[0].context["a"]["rows"]), 2)     # ants counted
        self.pipeline.steps = self._steps_with_species({"model": "beemachine"})
        self.pipeline.save()

        res = self._post(batch)

        self.assertEqual(res.status_code, 302)
        for run in runs:
            run.refresh_from_db()
            self.assertEqual(run.context["a"]["rows"], [])          # ants left out
            self.assertEqual(run.steps[4]["inputs"]["tracks"], "s")
            # The run keeps the model its crops were read with.
            self.assertEqual(run.steps[-1]["config"]["model"], "bioclip")
        self.assertEqual(PipelineRun.objects.filter(batch_id=batch).count(), 2)  # no new runs

    def test_beyond_the_first_few_the_sweep_finishes_the_job(self):
        from apps.pipelines import regroup
        batch, runs = self._batch(n=3)
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()

        done = regroup.apply_pipeline(batch, self.user, now=1)

        self.assertEqual((done["applied"], done["recounted"], done["queued"]), (3, 1, 2))
        self.assertEqual(recount.sweep(), 2)
        for run in runs:
            run.refresh_from_db()
            self.assertEqual(run.context["a"]["rows"], [])

    def test_taking_the_node_out_of_line_counts_every_track_again(self):
        batch, runs = self._batch(n=1, steps=self._steps_with_species())
        self.assertEqual(runs[0].context["a"]["rows"], [])
        self.pipeline.steps = self._steps_with_species(in_line=False)
        self.pipeline.save()

        self._post(batch)

        runs[0].refresh_from_db()
        self.assertEqual(len(runs[0].context["a"]["rows"]), 2)
        self.assertEqual(runs[0].steps[4]["inputs"]["tracks"], "m")

    def test_runs_tracked_without_the_node_are_left_alone(self):
        batch, runs = self._batch(n=1, steps=self._steps("analyze.events"))
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()

        self._post(batch)

        runs[0].refresh_from_db()
        self.assertFalse(any(s["block_type"] == "identify.species" for s in runs[0].steps))

    def test_only_the_launcher_can_apply(self):
        from django.contrib.auth.models import User
        batch, _runs = self._batch(n=1)
        self.client.force_login(User.objects.create_user("other", password="x"))
        res = self.client.post(reverse("pipelines:batch_apply_groups", args=[batch]))
        self.assertEqual(res.status_code, 404)

    def test_the_batch_page_offers_it(self):
        batch, _runs = self._batch(n=1)
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()
        self.client.force_login(self.user)
        res = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertContains(res, "Apply the pipeline's groups to this batch")
        self.assertContains(res, "leaves out ants")
