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
        self.assertEqual(groups[2]["why"], "only 2 of 2 crops named something in a group")
        self.assertTrue(groups[2]["counted"])

    def test_a_name_gbif_cant_place_is_unclear(self):
        g = taxon_groups.track_groups({1: ["Not a real thing"] * 5}, cfg())[1]
        self.assertEqual(g["group"], "Unclear")
        self.assertTrue(g["counted"])

    def test_a_failed_lookup_is_unclear_and_retried_later(self):
        self.gbif.side_effect = lambda name: None
        g = taxon_groups.track_groups({1: ["Vespula maculifrons"] * 5}, cfg())[1]
        self.assertEqual(g["group"], "Unclear")
        self.gbif.side_effect = fake_gbif
        g = taxon_groups.track_groups({1: ["Vespula maculifrons"] * 5}, cfg())[1]
        self.assertEqual(g["group"], "Wasps")

    def test_a_bee_genus_is_a_bee_even_when_gbif_cant_be_asked(self):
        self.gbif.side_effect = lambda name: None
        g = taxon_groups.track_groups({1: ["Megachile sculpturalis"] * 4}, cfg())[1]
        self.assertEqual(g["group"], "Bees")

    def test_a_cached_no_match_for_a_bee_genus_is_still_a_bee(self):
        cache.set(taxon_groups._cache_key("Megachile sculpturalis"), {}, 60)
        g = taxon_groups.track_groups({1: ["Megachile sculpturalis"] * 4}, cfg())[1]
        self.assertEqual(g["group"], "Bees")

    def test_crops_named_outside_every_group_dont_dilute_the_vote(self):
        # Track 4 on natalies: a Megachile whose other crops were named
        # things no group covers (a bat, names GBIF doesn't know).
        crops = (["Megachile sculpturalis"] * 3 + ["Myotis lucifugus"] * 4
                 + ["Not a real thing"] * 3 + ["Periplaneta americana"])
        g = taxon_groups.track_groups({4: crops}, cfg())[4]
        self.assertEqual(g["group"], "Bees")
        self.assertEqual(g["share"], 0.75)

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
        # A pipeline saved with the memory/65 rules is converted on read.
        groups = taxon_groups.track_groups(
            {1: ["Ectemnius continuus"] * 4, 2: ["Bombus impatiens"] * 4,
             3: ["Ectemnius lapidarius"] * 4},
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

    def test_with_rule_moves_a_taxon_and_undo_puts_it_back(self):
        c = taxon_groups.with_rule({}, {"kind": "name", "value": "Lasius niger", "group": "Tiny things"})
        g = taxon_groups.grouping_of(c)
        tiny = next(x for x in g["groups"] if x["name"] == "Tiny things")
        self.assertEqual(tiny["members"], [{"rank": "species", "name": "Lasius niger"}])
        c = taxon_groups.with_rule(c, {"kind": "family", "value": "Formicidae", "group": "Bees"})
        ants = next(x for x in taxon_groups.grouping_of(c)["groups"] if x["name"] == "Ants")
        self.assertEqual(ants["members"], [])
        c = taxon_groups.with_rule(c, {"kind": "family", "value": "Formicidae", "group": ""})
        ants = next(x for x in taxon_groups.grouping_of(c)["groups"] if x["name"] == "Ants")
        self.assertEqual(ants["members"], [{"rank": "family", "name": "Formicidae"}])

    def test_the_most_specific_taxon_wins(self):
        grouping = taxon_groups.default_grouping()
        taxon_groups.put_member(grouping, "genus", "Camponotus", "Bees")
        c = cfg(grouping=json.dumps(grouping))
        groups = taxon_groups.track_groups(
            {1: ["Camponotus pennsylvanicus"] * 4, 2: ["Lasius niger"] * 4,
             3: ["Vespula maculifrons"] * 4}, c)
        self.assertEqual(groups[1]["group"], "Bees")      # genus beats family
        self.assertEqual(groups[2]["group"], "Ants")      # family beats order
        self.assertEqual(groups[3]["group"], "Wasps")     # order Hymenoptera
        self.assertEqual(groups[1]["names"], {"Camponotus pennsylvanicus": 4})

    def test_the_default_grouping_is_written_out(self):
        g = taxon_groups.default_grouping()
        bees = next(x for x in g["groups"] if x["name"] == "Bees")
        self.assertIn({"rank": "family", "name": "Megachilidae"}, bees["members"])
        self.assertEqual([x["name"] for x in g["groups"]], taxon_groups.BUILTIN)


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
        steps = self._steps_with_species({"leave_out": '["Bees"]', "grouping": '{"groups": []}'})
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
        tracks = json.loads(self.pipeline.steps[-1]["config"]["grouping"])["tracks"]
        self.assertEqual(tracks[0]["track"], 1)

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
        self.assertIn("Camponotus pennsylvanicus", other.steps[-1]["config"]["grouping"])

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
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()

        res = self._post(batch)

        self.assertEqual(res.status_code, 302)
        for run in runs:
            run.refresh_from_db()
            self.assertEqual(run.context["a"]["rows"], [])          # ants left out
            self.assertEqual(run.steps[4]["inputs"]["tracks"], "s")
        self.assertEqual(PipelineRun.objects.filter(batch_id=batch).count(), 2)  # no new runs

    def test_a_change_that_needs_the_gpu_runs_the_chosen_clips_again_in_place(self):
        # memory/76: e.g. a new species model; only the picked clip, same run.
        batch, runs = self._batch()
        self.pipeline.steps = self._steps_with_species({"model": "beemachine"})
        self.pipeline.save()
        self.client.force_login(self.user)
        with mock.patch("apps.pipelines.engine.start_run") as start, \
                mock.patch("apps.analysis.views._drain_queue"):
            res = self.client.post(reverse("pipelines:batch_apply_groups", args=[batch]),
                                   {"which": "picked", "run": [runs[0].pk]})
        self.assertEqual(res.status_code, 302)
        self.assertEqual(start.call_count, 1)
        run, kwargs = start.call_args[0][0], start.call_args[1]
        self.assertEqual(run.pk, runs[0].pk)
        self.assertEqual(kwargs["steps"][-1]["config"]["model"], "beemachine")
        self.assertEqual(kwargs["steps"][0]["config"]["video_id"], str(self.video.pk))
        self.assertTrue(PipelineRun.objects.get(pk=runs[0].pk).rerun_votes)
        self.assertFalse(PipelineRun.objects.get(pk=runs[1].pk).rerun_votes)
        self.assertEqual(PipelineRun.objects.filter(batch_id=batch).count(), 2)  # no new runs

    def test_a_re_run_keeps_its_part_of_the_clip(self):
        from apps.pipelines import regroup
        steps = self._steps_with_species()
        steps[0]["config"].update(start_seconds=60, end_seconds=120)
        self.assertEqual(regroup._window_of(steps), (60.0, 120.0))
        self.assertIsNone(regroup._window_of(self._steps_with_species()))

    def test_votes_follow_to_the_new_events_when_the_re_run_finishes(self):
        from apps.pipelines import engine
        from apps.pipelines.models import EventReview
        from apps.pipelines.review import event_key

        batch, runs = self._batch(n=1, steps=self._steps_with_species(in_line=False))
        run = PipelineRun.objects.get(pk=runs[0].pk)
        row = run.context["a"]["rows"][0]
        EventReview.objects.create(run=run, event_key="x|old|99|" + str(row["frame"]),
                                   action=row["action"], reference=row["reference"],
                                   track_id="99", frame=row["frame"], user=self.user,
                                   verdict="accepted")
        run.rerun_votes = True
        engine._move_rerun_votes(run)
        vote = EventReview.objects.get()
        self.assertEqual(vote.event_key, event_key(row))
        self.assertFalse(PipelineRun.objects.get(pk=run.pk).rerun_votes)

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

    # memory/72: update only some clips.
    def _update(self, batch, **data):
        self.client.force_login(self.user)
        return self.client.post(reverse("pipelines:batch_apply_groups", args=[batch]), data)

    def test_only_the_ticked_clips_are_updated(self):
        batch, runs = self._batch()
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()

        self._update(batch, which="picked", run=[runs[0].pk])

        for run in runs:
            run.refresh_from_db()
        self.assertEqual(runs[0].context["a"]["rows"], [])          # updated
        self.assertEqual(len(runs[1].context["a"]["rows"]), 2)      # as it was
        from apps.pipelines import regroup
        self.assertEqual(regroup.pipeline_changes(runs[0], self.pipeline)["web"], [])
        self.assertTrue(regroup.pipeline_changes(runs[1], self.pipeline)["web"])

    def test_only_the_reviewed_clips_are_updated(self):
        from apps.pipelines.models import ClipReview
        batch, runs = self._batch()
        ClipReview.objects.create(run=runs[1], user=self.user)
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()

        self.client.force_login(self.user)
        page = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertEqual(page.context["update_counts"], {"all": 2, "reviewed": 1})
        self.assertContains(page, "data-pick ", count=2)

        self._update(batch, which="reviewed")

        for run in runs:
            run.refresh_from_db()
        self.assertEqual(len(runs[0].context["a"]["rows"]), 2)
        self.assertEqual(runs[1].context["a"]["rows"], [])

    def test_ticking_nothing_updates_nothing(self):
        batch, runs = self._batch(n=1)
        self.pipeline.steps = self._steps_with_species({"model": "beemachine"})
        self.pipeline.save()
        self._update(batch, which="picked")
        runs[0].refresh_from_db()
        self.assertEqual(len(runs[0].context["a"]["rows"]), 2)

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
        self.assertContains(res, "Update this batch")
        # The batch ran with Identify Species off to the side; the pipeline
        # now wires it into Events.
        self.assertContains(res, "Events · Wiring: changed")


class GroupingEditorTests(FilterBase):
    """The grouping's own page (memory/66)."""

    def setUp(self):
        super().setUp()
        self.pipeline.steps = self._steps_with_species()
        self.pipeline.save()
        self.client.force_login(self.user)

    def _url(self, name="pipelines:grouping"):
        return reverse(name, args=[self.pipeline.pk, "s"])

    def test_the_page_lists_the_groups_and_the_names_seen(self):
        self._run(self._steps_with_species())
        res = self.client.get(self._url())
        self.assertEqual(res.status_code, 200)
        page = json.loads(res.context["page_json"])
        self.assertEqual([g["name"] for g in page["grouping"]["groups"]], taxon_groups.BUILTIN)
        self.assertIsNone(page["names"])          # fetched after the page
        res = self.client.get(page["names_url"])
        self.assertEqual(res.json()["pending"], 0)
        names = {n["name"]: n for n in res.json()["names"]}
        self.assertEqual(names["Camponotus pennsylvanicus"]["family"], "Formicidae")
        self.assertEqual(names["Camponotus pennsylvanicus"]["tracks"], 1)

    def test_names_are_looked_up_a_few_at_a_time(self):
        from apps.pipelines import grouping_views
        self._run(self._steps_with_species())
        cache.clear()       # the run looked them up; start cold
        with mock.patch.object(grouping_views, "LOOKUPS_PER_REQUEST", 1):
            first = self.client.get(reverse("pipelines:grouping_names", args=[self.pipeline.pk, "s"])).json()
            second = self.client.get(reverse("pipelines:grouping_names", args=[self.pipeline.pk, "s"])).json()
        self.assertEqual(first["pending"], 1)
        self.assertEqual(sum(1 for n in first["names"] if n["pending"]), 1)
        self.assertEqual(second["pending"], 0)

    def test_saving_writes_the_grouping_into_the_pipeline(self):
        grouping = taxon_groups.default_grouping()
        grouping["groups"].append({"name": "Tiny things", "color": "#7c3aed",
                                   "members": [{"rank": "family", "name": "Formicidae"}]})
        res = self.client.post(self._url(), json.dumps({"grouping": grouping}),
                               content_type="application/json")
        self.assertEqual(res.status_code, 200, res.content)
        self.pipeline.refresh_from_db()
        saved = taxon_groups.grouping_of(self.pipeline.steps[-1]["config"])
        self.assertEqual(saved["groups"][-1]["name"], "Tiny things")

    def test_the_preview_counts_tracks_and_events_under_a_new_grouping(self):
        import uuid
        run, _job = self._run(self._steps_with_species())
        batch = uuid.uuid4()
        PipelineRun.objects.filter(pk=run.pk).update(batch_id=batch)
        grouping = taxon_groups.default_grouping()
        taxon_groups.put_member(grouping, "family", "Formicidae", "Bees")

        res = self.client.post(self._url("pipelines:grouping_preview"),
                               json.dumps({"grouping": grouping, "batch": str(batch)}),
                               content_type="application/json")

        data = res.json()
        self.assertTrue(data["ok"], data)
        rows = {r["group"]: r for r in data["rows"]}
        self.assertEqual((rows["Ants"]["now"], rows["Ants"]["after"]), (1, 0))
        self.assertEqual((rows["Bees"]["now"], rows["Bees"]["after"]), (1, 2))
        self.assertEqual((data["events_now"], data["events_after"]), (0, 2))
        run.refresh_from_db()
        self.assertEqual(run.context["a"]["rows"], [])          # nothing saved

    def test_only_the_owner_saves(self):
        from django.contrib.auth.models import User
        self.pipeline.is_published = True if hasattr(self.pipeline, "is_published") else False
        other = User.objects.create_user("viewer", password="x")
        self.client.force_login(other)
        res = self.client.post(self._url(), json.dumps({"grouping": {"groups": []}}),
                               content_type="application/json")
        self.assertEqual(res.status_code, 404)

    def test_the_canvas_gets_old_rules_written_out_as_a_grouping(self):
        steps = self._steps_with_species({"group_rules": json.dumps(
            {"rules": [{"kind": "name", "value": "Lasius niger", "group": "Bees"}]})})
        self.pipeline.steps = steps
        self.pipeline.save()
        res = self.client.get(reverse("pipelines:editor", args=[self.pipeline.pk]))
        nodes = json.loads(res.context["initial_steps_json"])
        species = next(n for n in nodes if n["block_type"] == "identify.species")
        self.assertIn("Lasius niger", species["config"]["grouping"])


class BatchCardTests(FilterBase):
    """The batch card says how the batch was counted, apart from what the
    pipeline now says."""

    def test_turning_the_filter_off_in_the_pipeline_says_to_apply_it(self):
        import uuid
        run, _job = self._run(self._steps_with_species())
        batch = uuid.uuid4()
        PipelineRun.objects.filter(pk=run.pk).update(batch_id=batch)
        self.pipeline.steps = self._steps_with_species({"group_filter": "off"})
        self.pipeline.save()
        self.client.force_login(self.user)

        res = self.client.get(reverse("pipelines:batch_detail", args=[batch]))

        self.assertContains(res, "The pipeline's filter is now off, but this batch was counted with it on")
        self.assertContains(res, "Filter by insect group: On: leave out the groups ticked → Off: name groups, count every track")
        self.assertContains(res, "This batch was counted leaving out ants, beetles")

        self.client.post(reverse("pipelines:batch_apply_groups", args=[batch]))
        res = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertContains(res, "This batch counts every track")

    def test_clips_counted_with_different_settings_are_reported_apart(self):
        from apps.pipelines import review
        a, _ = self._run(self._steps_with_species())
        b, _ = self._run(self._steps_with_species({"leave_out": '["Beetles"]'}))
        c, _ = self._run(self._steps_with_species())
        summary = review.batch_group_summary([a, b, c], {})
        self.assertEqual(summary["clips"], 3)
        self.assertEqual(summary["other_clips"], 1)
        self.assertIn("Ants", summary["left_out"])
        self.assertNotIn("Ants", summary["kept"])


class EventsSettingTests(FilterBase):
    """Recounting a batch with the pipeline's Events settings, in place."""

    def test_the_events_setting_is_applied_to_the_batch(self):
        import uuid
        run, _job = self._run(self._steps("analyze.events") + [])
        batch = uuid.uuid4()
        PipelineRun.objects.filter(pk=run.pk).update(batch_id=batch)
        steps = self._steps("analyze.events", {"track_rows": "all"})
        self.pipeline.steps = steps
        self.pipeline.save()
        self.client.force_login(self.user)

        res = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertContains(res, "Track start and end from: Detections only → Whole trajectory")
        self.client.post(reverse("pipelines:batch_apply_groups", args=[batch]))

        run.refresh_from_db()
        self.assertEqual(run.steps[4]["config"]["track_rows"], "all")
        self.assertEqual(run.context["a"]["track_rows"], "all")

    def test_an_unchanged_batch_is_up_to_date_and_a_gpu_change_is_listed_apart(self):
        import uuid
        from apps.pipelines import regroup
        steps = self._steps_with_species()
        run, _job = self._run(steps)
        batch = uuid.uuid4()
        PipelineRun.objects.filter(pk=run.pk).update(batch_id=batch)
        self.pipeline.steps = steps
        self.pipeline.save()
        run.refresh_from_db()
        self.assertIsNone(regroup.batch_changes([run], self.pipeline))

        changed = self._steps_with_species({"model": "beemachine"})
        self.pipeline.steps = changed
        self.pipeline.save()
        changes = regroup.batch_changes([run], self.pipeline)
        self.assertEqual(changes["web"], [])
        self.assertEqual(changes["gpu"][0]["field"], "Model")
