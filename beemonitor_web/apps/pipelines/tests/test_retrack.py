"""Re-tracking on the web from saved detections (memory/69).

The GPU detects and tracks in one pass; only detection needs the GPU. The web
replays BeeTracker over the saved detections with the MOT node's current
settings, then species votes, review votes and every count follow the new
tracks — no GPU.
"""

import csv
import json
import tempfile
import uuid
from pathlib import Path

from django.test import SimpleTestCase
from django.urls import reverse

from apps.analysis.models import Job
from apps.pipelines import recount, regroup
from apps.pipelines.models import EventReview, PipelineRun
from apps.pipelines.retrack import replay as rp
from apps.pipelines.retrack import run as rt

from .test_primitive_blocks import PrimitiveBlockTestCase

FPS = 30.0
HEIGHT = 480


def box(x, y, size=20):
    return [x, y, x + size, y + size, 0.9, "yolo", "bee"]


def bee_calls():
    """The worker's update() calls for one bee flying right, lost for a
    second (frames 20-49) and seen again: motion mode before frame 5 (empty
    calls), a lookback burst replaying frames 0-5 when motion starts at 5."""
    calls = []
    for f in range(0, 5):
        calls.append((f, [], "current"))                  # motion mode, no YOLO
    for f in range(0, 6):                                   # lookback at frame 5
        calls.append((f, [box(100 + 2 * f, 100)] if f >= 2 else [], "lookback"))
    for f in range(5, 90):
        seen = f < 20 or f >= 50
        calls.append((f, [box(100 + 2 * f, 100)] if seen else [], "current"))
    return calls


def worker_files(calls):
    """What the worker writes for these calls: detections.csv (with seq and
    mode) and update_log.csv."""
    detections, log = [], []
    for seq, (frame, rows, kind) in enumerate(calls):
        log.append({"seq": seq, "frame": frame, "kind": kind})
        for r in rows:
            detections.append({"frame": frame, "x1": r[0], "y1": r[1], "x2": r[2], "y2": r[3],
                               "cx": (r[0] + r[2]) / 2, "cy": (r[1] + r[3]) / 2,
                               "confidence": r[4], "source": r[5], "taxon": r[6],
                               "mode": "lookback" if kind == "lookback" else "tracking",
                               "seq": seq})
    detections.sort(key=lambda d: d["frame"])
    return detections, log


class ReplayTests(SimpleTestCase):
    def test_the_vendored_tracker_is_the_workers(self):
        root = Path(__file__).resolve().parents[4]
        source = (root / "src/beemonitor/tracking/mot/bee_tracker.py").read_text()
        vendored = (Path(rp.__file__).parent / "bee_tracker.py").read_text()
        self.assertTrue(vendored.endswith(source), "re-copy src/beemonitor/tracking/mot/bee_tracker.py")

    def test_with_the_frame_log_the_replay_is_exact(self):
        params = rp.tracker_params({"max_age_seconds": 3.0, "max_resurrection_seconds": 0.0})
        direct = rp.tracking_rows(rp.run_calls(bee_calls(), FPS, HEIGHT, params))
        detections, log = worker_files(bee_calls())
        replayed = rp.replay(detections, FPS, HEIGHT, params, 0, 89, log)
        self.assertEqual(replayed, direct)

    def test_without_the_log_the_calls_are_rebuilt_from_the_lookback_rows(self):
        detections, _log = worker_files(bee_calls())
        for d in detections:
            d.pop("seq")
        calls = rp.calls_from_detections(detections, FPS, 0, 89)
        # Every frame once as current, and the 15-frame lookback before frame 5.
        self.assertEqual(sum(1 for c in calls if c[2] == "current"), 90)
        lookback = [c[0] for c in calls if c[2] == "lookback"]
        self.assertEqual(lookback, list(range(0, 6)))
        self.assertEqual(calls.index((5, [box(110, 100)], "current")), 5 + 6)

    def test_a_shorter_lost_window_splits_the_bee(self):
        detections, log = worker_files(bee_calls())
        long_ = rp.replay(detections, FPS, HEIGHT, rp.tracker_params(
            {"max_age_seconds": 3.0, "max_resurrection_seconds": 0.0}), 0, 89, log)
        short = rp.replay(detections, FPS, HEIGHT, rp.tracker_params(
            {"max_age_seconds": 0.5, "max_resurrection_seconds": 0.0}), 0, 89, log)
        self.assertEqual(len({r["track_id"] for r in long_}), 1)
        self.assertEqual(len({r["track_id"] for r in short}), 2)

    def test_settings_come_from_the_worker_defaults_and_the_node(self):
        p = rp.tracker_params({"max_age_seconds": 1, "frames_old": 3})
        self.assertEqual(p["max_age_seconds"], 1.0)
        self.assertEqual(p["min_hits_seconds"], 0.0)
        self.assertEqual(p["match_distance_multiplier"], 8.0)
        self.assertNotIn("frames_old", p)

    def test_crops_go_to_the_new_track_with_the_same_box(self):
        old = [{"track_id": 7, "frame": 3, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "detected": 1},
               {"track_id": 7, "frame": 4, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "detected": 0}]
        new = [{"track_id": 2, "frame": 3, "x1": 1, "y1": 0, "x2": 11, "y2": 10, "detected": 1},
               {"track_id": 5, "frame": 3, "x1": 50, "y1": 50, "x2": 60, "y2": 60, "detected": 1}]
        self.assertEqual(rt.crop_mapping(old, new), {("7", 3): "2"})


def write_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return str(path)


class RetrackRunTests(PrimitiveBlockTestCase):
    """A batch whose MOT node changed is re-tracked on the web and recounted."""

    def setUp(self):
        super().setUp()
        self.video.width, self.video.height, self.video.duration_seconds = 640, HEIGHT, 3.0
        self.video.save()
        folder = Path(tempfile.mkdtemp())
        detections, log = worker_files(bee_calls())
        gpu = rp.replay(detections, FPS, HEIGHT, rp.tracker_params(
            {"max_age_seconds": 3.0, "max_resurrection_seconds": 0.0}), 0, 89, log)
        self.gpu_tracking = write_csv(gpu, folder / "tracking.csv")
        for d in detections:
            d.pop("seq")              # a clip from before the frame log
        self.detections = write_csv(detections, folder / "detections.csv")
        tid = gpu[0]["track_id"]
        self.votes = write_csv([{"track_id": tid, "frame": f, "crop": f"frame_{f}.jpg",
                                 "species": "Bombus impatiens", "species_confidence": 0.8}
                                for f in (2, 10, 60, 70)], folder / "track_votes.csv")
        self.crops = write_csv([{"track_id": tid, "frame": f, "crop_key": f"k{f}", "sharpness": 1}
                                for f in (10, 60)], folder / "track_crops.csv")

    def _steps(self, max_age=3.0):
        steps = super()._steps("analyze.events")
        steps[2]["config"] = {"tracker": "beetrack", "beetrack_max_age_seconds": max_age}
        return steps

    def _run(self, batch=None):
        from apps.analysis.models import JobResult
        job = Job.objects.create(user=self.user, video=self.video, status=Job.Status.COMPLETED,
                                 modal_job_id=uuid.uuid4().hex,
                                 config={"tracker": "beetrack",
                                         "tracker_params": {"max_age_seconds": 3.0,
                                                            "max_resurrection_seconds": 0.0}})
        JobResult.objects.create(job=job, tracking_csv_path=self.gpu_tracking)
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user,
                                         steps=self._steps(), batch_id=batch,
                                         status=PipelineRun.Status.COMPLETED)
        result = {"tracking_csv_path": self.gpu_tracking, "detections_csv_path": self.detections,
                  "crops_csv_path": self.crops, "unique_tracks": 1,
                  "summary_stats": {"video_fps": FPS, "track_votes_csv": self.votes,
                                    "crops_manifest": {"1": ["k10", "k60"]}}}
        run.context = {"v": {"artifact": "video", "video_id": self.video.pk},
                       "m": {"artifact": "tracks", "job_id": job.pk, "result": result},
                       "a": {"artifact": "events", "table_kind": "events", "rows": []}}
        run.save()
        recount.recount_run(run)
        run.refresh_from_db()
        return run, job

    def test_an_unchanged_run_is_not_re_tracked(self):
        run, _job = self._run()
        self.assertFalse(rt.needs_retrack(run))
        self.assertNotIn("retracked", run.context["m"]["result"])

    def test_update_this_batch_re_tracks_with_the_new_window(self):
        batch = uuid.uuid4()
        run, job = self._run(batch)
        self.pipeline.steps = self._steps(max_age=0.5)
        self.pipeline.save()

        changes = regroup.batch_changes([run], self.pipeline)
        self.assertEqual(changes["gpu"], [])
        self.assertIn("tracks again on the web", changes["web"][0]["after"])

        done = regroup.apply_pipeline(batch, self.user)
        self.assertEqual(done.get("retrack"), 1)          # queued, not done in the request
        recount.sweep()

        run.refresh_from_db()
        result = run.context["m"]["result"]
        self.assertEqual(result["retracked"]["tracks"], 2)
        self.assertFalse(result["retracked"]["exact"])
        self.assertEqual(run.context["m"]["gpu_result"]["tracking_csv_path"], self.gpu_tracking)
        # The GPU's own record is untouched; the run points at its own tables.
        job.refresh_from_db()
        self.assertNotEqual(result["tracking_csv_path"], self.gpu_tracking)
        # Species votes and crops followed the bee into both new tracks.
        by_track = result["summary_stats"]["identification"]["by_track"]
        self.assertEqual(len(by_track), 2)
        self.assertEqual({v["taxon"] for v in by_track.values()}, {"Bombus impatiens"})
        self.assertEqual(sorted(len(k) for k in result["summary_stats"]["crops_manifest"].values()),
                         [1, 1])
        # Settings recorded, so the run isn't re-tracked again.
        self.assertFalse(rt.needs_retrack(run))
        self.assertEqual(run.steps[2]["config"]["_gpu_mot"]["beetrack_max_age_seconds"], 3.0)

    def test_the_tube_rule_re_tracks_on_the_web_with_the_runs_tubes(self):
        batch = uuid.uuid4()
        run, _job = self._run(batch)
        steps = self._steps()
        steps[2]["config"]["beetrack_stay_in_reference"] = "on"
        steps[2]["config"]["beetrack_lost_search"] = "speed"
        self.pipeline.steps = steps
        self.pipeline.save()
        self.assertEqual(regroup.batch_changes([run], self.pipeline)["gpu"], [])
        regroup.apply_pipeline(batch, self.user)
        recount.sweep()
        run.refresh_from_db()
        done = run.context["m"]["result"]["retracked"]
        self.assertTrue(done["settings"]["stay_in_reference"])
        self.assertEqual(done["settings"]["lost_search"], "speed")
        self.assertTrue(rt._reference_boxes(run, self.video))

    def test_tracks_made_under_older_lost_track_rules_are_offered_again(self):
        # memory/72: same settings, newer rules behind them.
        batch = uuid.uuid4()
        run, _job = self._run(batch)
        steps = self._steps()
        steps[2]["config"]["beetrack_lost_search"] = "speed"
        self.pipeline.steps = steps
        self.pipeline.save()
        regroup.apply_pipeline(batch, self.user)
        recount.sweep()
        run.refresh_from_db()
        self.assertFalse(rt.needs_retrack(run))
        self.assertIsNone(regroup.batch_changes([run], self.pipeline))

        # As a re-track before the rules changed recorded it.
        ctx = run.context
        ctx["m"]["result"]["retracked"]["settings"].pop("lost_rules")
        PipelineRun.objects.filter(pk=run.pk).update(context=ctx)
        run.refresh_from_db()
        self.assertTrue(rt.needs_retrack(run))
        changes = regroup.batch_changes([run], self.pipeline)
        self.assertEqual(changes["web"][0]["field"], "Lost-track rules")
        self.assertEqual(changes["gpu"], [])

    def test_review_votes_follow_their_event(self):
        run, _job = self._run()
        # A vote on an event as the GPU tracks had it, on a track id that the
        # re-track renumbers.
        EventReview.objects.create(run=run, event_key="exit|reference_1|99|40", action="exit",
                                   reference="reference_1", track_id="99", frame=40,
                                   user=self.user, verdict="accepted")
        from apps.pipelines.retrack import votes
        run.context["a"]["rows"] = [{"action": "exit", "reference": "reference_1",
                                     "track_id": 3, "frame": 42}]
        moved = votes.move_votes(run, FPS)
        self.assertEqual(moved, {"moved": 1, "gone": 0})
        v = EventReview.objects.get(run=run)
        self.assertEqual((v.event_key, v.track_id, v.frame), ("exit|reference_1|3|42", "3", 42))

    def test_the_clip_pages_read_the_runs_tracks(self):
        batch = uuid.uuid4()
        run, job = self._run(batch)
        self.pipeline.steps = self._steps(max_age=0.5)
        self.pipeline.save()
        regroup.apply_pipeline(batch, self.user)
        recount.sweep()
        run.refresh_from_db()
        view = rt.effective_result(job, run)
        self.assertEqual(view.tracking_csv_path, run.context["m"]["result"]["tracking_csv_path"])
        # Fields the re-track doesn't change still come from the GPU's record.
        self.assertEqual(view.job_id, job.pk)
        job.refresh_from_db()
        self.assertEqual(job.result.tracking_csv_path, self.gpu_tracking)

    def test_a_clip_with_no_saved_detections_says_so(self):
        batch = uuid.uuid4()
        run, _job = self._run(batch)
        result = dict(run.context["m"]["result"], detections_csv_path="")
        run.context["m"]["result"] = result
        run.save()
        self.pipeline.steps = self._steps(max_age=0.5)
        self.pipeline.save()
        regroup.apply_pipeline(batch, self.user)
        recount.sweep()
        run.refresh_from_db()
        self.assertIn("no saved detections", run.context["m"]["retrack_note"])

    def test_the_replay_check_agrees_with_itself(self):
        from apps.pipelines.retrack import check
        run, _job = self._run()
        c = check.compare(run)
        self.assertEqual(c["boxes"], 1.0)
        self.assertEqual(c["same_track"], 1.0)
        self.assertEqual(c["gpu_tracks"], c["tracks"])

    def test_the_batch_page_shows_the_update_and_the_check(self):
        batch = uuid.uuid4()
        self._run(batch)
        self.pipeline.steps = self._steps(max_age=0.5)
        self.pipeline.save()
        self.client.force_login(self.user)
        res = self.client.get(reverse("pipelines:batch_detail", args=[batch]))
        self.assertContains(res, "Keep a lost track (seconds): 3.0 → 0.5 (tracks again on the web)")
        res = self.client.post(reverse("pipelines:batch_retrack_check", args=[batch]))
        self.assertTrue(res.json()["ok"], res.content)
        self.assertEqual(res.json()["check"]["boxes"], 100)

    def test_after_a_recount_votes_follow_the_same_tracks_event_even_if_it_moved(self):
        # "Track start and end from: whole trajectory" moved an entry 3 s later.
        run, _job = self._run()
        EventReview.objects.create(run=run, event_key="enter|reference_1|4|100", action="enter",
                                   reference="reference_1", track_id="4", frame=100,
                                   user=self.user, verdict="rejected")
        from apps.pipelines.retrack import votes
        run.context["a"]["rows"] = [
            {"action": "enter", "reference": "reference_1", "track_id": 4, "frame": 190},
            {"action": "enter", "reference": "reference_1", "track_id": 9, "frame": 101}]
        self.assertEqual(votes.move_votes(run, FPS, same_tracks=True), {"moved": 1, "gone": 0})
        self.assertEqual(EventReview.objects.get(run=run).event_key, "enter|reference_1|4|190")

    def test_after_a_re_track_a_same_numbered_track_is_not_trusted(self):
        run, _job = self._run()
        EventReview.objects.create(run=run, event_key="enter|reference_1|4|100", action="enter",
                                   reference="reference_1", track_id="4", frame=100,
                                   user=self.user, verdict="rejected")
        from apps.pipelines.retrack import votes
        run.context["a"]["rows"] = [
            {"action": "enter", "reference": "reference_1", "track_id": 4, "frame": 190},
            {"action": "enter", "reference": "reference_1", "track_id": 9, "frame": 101}]
        votes.move_votes(run, FPS, same_tracks=False)
        self.assertEqual(EventReview.objects.get(run=run).event_key, "enter|reference_1|9|101")


def ids_by_phase(calls, params, references=None, split=31):
    """Track ids seen before and from ``split`` on."""
    rows = rp.tracking_rows(rp.run_calls(calls, FPS, HEIGHT, params, references))
    before = {r["track_id"] for r in rows if r["frame"] < split and r["detected"]}
    after = {r["track_id"] for r in rows if r["frame"] >= split and r["detected"]}
    return before, after


TUBE_A, TUBE_B = (228, 88, 262, 122), (318, 88, 352, 122)


def teleport_calls():
    """A bee flies right, slows and stops in tube A (frames 0-20), is lost
    in it; another bee appears in tube B, 90 px away, at frame 31."""
    calls, x = [], 100.0
    for f in range(0, 21):
        step = 10.0 if f < 10 else max(0.0, 10.0 - (f - 9) * 1.0)
        x += step
        calls.append((f, [box(min(x, 235), 95)], "current"))
    for f in range(21, 31):
        calls.append((f, [], "current"))
    for f in range(31, 45):
        calls.append((f, [box(325, 95)], "current"))
    return calls


class LostSearchTests(SimpleTestCase):
    """Lost tracks jumping to another bee (memory/71)."""

    BASE = {"max_age_seconds": 3.0, "max_resurrection_seconds": 0.0}

    def test_today_a_lost_track_takes_the_bee_at_the_next_tube(self):
        before, after = ids_by_phase(teleport_calls(), rp.tracker_params(self.BASE))
        self.assertEqual(before, after)                       # the teleport

    def test_following_its_speed_a_stopped_bee_stays_put(self):
        before, after = ids_by_phase(teleport_calls(),
                                     rp.tracker_params({**self.BASE, "lost_search": "speed"}))
        self.assertTrue(before and after and not (before & after))

    def test_a_track_lost_at_a_tube_stays_there(self):
        params = rp.tracker_params({**self.BASE, "stay_in_reference": "on"})
        before, after = ids_by_phase(teleport_calls(), params, [TUBE_A, TUBE_B])
        self.assertTrue(before and after and not (before & after))

    def test_a_fast_bee_lost_mid_flight_is_still_found(self):
        calls = [(f, [box(100 + 15 * f, 200)], "current") for f in range(0, 20)]
        calls += [(f, [], "current") for f in range(20, 25)]
        calls += [(f, [box(100 + 15 * f, 200)], "current") for f in range(25, 35)]
        before, after = ids_by_phase(calls, rp.tracker_params({**self.BASE, "lost_search": "speed"}),
                                     split=25)
        self.assertEqual(before, after)

    # A track that dies (max age) and is revived: the same jump by the
    # resurrection road, which must follow the settings too.
    REVIVE = {"max_age_seconds": 0.1, "max_resurrection_seconds": 2.0,
              "resurrection_search_multiplier": 8.0}

    def test_today_a_dead_track_is_revived_at_the_next_tube(self):
        before, after = ids_by_phase(teleport_calls(), rp.tracker_params(self.REVIVE))
        self.assertEqual(before, after)

    def test_revival_follows_its_speed(self):
        before, after = ids_by_phase(teleport_calls(),
                                     rp.tracker_params({**self.REVIVE, "lost_search": "speed"}))
        self.assertTrue(before and after and not (before & after))

    def test_revival_stays_at_its_tube(self):
        params = rp.tracker_params({**self.REVIVE, "stay_in_reference": "on"})
        before, after = ids_by_phase(teleport_calls(), params, [TUBE_A, TUBE_B])
        self.assertTrue(before and after and not (before & after))

    # memory/72 — a fast bee flies off right and is lost; another bee shows up
    # behind it, where it came from. The reach is a circle; the heading isn't.
    def fast_then_behind(self):
        calls = [(f, [box(100 + 15 * f, 200)], "current") for f in range(0, 20)]
        calls += [(f, [], "current") for f in range(20, 28)]
        calls += [(f, [box(330, 200)], "current") for f in range(28, 40)]
        return calls

    def test_a_fast_bee_is_looked_for_ahead_not_behind(self):
        before, after = ids_by_phase(self.fast_then_behind(),
                                     rp.tracker_params({**self.BASE, "lost_search": "speed"}),
                                     split=28)
        self.assertTrue(before and after and not (before & after))

    def test_a_bee_out_of_the_next_tube_is_another_bee(self):
        # Tube N starts 4 px right of tube A: inside A's box widened by a bee.
        tube_n = (266, 88, 300, 122)
        calls = [c for c in teleport_calls() if c[0] < 31]
        calls += [(f, [box(270, 95)], "current") for f in range(31, 45)]
        params = rp.tracker_params({**self.BASE, "lost_search": "speed", "stay_in_reference": "on"})
        before, after = ids_by_phase(calls, params, [TUBE_A, tube_n])
        self.assertTrue(before and after and not (before & after))

    def test_the_settings_reach_the_tracker_from_the_mot_node(self):
        from apps.pipelines import executors
        sent = executors.tracker_settings({"tracker": "beetrack", "beetrack_lost_search": "speed",
                                           "beetrack_stay_in_reference": "on",
                                           "beetrack_match_distance_multiplier": 5})
        p = rp.tracker_params(sent)
        self.assertEqual((p["lost_search"], p["stay_in_reference"], p["match_distance_multiplier"]),
                         ("speed", True, 5.0))
        # Left at their defaults they aren't sent: GPU cache keys don't change.
        self.assertNotIn("lost_search", executors.tracker_settings({"tracker": "beetrack"}))
