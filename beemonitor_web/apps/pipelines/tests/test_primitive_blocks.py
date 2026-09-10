"""The Events and Interactions blocks, end to end through the executor.

The claim these tests defend is the one the whole collapse rests on: run the
two primitives over the same clip and they describe the same episodes, and the
retired Visitation block — which now rolls up from the same pass — agrees with
both. Before, those were three traversals that had no way to be held to each
other.
"""

import csv
import tempfile
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.devices.models import Device
from apps.pipelines import executors
from apps.pipelines.models import Pipeline, PipelineRun
from apps.videos.models import Video

User = get_user_model()

# Track 1 sits inside the nest tube (0.1-0.3) for 20 frames, leaves, and comes
# back for 10 more. Track 2 never enters it.
TRACKING_ROWS = (
    [{"frame": f, "track_id": 1, "cx": 0.2, "cy": 0.2} for f in range(0, 20)]
    + [{"frame": f, "track_id": 1, "cx": 0.8, "cy": 0.8} for f in range(20, 60)]
    + [{"frame": f, "track_id": 1, "cx": 0.2, "cy": 0.2} for f in range(60, 70)]
    + [{"frame": f, "track_id": 2, "cx": 0.9, "cy": 0.9} for f in range(0, 30)]
)

WORKER_EVENTS = [
    {"frame": 5, "action": "Entry", "nest": "nest_9", "track_id": 4},
    {"frame": 90, "action": "Exit", "nest": "nest_9", "track_id": 4},
]


def _write_csv(rows, name):
    path = Path(tempfile.mkdtemp()) / name
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


class PrimitiveBlockTestCase(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("prim", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="Hotel", key_hash="hp", prefix="bmk_p",
            roi_override=[0.0, 0.0, 0.6, 0.6],
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}],
        )
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="prim/clip.mp4", file_size_bytes=1,
            status=Video.Status.READY,
        )
        self.pipeline = Pipeline.objects.create(user=self.user, title="P", steps=[])
        self.tracking_csv = _write_csv(TRACKING_ROWS, "tracking.csv")
        self.events_csv = _write_csv(WORKER_EVENTS, "events.csv")

    def _steps(self, analyzer_type, config=None):
        return [
            {"id": "v", "block_type": "input.video",
             "config": {"video_id": str(self.video.pk)}},
            {"id": "d", "block_type": "detect.objects", "config": {"label": "bee"},
             "inputs": {"video": "v"}},
            {"id": "m", "block_type": "track.mot", "config": {"tracker": "beetrack"},
             "inputs": {"detections": "d"}},
            {"id": "r", "block_type": "reference.layout",
             "config": {"source": "device_layout"}, "inputs": {"video": "v"}},
            {"id": "a", "block_type": analyzer_type, "config": config or {},
             "inputs": {"tracks": "m", "rois": "r"}},
        ]

    def _execute(self, analyzer_type, config=None, with_worker_events=False):
        steps = self._steps(analyzer_type, config)
        result = {
            "tracking_csv_path": self.tracking_csv,
            "summary_stats": {"video_fps": 25.0},
        }
        if with_worker_events:
            result["events_csv_path"] = self.events_csv
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = {
            "v": {"artifact": "video", "video_id": self.video.pk},
            "m": {"artifact": "tracks", "result": result},
        }
        inputs = {"tracks": run.context["m"], "rois": None}
        return executors.LOCAL_EXECUTORS[analyzer_type](
            steps[4], run, run.context, inputs, 4)


class EventsBlockTests(PrimitiveBlockTestCase):
    def test_two_stays_in_the_tube_produce_two_enters_and_two_exits(self):
        out = self._execute("analyze.events")

        self.assertEqual(out["table_kind"], "events")
        self.assertEqual(out["enter_count"], 2)
        self.assertEqual(out["exit_count"], 2)
        self.assertEqual(out["subjects"], 1)

    def test_seconds_use_the_clips_real_frame_rate(self):
        out = self._execute("analyze.events")

        first = min(out["rows"], key=lambda r: r["frame"])
        self.assertEqual(first["frame"], 0)
        self.assertEqual(out["fps"], 25.0)
        self.assertEqual(out["fps_source"], "analysis")

    def test_gap_tolerance_merges_a_dropped_frame_rather_than_doubling(self):
        wide = self._execute("analyze.events", {"gap_frames": 100})

        # With 100 frames of slack the two stays are one: a single enter/exit.
        self.assertEqual(wide["enter_count"], 1)

    def test_worker_nest_events_join_the_same_table_but_stay_labelled(self):
        out = self._execute("analyze.events", with_worker_events=True)

        kinds = {r["target_kind"] for r in out["rows"]}
        self.assertEqual(kinds, {"reference", "nest"})
        self.assertEqual({r["source"] for r in out["rows"]}, {"derived", "gpu"})

    def test_no_reference_says_so_instead_of_reporting_zero(self):
        steps = self._steps("analyze.events")
        del steps[3]  # drop the reference.layout node
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = {"v": {"artifact": "video", "video_id": self.video.pk},
                       "m": {"artifact": "tracks", "result": {
                           "tracking_csv_path": self.tracking_csv,
                           "summary_stats": {"video_fps": 25.0}}}}
        out = executors.LOCAL_EXECUTORS["analyze.events"](
            steps[3], run, run.context, {"tracks": run.context["m"]}, 3)

        self.assertIn("note", out)


class InteractionsBlockTests(PrimitiveBlockTestCase):
    def test_each_stay_is_one_interaction_with_the_reference(self):
        out = self._execute("analyze.interactions")

        self.assertEqual(out["table_kind"], "interactions")
        self.assertEqual(out["organism_reference"], 2)
        self.assertEqual(out["organism_organism"], 0)

    def test_duration_is_measured_at_the_clips_real_rate(self):
        out = self._execute("analyze.interactions")

        # 20 frames then 10 frames, at 25 fps.
        self.assertEqual(sorted(r["duration_sec"] for r in out["rows"]), [0.4, 0.8])

    def test_the_visits_filter_keeps_only_reference_episodes(self):
        out = self._execute("analyze.interactions",
                            {"interaction_type": "organism_reference"})

        self.assertTrue(all(r["b_kind"] == "reference" for r in out["rows"]))

    def test_the_insect_to_insect_filter_empties_a_reference_only_clip(self):
        out = self._execute("analyze.interactions",
                            {"interaction_type": "organism_organism"})

        self.assertEqual(out["rows"], [])
        self.assertEqual(out["interaction_count"], 0)


class PrimitivesAgreeTests(PrimitiveBlockTestCase):
    """The point of the collapse: the tables cannot disagree about a clip."""

    def test_events_and_interactions_describe_the_same_episodes(self):
        events = self._execute("analyze.events")
        interactions = self._execute("analyze.interactions")

        self.assertEqual(events["enter_count"], interactions["organism_reference"])
        enters = sorted(r["frame"] for r in events["rows"] if r["action"] == "enter")
        self.assertEqual(enters, sorted(r["start_frame"] for r in interactions["rows"]))

    def test_the_retired_visitation_block_agrees_with_the_primitives(self):
        visitation = self._execute("analyze.visitation")
        interactions = self._execute("analyze.interactions")

        self.assertEqual(visitation["total_visits"], interactions["organism_reference"])
        self.assertEqual(visitation["total_dwell_sec"],
                         round(sum(r["duration_sec"] for r in interactions["rows"]), 2))


# Two bees flying together well away from any reference, plus the big-flower
# case the worker's centroid rule could never see.
PAIR_ROWS = (
    [{"frame": f, "track_id": 5, "cx": 0.70, "cy": 0.50} for f in range(0, 30)]
    + [{"frame": f, "track_id": 6, "cx": 0.72, "cy": 0.50} for f in range(0, 30)]
)


class ProximityBlockTests(PrimitiveBlockTestCase):
    def setUp(self):
        super().setUp()
        self.pair_csv = _write_csv(PAIR_ROWS, "pair_tracking.csv")

    def _execute_with(self, csv_path, config=None):
        steps = self._steps("analyze.interactions", config)
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = {
            "v": {"artifact": "video", "video_id": self.video.pk},
            "m": {"artifact": "tracks", "result": {
                "tracking_csv_path": csv_path,
                "summary_stats": {"video_fps": 25.0,
                                  "frame_width": 1920, "frame_height": 1080},
            }},
        }
        return executors.LOCAL_EXECUTORS["analyze.interactions"](
            steps[4], run, run.context, {"tracks": run.context["m"]}, 4)

    def test_two_bees_flying_together_are_recorded(self):
        out = self._execute_with(self.pair_csv)

        self.assertEqual(out["organism_organism"], 1)
        self.assertEqual(out["rows"][0]["relation"], "proximity")

    def test_a_tighter_radius_stops_seeing_them(self):
        # They are 0.02 of frame width apart.
        out = self._execute_with(self.pair_csv, {"proximity_percent": 1})

        self.assertEqual(out["organism_organism"], 0)

    def test_the_radius_is_read_as_a_percent_not_a_fraction(self):
        wide = self._execute_with(self.pair_csv, {"proximity_percent": 5})
        self.assertEqual(wide["organism_organism"], 1)

    def test_junk_in_the_radius_field_falls_back_to_the_default(self):
        out = self._execute_with(self.pair_csv, {"proximity_percent": "wide"})

        self.assertEqual(out["organism_organism"], 1)


class ContainmentBeatsCentroidDistanceTests(PrimitiveBlockTestCase):
    """The bug this replaces: a bee on a big flower was never recorded.

    The worker matches an insect to a reference by centroid-to-centroid
    distance under a flat 50 px, discarding the reference's size. On a 1920-wide
    frame a bee resting on the edge of a 400 px flower sits ~200 px from its
    centre — visibly inside the box in the annotated video, and absent from the
    interactions table. Containment does not care how big the flower is.
    """

    def test_a_bee_at_the_edge_of_a_large_reference_still_counts(self):
        # The device layout tube spans 0.1-0.3, so its centre is (0.2, 0.2).
        # This track sits at (0.29, 0.29): inside, but far from the centre —
        # 0.09 of frame width, ~173 px on a 1920 frame, well past a 50 px rule.
        edge_rows = [{"frame": f, "track_id": 3, "cx": 0.29, "cy": 0.29}
                     for f in range(20)]
        csv_path = _write_csv(edge_rows, "edge_tracking.csv")

        steps = self._steps("analyze.interactions")
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = {
            "v": {"artifact": "video", "video_id": self.video.pk},
            "m": {"artifact": "tracks", "result": {
                "tracking_csv_path": csv_path,
                "summary_stats": {"video_fps": 25.0,
                                  "frame_width": 1920, "frame_height": 1080},
            }},
        }
        out = executors.LOCAL_EXECUTORS["analyze.interactions"](
            steps[4], run, run.context, {"tracks": run.context["m"]}, 4)

        self.assertEqual(out["organism_reference"], 1)
        self.assertEqual(out["rows"][0]["relation"], "inside")
        self.assertEqual(out["rows"][0]["duration_sec"], 0.8)
