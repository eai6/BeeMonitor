"""The Events and Interactions blocks, end to end through the executor.

Both are computed from the tracking table against the references the PIPELINE
defines (memory/60) — never the worker's own nests or events.
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
    def test_coming_out_and_going_back_in_are_the_events(self):
        # Track 1 starts on the tube (came out), leaves, and is last seen on it
        # at frame 69 (went in). Track 2 never touches it.
        out = self._execute("analyze.events")

        self.assertEqual(out["table_kind"], "events")
        self.assertEqual([(r["action"], r["frame"], r["track_id"]) for r in out["rows"]],
                         [("exit", 0, 1), ("enter", 69, 1)])
        self.assertEqual(out["subjects"], 1)
        self.assertEqual(out["reference_source"], "layout")

    def test_seconds_use_the_clips_real_frame_rate(self):
        out = self._execute("analyze.events")

        self.assertEqual(out["fps"], 25.0)
        self.assertEqual(out["fps_source"], "analysis")

    def test_the_hotel_roi_is_not_a_reference(self):
        # The device's ROI (0-0.6) contains every point track 1 visits; only
        # the tube is a reference, so only the tube has events.
        out = self._execute("analyze.events")

        self.assertEqual({r["reference"] for r in out["rows"]}, {"reference_1"})

    def test_the_workers_own_events_file_is_ignored(self):
        out = self._execute("analyze.events", with_worker_events=True)

        self.assertEqual({r["reference"] for r in out["rows"]}, {"reference_1"})
        self.assertFalse(any("source" in r for r in out["rows"]))

    def test_a_tracking_table_without_detected_rows_says_so(self):
        out = self._execute("analyze.events")

        self.assertIs(out["tracking_has_detected"], False)
        self.assertIn("predicted positions", out["note"])

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

        self.assertEqual(out["rows"], [])
        self.assertEqual(out["reference_source"], "none")
        self.assertIn("No reference in this pipeline", out["note"])


class InteractionsBlockTests(PrimitiveBlockTestCase):
    def test_each_stay_is_one_interaction_with_the_reference(self):
        out = self._execute("analyze.interactions")

        self.assertEqual(out["table_kind"], "interactions")
        self.assertEqual(out["organism_reference"], 2)
        self.assertEqual(out["organism_organism"], 0)

    def test_by_default_every_contact_is_kept_with_its_duration(self):
        # 20 frames then 10 frames, at 25 fps; no minimum unless the node sets one.
        out = self._execute("analyze.interactions")

        self.assertEqual(sorted(r["duration_sec"] for r in out["rows"]), [0.4, 0.8])

    def test_a_minimum_length_drops_the_shorter_contacts(self):
        out = self._execute("analyze.interactions", {"min_seconds": 0.5})

        self.assertEqual([r["duration_sec"] for r in out["rows"]], [0.8])
        self.assertEqual(out["organism_reference"], 1)

    def test_the_visits_filter_keeps_only_reference_episodes(self):
        out = self._execute("analyze.interactions",
                            {"interaction_type": "organism_reference"})

        self.assertTrue(out["rows"])
        self.assertTrue(all(r["relation"] == "in_reference" for r in out["rows"]))

    def test_the_insect_to_insect_filter_empties_a_reference_only_clip(self):
        out = self._execute("analyze.interactions",
                            {"interaction_type": "organism_organism"})

        self.assertEqual(out["rows"], [])
        self.assertEqual(out["interaction_count"], 0)


class PrimitivesAgreeTests(PrimitiveBlockTestCase):
    """The tables cannot disagree about a clip."""

    def test_an_exit_starts_a_stay_and_an_entry_ends_one(self):
        events = self._execute("analyze.events")
        interactions = self._execute("analyze.interactions")

        starts = {r["start_frame"] for r in interactions["rows"]}
        ends = {r["end_frame"] for r in interactions["rows"]}
        for r in events["rows"]:
            self.assertIn(r["frame"], starts if r["action"] == "exit" else ends)

    def test_the_retired_visitation_block_runs_as_interactions(self):
        visitation = self._execute("analyze.visitation")
        interactions = self._execute("analyze.interactions")

        self.assertEqual(visitation["table_kind"], "interactions")
        self.assertEqual(visitation["rows"], interactions["rows"])

    def test_the_retired_foraging_trips_block_runs_as_events(self):
        trips = self._execute("analyze.foraging_trips")
        events = self._execute("analyze.events")

        self.assertEqual(trips["table_kind"], "events")
        self.assertEqual(trips["rows"], events["rows"])


# Two bees flying together well away from any reference, in pixels on a
# 1920x1080 frame: 38.4 px apart, each box 30 px — 1.28 body lengths apart.
def _pair_row(frame, tid, cx):
    return {"frame": frame, "track_id": tid, "cx": cx, "cy": 540.0,
            "x1": cx - 15, "y1": 525.0, "x2": cx + 15, "y2": 555.0, "detected": 1}


PAIR_ROWS = ([_pair_row(f, 5, 1344.0) for f in range(0, 30)]
             + [_pair_row(f, 6, 1382.4) for f in range(0, 30)])


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
        out = self._execute_with(self.pair_csv)      # default radius: 2 body lengths

        self.assertEqual(out["organism_organism"], 1)
        row = out["rows"][0]
        self.assertEqual((row["relation"], row["track_id"], row["with"]), ("near", 5, 6))
        self.assertAlmostEqual(row["min_distance_bl"], 1.28, places=2)

    def test_a_tighter_radius_stops_seeing_them(self):
        out = self._execute_with(self.pair_csv, {"proximity_bl": 1})

        self.assertEqual(out["organism_organism"], 0)

    def test_junk_in_the_radius_field_falls_back_to_the_default(self):
        out = self._execute_with(self.pair_csv, {"proximity_bl": "wide"})

        self.assertEqual(out["organism_organism"], 1)

    def test_no_reference_still_measures_insect_to_insect(self):
        steps = self._steps("analyze.interactions")
        del steps[3]
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = {"v": {"artifact": "video", "video_id": self.video.pk},
                       "m": {"artifact": "tracks", "result": {
                           "tracking_csv_path": self.pair_csv,
                           "summary_stats": {"video_fps": 25.0, "frame_width": 1920,
                                             "frame_height": 1080}}}}
        out = executors.LOCAL_EXECUTORS["analyze.interactions"](
            steps[3], run, run.context, {"tracks": run.context["m"]}, 3)

        self.assertEqual((out["organism_organism"], out["organism_reference"]), (1, 0))


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
        self.assertEqual(out["rows"][0]["relation"], "in_reference")
        self.assertEqual(out["rows"][0]["duration_sec"], 0.8)


class NoWorkerFallbackTests(PrimitiveBlockTestCase):
    """References come only from the pipeline (memory/60). The worker's own
    nest boxes (``summary_stats["nest_bboxes"]``) and events file are never a
    stand-in for a reference the pipeline didn't define."""

    def _execute_without_reference_node(self, block_type):
        steps = self._steps(block_type)
        del steps[3]                      # no reference.layout node at all
        result = {
            "tracking_csv_path": self.tracking_csv,
            "events_csv_path": self.events_csv,
            "summary_stats": {"video_fps": 25.0,
                              "nest_bboxes": {"1": [192, 108, 576, 324]}},
        }
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = {"v": {"artifact": "video", "video_id": self.video.pk},
                       "m": {"artifact": "tracks", "result": result}}
        return executors.LOCAL_EXECUTORS[block_type](
            steps[3], run, run.context, {"tracks": run.context["m"]}, 3)

    def setUp(self):
        super().setUp()
        self.video.width, self.video.height = 1920, 1080
        self.video.save(update_fields=["width", "height"])

    def test_no_events_from_the_workers_nests_or_events_file(self):
        out = self._execute_without_reference_node("analyze.events")

        self.assertEqual(out["reference_source"], "none")
        self.assertEqual(out["reference_count"], 0)
        self.assertEqual(out["rows"], [])

    def test_no_reference_interactions_from_the_workers_nests(self):
        out = self._execute_without_reference_node("analyze.interactions")

        self.assertEqual(out["organism_reference"], 0)
        self.assertIn("No reference in this pipeline", out["note"])

    def test_a_graph_reference_is_reported_as_the_layout(self):
        out = self._execute("analyze.interactions")

        self.assertEqual(out["reference_source"], "layout")
