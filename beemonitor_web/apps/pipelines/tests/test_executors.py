"""Executor behaviour for the Detector / MOT / Analyzer modules.

The two things most worth pinning down:

* the reference (ROI) is resolved from the Detector's *config* at read time, and
  the upstream walk sees past the local MOT step to find it;
* the GPU job config no longer carries the ``identify_bees`` flag, whose only
  observable effect was busting the StepResult cache.
"""

import csv
import tempfile
from unittest.mock import patch
from pathlib import Path

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.devices.models import Device
from apps.pipelines import executors
from apps.pipelines.models import Pipeline, PipelineRun
from apps.training.models import CustomModel
from apps.videos.models import Video

User = get_user_model()

TRACKING_ROWS = [
    # frame, track_id, cx, cy — two tracks, one inside the ROI box and one outside.
    {"frame": 0, "track_id": 1, "cx": 0.5, "cy": 0.5},
    {"frame": 1, "track_id": 1, "cx": 0.5, "cy": 0.5},
    {"frame": 1, "track_id": 2, "cx": 0.9, "cy": 0.9},
    {"frame": 2, "track_id": 2, "cx": 0.9, "cy": 0.9},
]

# Raw detector output: same scene as TRACKING_ROWS plus one detection the
# tracker never associated into a track — the difference the raw table exists
# to capture.
DETECTION_ROWS = [
    {"frame": 0, "cx": 0.5, "cy": 0.5, "confidence": 0.9, "taxon": "bee"},
    {"frame": 1, "cx": 0.5, "cy": 0.5, "confidence": 0.9, "taxon": "bee"},
    {"frame": 1, "cx": 0.9, "cy": 0.9, "confidence": 0.8, "taxon": "bee"},
    {"frame": 2, "cx": 0.9, "cy": 0.9, "confidence": 0.8, "taxon": "bee"},
    {"frame": 2, "cx": 0.2, "cy": 0.2, "confidence": 0.5, "taxon": "bee"},
]

INTERACTION_ROWS = [
    {"interaction_type": "organism-to-organism", "organism_track_id": 1,
     "partner_track_id": 2, "reference_id": "", "start_frame": 0,
     "duration_seconds": 1.5},
    {"interaction_type": "organism-to-reference", "organism_track_id": 1,
     "partner_track_id": "", "reference_id": "nest_3", "start_frame": 10,
     "duration_seconds": 4.0},
]


def _write_csv(rows, name):
    path = Path(tempfile.mkdtemp()) / name
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)


class ExecutorTestCase(TestCase):
    """Shared fixture: a user, a device with a saved layout, a video, a run."""

    def setUp(self):
        self.user = User.objects.create_user("alice", password="x")
        self.device = Device.objects.create(
            owner=self.user, name="Danniella", key_hash="h1", prefix="bmk_1",
            roi_override=[0.0, 0.0, 0.6, 0.6],
            nest_layout=[{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}],
        )
        self.video = Video.objects.create(
            user=self.user, device=self.device, title="clip",
            storage_key="alice/clip.mp4", file_size_bytes=1,
            status=Video.Status.READY,
        )
        self.pipeline = Pipeline.objects.create(user=self.user, title="P", steps=[])

    def _run(self, steps, context=None):
        run = PipelineRun.objects.create(pipeline=self.pipeline, user=self.user)
        run.steps = steps
        run.context = context or {}
        return run

    def _layout(self, config):
        """A reference.layout node — saved geometry, not a detection."""
        return {"id": "r", "block_type": "reference.layout", "config": config,
                "inputs": {"video": "v"}}

    def _module_steps(self, detector_config=None, analyzer=None, reference=None):
        steps = [
            {"id": "v", "block_type": "input.video",
             "config": {"video_id": str(self.video.pk)}},
            {"id": "d", "block_type": "detect.objects",
             "config": detector_config or {"label": "bee"},
             "inputs": {"video": "v"}},
            {"id": "m", "block_type": "track.mot", "config": {"tracker": "beetrack"},
             "inputs": {"detections": "d"}},
        ]
        if reference is not None:
            steps.append(self._layout(reference))
        if analyzer:
            steps.append(analyzer)
        return steps

    def _video_context(self):
        return {"v": {"artifact": "video", "video_id": self.video.pk}}


class ResolveReferenceTests(ExecutorTestCase):
    """Saved geometry lives on its own node now — Detect means detect."""

    def _resolve(self, config):
        steps = self._module_steps(reference=config)
        run = self._run(steps, self._video_context())
        idx = next(i for i, st in enumerate(steps) if st["id"] == "r")
        return executors.resolve_reference(steps[idx], run, run.context, idx)

    def test_device_layout_reads_the_saved_roi_and_tubes(self):
        ref = self._resolve({"source": "device_layout"})

        self.assertEqual(ref["hotel_roi"], [0.0, 0.0, 0.6, 0.6])
        self.assertEqual(ref["nest_layout"], [{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])

    def test_drawn_parses_the_regions_json(self):
        ref = self._resolve({"source": "drawn", "regions": '[{"box": [0, 0, 1, 1]}]'})

        self.assertEqual(ref["regions"], [{"box": [0, 0, 1, 1]}])

    def test_drawn_survives_malformed_json(self):
        self.assertEqual(self._resolve({"source": "drawn", "regions": "{oops"})["regions"], [])

    def test_legacy_detector_reference_config_still_resolves(self):
        """Pre-split graphs kept reference_source on the Detect node itself."""
        steps = self._module_steps({"label": "bee", "reference_source": "drawn",
                                    "regions": '[{"box": [0, 0, 1, 1]}]'})
        run = self._run(steps, self._video_context())

        ref = executors.resolve_reference(steps[1], run, run.context, 1)

        self.assertEqual(ref["regions"], [{"box": [0, 0, 1, 1]}])


class FindReferenceTests(ExecutorTestCase):
    def test_walks_through_the_mot_step_to_the_detector(self):
        """detector → mot → analyzer must still resolve the Detector's reference."""
        analyzer = {"id": "g", "block_type": "analyze.visitation", "config": {},
                    "inputs": {"tracks": "m", "rois": "r"}}
        steps = self._module_steps(reference={"source": "device_layout"},
                                   analyzer=analyzer)
        run = self._run(steps, self._video_context())
        idx = next(i for i, st in enumerate(steps) if st["id"] == "g")

        ref = executors.find_reference(steps, idx, run.context, run)

        self.assertEqual(ref["hotel_roi"], [0.0, 0.0, 0.6, 0.6])

    def test_finds_a_legacy_roi_step_output(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "r", "block_type": "roi.draw", "config": {}, "inputs": {"in": "v"}},
            {"id": "t", "block_type": "track.bee", "config": {},
             "inputs": {"video": "v", "rois": "r"}},
            {"id": "f", "block_type": "analyze.foraging_trips", "config": {},
             "inputs": {"tracks": "t"}},
        ]
        context = {"r": {"artifact": "roi", "regions": [[0, 0, 1, 1]], "source": "drawn"}}
        run = self._run(steps, context)

        ref = executors.find_reference(steps, 3, run.context, run)

        self.assertEqual(ref["regions"], [[0, 0, 1, 1]])

    def test_returns_empty_when_there_is_no_reference(self):
        analyzer = {"id": "g", "block_type": "analyze.visitation", "config": {},
                    "inputs": {"tracks": "m"}}
        steps = self._module_steps(analyzer=analyzer)
        run = self._run(steps, self._video_context())

        self.assertEqual(executors.find_reference(steps, 3, run.context, run), {})


class DownstreamWalkTests(ExecutorTestCase):
    def test_only_the_matching_branch_is_visited(self):
        """A sibling branch's analyzer must not configure this Detector's job."""
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "d1", "block_type": "detect.objects", "config": {}, "inputs": {"video": "v"}},
            {"id": "d2", "block_type": "detect.objects", "config": {}, "inputs": {"video": "v"}},
            {"id": "m1", "block_type": "track.mot", "config": {}, "inputs": {"detections": "d1"}},
            {"id": "f", "block_type": "analyze.foraging_trips",
             "config": {"event_confidence": 0.25}, "inputs": {"tracks": "m1"}},
        ]
        d1, d2 = steps[1], steps[2]

        self.assertEqual(executors._pipeline_event_confidence(d1, steps), 0.25)
        self.assertEqual(executors._pipeline_event_confidence(d2, steps), 0.6)

    def test_tracker_comes_from_the_downstream_mot_node(self):
        steps = self._module_steps()
        self.assertEqual(executors._pipeline_tracker(steps[1], steps), "beetrack")

        no_mot = steps[:2]
        self.assertEqual(executors._pipeline_tracker(no_mot[1], no_mot), "beetrack")


class BuildJobConfigTests(ExecutorTestCase):
    def _build(self, steps, index=1):
        run = self._run(steps, self._video_context())
        return executors.build_detect_and_track_config(steps[index], run, run.context, index)

    def test_full_scope_runs_tracking(self):
        built, err = self._build(self._module_steps())
        self.assertIsNone(err)
        self.assertTrue(built["config"]["run_tracking"])

    def test_reference_only_scope_skips_tracking(self):
        steps = self._module_steps({"label": "bee", "run_scope": "reference_only"})
        built, err = self._build(steps)
        self.assertIsNone(err)
        self.assertFalse(built["config"]["run_tracking"])

    def test_legacy_blocks_keep_their_tracking_behaviour(self):
        for block_type, expected in (("track.bee", True), ("detect.bee", True),
                                     ("detect.nest", False)):
            step = {"id": "t", "block_type": block_type, "config": {},
                    "inputs": {"video": "v"}}
            self.assertEqual(executors._run_tracking_for(step), expected, block_type)

    def test_reference_is_forwarded_to_the_job(self):
        built, _ = self._build(self._module_steps())
        self.assertEqual(built["config"]["hotel_roi"], [0.0, 0.0, 0.6, 0.6])
        self.assertEqual(built["config"]["nest_layout"],
                         [{"id": 1, "box": [0.1, 0.1, 0.3, 0.3]}])

    def test_every_detect_label_rides_one_job(self):
        """All Detect nodes share a GPU pass, so every node's config lists every
        label — which makes their cache keys identical and de-dupes the job."""
        steps = self._module_steps()
        steps.append({"id": "n", "block_type": "detect.objects",
                      "config": {"label": "nest"}, "inputs": {"video": "v"}})
        run = self._run(steps, self._video_context())
        nest_idx = next(i for i, st in enumerate(steps) if st["id"] == "n")

        bee, _ = executors.build_detect_and_track_config(steps[1], run, run.context, 1)
        nest, _ = executors.build_detect_and_track_config(
            steps[nest_idx], run, run.context, nest_idx)

        self.assertEqual(bee["config"]["detect_labels"], ["bee", "nest"])
        self.assertEqual(bee["config"], nest["config"])  # same key -> one job

    def test_sam3_family_selects_the_sam3_detector(self):
        steps = self._module_steps({"model_family": "sam3", "text_prompt": " hoverfly "})
        built, _ = self._build(steps)
        self.assertEqual(built["config"]["detector_kind"], "sam3")
        self.assertEqual(built["config"]["text_prompt"], "hoverfly")

    def test_legacy_detector_key_still_selects_sam3(self):
        steps = [
            {"id": "v", "block_type": "input.video", "config": {}},
            {"id": "t", "block_type": "track.bee", "config": {"detector": "sam3"},
             "inputs": {"video": "v"}},
        ]
        built, _ = self._build(steps)
        self.assertEqual(built["config"]["detector_kind"], "sam3")

    def test_event_confidence_comes_from_the_downstream_foraging_node(self):
        analyzer = {"id": "f", "block_type": "analyze.foraging_trips",
                    "config": {"event_confidence": 0.3}, "inputs": {"tracks": "m"}}
        built, _ = self._build(self._module_steps(analyzer=analyzer))
        self.assertEqual(built["config"]["ml_threshold"], 0.3)

    def test_species_node_turns_classification_on_in_the_job(self):
        """The whole chain hangs off this flag: _spawn_gpu_job forwards it, the
        handler passes it to CloudPipeline, and the tracker classifies with it."""
        analyzer = {"id": "sp", "block_type": "identify.species",
                    "config": {"min_confidence": 0.7}, "inputs": {"tracks": "m"}}
        built, err = self._build(self._module_steps(analyzer=analyzer))

        self.assertIsNone(err)
        self.assertTrue(built["config"]["identify_species"])
        self.assertEqual(built["config"]["species_min_confidence"], 0.7)

    def test_species_flag_absent_without_the_node(self):
        built, _ = self._build(self._module_steps())
        self.assertNotIn("identify_species", built["config"])

    def test_adding_the_species_node_changes_the_cache_key(self):
        """Unlike the marker flag this replaced, species classification really
        does change the output — the taxon column differs — so it must re-run
        rather than serve a result computed without it."""
        analyzer = {"id": "sp", "block_type": "identify.species",
                    "config": {}, "inputs": {"tracks": "m"}}
        without, _ = self._build(self._module_steps())
        with_species, _ = self._build(self._module_steps(analyzer=analyzer))

        self.assertNotEqual(without["config"], with_species["config"])

    def test_marker_node_does_not_add_identify_flags(self):
        """Regression: identify_bees was a pure cache-buster.

        Nothing downstream consumed it — analysis.views._spawn_gpu_job builds the
        SageMaker payload key-by-key and dropped it — but engine._gpu_cache_key
        hashes this dict, so its only effect was re-billing a full GPU run.
        """
        analyzer = {"id": "i", "block_type": "identify.marker",
                    "config": {"marker_type": "qr"}, "inputs": {"tracks": "m"}}
        with_marker, _ = self._build(self._module_steps(analyzer=analyzer))
        without, _ = self._build(self._module_steps())

        self.assertNotIn("identify_bees", with_marker["config"])
        self.assertNotIn("marker_method", with_marker["config"])
        self.assertEqual(with_marker["config"], without["config"])

    def test_custom_object_model_is_resolved_by_pk(self):
        model = CustomModel.objects.create(
            user=self.user, name="mine", model_type="bee_tracking",
            storage_key="custom/alice/1/best.pt", is_active=True,
        )
        steps = self._module_steps({"object_model": str(model.pk)})
        built, err = self._build(steps)
        self.assertIsNone(err)
        self.assertEqual(built["config"]["custom_bee_model_path"], model.storage_key)

    def test_another_users_model_is_refused(self):
        mallory = User.objects.create_user("mallory", password="x")
        model = CustomModel.objects.create(
            user=mallory, name="theirs", model_type="bee_tracking",
            storage_key="custom/mallory/1/best.pt", is_active=True,
        )
        steps = self._module_steps({"object_model": str(model.pk)})
        built, err = self._build(steps)
        self.assertIsNone(built)
        self.assertIn("unavailable", err)

    def test_missing_video_is_an_error(self):
        steps = self._module_steps()
        run = self._run(steps, {})  # no resolved video in context
        built, err = executors.build_detect_and_track_config(steps[1], run, {}, 1)
        self.assertIsNone(built)
        self.assertIn("No upstream video", err)


class MotStepTests(ExecutorTestCase):
    def test_relabels_the_detector_result_as_tracks(self):
        result = {"tracking_csv_path": "/tmp/t.csv", "unique_tracks": 7}
        out = executors._exec_mot(
            {"id": "m", "config": {"tracker": "beetrack"}}, None, {},
            {"detections": {"artifact": "detections", "result": result, "job_id": 42}}, 2,
        )
        self.assertEqual(out["artifact"], "tracks")
        self.assertEqual(out["result"], result)  # copied through for the analyzers
        self.assertEqual(out["job_id"], 42)
        self.assertEqual(out["unique_tracks"], 7)
        self.assertEqual(out["tracker"], "beetrack")

    def test_errors_when_the_detector_ran_reference_only(self):
        out = executors._exec_mot(
            {"id": "m", "config": {}}, None, {},
            {"detections": {"artifact": "detections", "result": {"nest_count": 4}}}, 2,
        )
        self.assertIn("error", out)
        self.assertIn("Reference only", out["error"])

    def test_propagates_an_upstream_failure(self):
        out = executors._exec_mot(
            {"id": "m", "config": {}}, None, {},
            {"detections": {"error": "GPU job failed."}}, 2,
        )
        self.assertIn("error", out)


class AnalyzerTests(ExecutorTestCase):
    @staticmethod
    def _idx(steps, step_id):
        """Locate a step by id — a reference node shifts positional indices."""
        return next(i for i, st in enumerate(steps) if st["id"] == step_id)

    def _analyzer_inputs(self, csv_key, path):
        return {"tracks": {"artifact": "tracks", "result": {csv_key: path, "fps": 1}},
                "detections": {"artifact": "tracks", "result": {csv_key: path, "fps": 1}}}

    def test_detection_count_prefers_raw_detections(self):
        """Raw detector output includes detections the tracker discarded."""
        path = _write_csv(DETECTION_ROWS, "detections.csv")
        steps = self._module_steps(
            analyzer={"id": "c", "block_type": "analyze.detection_count",
                      "config": {"metric": "total"}, "inputs": {"detections": "m"}})
        run = self._run(steps, self._video_context())

        out = executors._exec_analyze_detection_count(
            steps[self._idx(steps, "c")], run, run.context,
            self._analyzer_inputs("detections_csv_path", path), self._idx(steps, "c"),
        )

        self.assertEqual(out["detections"], 5)
        self.assertEqual(out["frames_with_detections"], 3)
        # No real track ids in the raw table — don't restate the row count as
        # a track count.
        self.assertNotIn("unique_tracks", out)
        self.assertIn("raw detector output", out["note"])

    def test_detection_count_falls_back_to_the_tracked_table(self):
        """Jobs that predate raw-detection export still work, and say so."""
        path = _write_csv(TRACKING_ROWS, "tracking.csv")
        steps = self._module_steps(
            analyzer={"id": "c", "block_type": "analyze.detection_count",
                      "config": {"metric": "total"}, "inputs": {"detections": "m"}})
        run = self._run(steps, self._video_context())

        out = executors._exec_analyze_detection_count(
            steps[self._idx(steps, "c")], run, run.context,
            self._analyzer_inputs("tracking_csv_path", path), self._idx(steps, "c"),
        )

        self.assertEqual(out["detections"], 4)
        self.assertEqual(out["unique_tracks"], 2)
        self.assertEqual(out["frames_with_detections"], 3)
        self.assertIn("predates raw-detection export", out["note"])

    def test_detection_count_respects_the_reference(self):
        """With a reference box, only detections inside it count."""
        path = _write_csv(TRACKING_ROWS, "tracking.csv")
        steps = self._module_steps(
            reference={"source": "drawn", "regions": '[{"box": [0.0, 0.0, 0.6, 0.6]}]'},
            analyzer={"id": "c", "block_type": "analyze.detection_count",
                      "config": {"metric": "total"},
                      "inputs": {"detections": "m", "rois": "r"}})
        run = self._run(steps, self._video_context())

        out = executors._exec_analyze_detection_count(
            steps[self._idx(steps, "c")], run, run.context,
            self._analyzer_inputs("tracking_csv_path", path), self._idx(steps, "c"),
        )

        self.assertEqual(out["detections"], 2)  # track 2 sits outside the box
        self.assertEqual(out["unique_tracks"], 1)

    def test_detection_count_falls_back_to_the_job_summary(self):
        steps = self._module_steps(
            analyzer={"id": "c", "block_type": "analyze.detection_count",
                      "config": {"metric": "total"}, "inputs": {"detections": "m"}})
        run = self._run(steps, self._video_context())

        out = executors._exec_analyze_detection_count(
            steps[3], run, run.context,
            {"detections": {"result": {"unique_tracks": 3}}}, 3,
        )

        self.assertEqual(out["rows"], [])
        self.assertIn("not available", out["note"])

    def test_interaction_summarises_both_kinds(self):
        path = _write_csv(INTERACTION_ROWS, "interactions.csv")
        step = {"id": "x", "block_type": "analyze.interaction",
                "config": {"interaction_type": "all"}}
        run = self._run(self._module_steps(), self._video_context())

        out = executors._exec_analyze_interaction(
            step, run, run.context,
            self._analyzer_inputs("interactions_csv_path", path), 3,
        )

        self.assertEqual(out["interaction_count"], 2)
        self.assertEqual(out["organism_organism"], 1)
        self.assertEqual(out["organism_reference"], 1)
        self.assertEqual(out["total_duration_sec"], 5.5)

    def test_interaction_filters_by_type(self):
        path = _write_csv(INTERACTION_ROWS, "interactions.csv")
        step = {"id": "x", "block_type": "analyze.interaction",
                "config": {"interaction_type": "organism_reference"}}
        run = self._run(self._module_steps(), self._video_context())

        out = executors._exec_analyze_interaction(
            step, run, run.context,
            self._analyzer_inputs("interactions_csv_path", path), 3,
        )

        self.assertEqual(out["interaction_count"], 1)
        self.assertEqual(out["organism_organism"], 0)

    def test_interaction_falls_back_to_the_job_count(self):
        step = {"id": "x", "block_type": "analyze.interaction", "config": {}}
        run = self._run(self._module_steps(), self._video_context())

        out = executors._exec_analyze_interaction(
            step, run, run.context, {"tracks": {"result": {"interaction_count": 9}}}, 3,
        )

        self.assertEqual(out["interaction_count"], 9)
        self.assertIn("not available", out["note"])

    def test_marker_step_decodes_from_crops(self):
        """No bee_id columns in the tracking CSV, so it falls through to the
        crops the job already stored."""
        run = self._run(self._module_steps(), self._video_context())
        ident = {"identified_tracks": 1, "unique_markers": 1, "source": "crops",
                 "rows": [{"track": 1, "marker": "green", "method": "color",
                           "confidence": 0.8, "votes": 3, "crops_read": 3}]}

        with patch("apps.pipelines.markers.identify_from_crops", return_value=ident):
            out = executors._exec_identify_marker(
                {"id": "i", "config": {"marker_type": "auto"}}, run, run.context,
                {"tracks": {"result": {"crops_csv_path": "u/j/track_crops.csv"}}}, 3,
            )

        self.assertEqual(out["identified_tracks"], 1)
        self.assertEqual(out["rows"][0]["marker"], "green")
        self.assertEqual(out["source"], "crops")

    def test_marker_step_prefers_the_trackers_own_ids(self):
        """When the worker ever decodes markers itself, that wins over crops."""
        rows = [dict(r, bee_id="blue-07", bee_id_method="color",
                     bee_id_confidence=0.9) for r in TRACKING_ROWS]
        path = _write_csv(rows, "tracking.csv")
        run = self._run(self._module_steps(), self._video_context())

        with patch("apps.pipelines.markers.identify_from_crops") as from_crops:
            out = executors._exec_identify_marker(
                {"id": "i", "config": {}}, run, run.context,
                {"tracks": {"result": {"tracking_csv_path": path}}}, 3,
            )

        from_crops.assert_not_called()
        self.assertEqual(out["source"], "tracker")
        self.assertEqual(out["identified_tracks"], 2)

    def test_marker_step_explains_an_unsupported_marker_type(self):
        run = self._run(self._module_steps(), self._video_context())

        with patch("apps.pipelines.markers.identify_from_crops", return_value=None):
            out = executors._exec_identify_marker(
                {"id": "i", "config": {"marker_type": "qr"}}, run, run.context,
                {"tracks": {"result": {}}}, 3,
            )

        self.assertEqual(out["identified_tracks"], 0)
        self.assertIn("No decoder for 'qr'", out["note"])

    def test_marker_step_explains_missing_crops(self):
        run = self._run(self._module_steps(), self._video_context())

        with patch("apps.pipelines.markers.identify_from_crops", return_value=None):
            out = executors._exec_identify_marker(
                {"id": "i", "config": {"marker_type": "auto"}}, run, run.context,
                {"tracks": {"result": {}}}, 3,
            )

        self.assertIn("No per-track crops are stored", out["note"])
