"""Counting static objects from sampled frames.

Nest tubes and flowers don't move, so two things follow: you don't need every
frame, and you can't count detections. The same tube appears in every sampled
frame, so summing detections multiplies it by the frame count — these tests pin
the dedup that turns detections back into objects.
"""

from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from apps.pipelines import executors, ops

from .test_executors import ExecutorTestCase


def frame(n, boxes):
    return {"frame_number": n, "boxes": boxes}


def box(x, y, w=20, h=20, cls="nest_tube", conf=0.9):
    return {"x": x, "y": y, "w": w, "h": h, "class": cls, "confidence": conf}


def static_scene(n_frames=5, n_objects=3, jitter=2):
    """The same objects in every frame, with a little detector jitter."""
    return [
        frame(i * 30, [box(100 * j + (i % jitter), 50 + (i % jitter))
                       for j in range(n_objects)])
        for i in range(n_frames)
    ]


class DistinctObjectTests(SimpleTestCase):
    def test_three_tubes_in_five_frames_are_three_objects(self):
        """The headline case: 15 detections, 3 physical objects."""
        frames = static_scene(n_frames=5, n_objects=3)
        raw = sum(len(f["boxes"]) for f in frames)

        out = ops.count_distinct_objects(frames, "nest_tube")

        self.assertEqual(raw, 15)                      # naive count
        self.assertEqual(out["distinct_objects"], 3)   # actual objects
        self.assertEqual(out["frames_sampled"], 5)

    def test_jitter_does_not_split_one_object_into_many(self):
        """Exact-coordinate dedup would report one object per frame. Jitter here
        is +/-2 px on a 40 px box — a few percent, which is what a detector
        actually does frame to frame."""
        frames = [frame(i * 30, [box(100 + (i % 3), 50 + (i % 3), w=40, h=40)])
                  for i in range(10)]

        self.assertEqual(ops.count_distinct_objects(frames, "nest_tube")["distinct_objects"], 1)

    def test_far_apart_boxes_stay_separate(self):
        frames = [frame(0, [box(0, 0), box(500, 500)])]

        self.assertEqual(ops.count_distinct_objects(frames, "nest_tube")["distinct_objects"], 2)

    def test_object_missed_in_some_frames_still_counts(self):
        """Occlusion is why dedup beats a per-frame count."""
        frames = [frame(0, [box(0, 0), box(200, 0)]),
                  frame(30, [box(0, 0)]),
                  frame(60, [box(0, 0)])]

        out = ops.count_distinct_objects(frames, "nest_tube")

        self.assertEqual(out["distinct_objects"], 2)
        hits = sorted(r["seen_in_frames"] for r in out["rows"])
        self.assertEqual(hits, [1, 3])

    def test_label_filter_applies(self):
        frames = [frame(0, [box(0, 0, cls="nest_tube"), box(300, 0, cls="bee")])]

        self.assertEqual(ops.count_distinct_objects(frames, "nest_tube")["distinct_objects"], 1)
        self.assertEqual(ops.count_distinct_objects(frames, "bee")["distinct_objects"], 1)
        self.assertEqual(ops.count_distinct_objects(frames, "")["distinct_objects"], 2)

    def test_rows_carry_positions_for_export(self):
        out = ops.count_distinct_objects(static_scene(n_frames=3, n_objects=2), "nest_tube")

        self.assertEqual(len(out["rows"]), 2)
        for r in out["rows"]:
            self.assertLess(r["x1"], r["x2"])
            self.assertLess(r["y1"], r["y2"])

    def test_empty_input(self):
        self.assertEqual(ops.count_distinct_objects([], "nest_tube")["distinct_objects"], 0)
        self.assertEqual(ops.count_distinct_objects(None, "x")["distinct_objects"], 0)

    def test_malformed_boxes_are_skipped_not_fatal(self):
        frames = [frame(0, [{"x": 1}, box(0, 0), {"x": "a", "y": "b", "w": 1, "h": 1}])]

        self.assertEqual(ops.count_distinct_objects(frames, "")["distinct_objects"], 1)


class ModalCountTests(SimpleTestCase):
    def test_modal_count_of_a_static_scene(self):
        out = ops.modal_frame_count(static_scene(n_frames=5, n_objects=3), "nest_tube")

        self.assertEqual(out["modal_count"], 3)
        self.assertEqual(out["frames_agreeing"], 5)

    def test_one_bad_frame_does_not_move_the_mode(self):
        frames = static_scene(n_frames=5, n_objects=3)
        frames[2]["boxes"] = frames[2]["boxes"][:1]   # two objects missed

        out = ops.modal_frame_count(frames, "nest_tube")

        self.assertEqual(out["modal_count"], 3)
        self.assertEqual(out["frames_agreeing"], 4)

    def test_frames_with_no_detections_count_as_zero(self):
        frames = [frame(0, []), frame(30, []), frame(60, [box(0, 0)])]

        self.assertEqual(ops.modal_frame_count(frames, "nest_tube")["modal_count"], 0)

    def test_empty_input(self):
        self.assertEqual(ops.modal_frame_count([], "x")["modal_count"], 0)


class SampledJobRoutingTests(ExecutorTestCase):
    def _sampled_steps(self, **extra):
        cfg = {"label": "nest_tube", "analyse": "sampled", **extra}
        return [
            {"id": "v", "block_type": "input.video",
             "config": {"video_id": str(self.video.pk)}},
            {"id": "d", "block_type": "detect.objects", "config": cfg,
             "inputs": {"video": "v"}},
        ]

    def test_sampled_node_submits_the_pre_annotate_task(self):
        steps = self._sampled_steps(sample_interval=60, max_frames=15)
        run = self._run(steps, self._video_context())

        built, err = executors.build_job_config(steps[1], run, run.context, 1)

        self.assertIsNone(err)
        self.assertEqual(built["config"]["task"], "pre_annotate")
        self.assertEqual(built["config"]["classes"], ["nest_tube"])
        self.assertEqual(built["config"]["sample_interval"], 60)
        self.assertEqual(built["config"]["max_frames"], 15)
        self.assertNotIn("run_tracking", built["config"])

    def test_full_frame_node_still_submits_tracking(self):
        run = self._run(self._module_steps(), self._video_context())

        built, _ = executors.build_job_config(
            self._module_steps()[1], run, run.context, 1)

        self.assertNotIn("task", built["config"])
        self.assertTrue(built["config"]["run_tracking"])

    def test_sampling_knobs_are_clamped(self):
        steps = self._sampled_steps(sample_interval=99999, max_frames=0)
        run = self._run(steps, self._video_context())

        built, _ = executors.build_job_config(steps[1], run, run.context, 1)

        self.assertEqual(built["config"]["sample_interval"], 600)
        self.assertEqual(built["config"]["max_frames"], 1)

    def test_sampled_node_requires_a_label(self):
        steps = self._sampled_steps()
        steps[1]["config"]["label"] = ""
        run = self._run(steps, self._video_context())

        built, err = executors.build_job_config(steps[1], run, run.context, 1)

        self.assertIsNone(built)
        self.assertIn("needs a label", err)

    def test_sampled_nodes_do_not_join_the_shared_tracking_job(self):
        """A sampled node runs its own job, so its label must not leak into the
        tracking job's shared class list."""
        steps = self._module_steps()
        steps.append({"id": "n", "block_type": "detect.objects",
                      "config": {"label": "nest_tube", "analyse": "sampled"},
                      "inputs": {"video": "v"}})
        run = self._run(steps, self._video_context())

        built, _ = executors.build_job_config(steps[1], run, run.context, 1)

        self.assertEqual(built["config"]["detect_labels"], ["bee"])


class SampledAnalyzerTests(ExecutorTestCase):
    def _count(self, metric, frames, **cfg):
        step = {"id": "c", "block_type": "analyze.detection_count",
                "config": {"metric": metric, **cfg}, "inputs": {"detections": "d"}}
        # The Detect node must be labelled for what the boxes actually are —
        # the analyzer inherits its label to pick its own rows.
        steps = self._module_steps({"label": "nest_tube", "analyse": "sampled"},
                                   analyzer=step)
        run = self._run(steps, self._video_context())
        idx = next(i for i, s in enumerate(steps) if s["id"] == "c")
        result = {"summary_stats": {"sampled_frames": frames}} if frames is not None else {}
        return executors._exec_analyze_detection_count(
            steps[idx], run, run.context,
            {"detections": {"artifact": "detections", "result": result}}, idx)

    def test_distinct_reads_the_sampled_frames(self):
        out = self._count("distinct", static_scene(n_frames=4, n_objects=3))

        self.assertEqual(out["distinct_objects"], 3)
        self.assertIn("distinct object", out["note"])

    def test_modal_reads_the_sampled_frames(self):
        out = self._count("modal", static_scene(n_frames=4, n_objects=2))

        self.assertEqual(out["modal_count"], 2)

    def test_distinct_without_sampling_explains_itself(self):
        """Rather than silently returning a meaningless number."""
        out = self._count("distinct", None)

        self.assertEqual(out["rows"], [])
        self.assertIn("Sampled frames", out["note"])
