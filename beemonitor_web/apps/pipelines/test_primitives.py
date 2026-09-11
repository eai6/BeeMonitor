"""Events and interactions are two views of one pass — and must agree.

The whole point of the collapse is that a visit, an interaction with a
reference, and a pair of enter/exit events stopped being three computations
that could disagree about the same clip. These tests hold them to that.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import ops, primitives

# One tube on the left, one on the right, in normalised coords.
REFS = [
    {"id": "tube_1", "label": "Tube 1", "box": (0.0, 0.0, 0.2, 0.2), "points": None},
    {"id": "tube_2", "label": "Tube 2", "box": (0.8, 0.8, 1.0, 1.0), "points": None},
]


def tracks(*rows):
    """rows of (tid, frame, x, y) as the tidy frame ops produces."""
    return pd.DataFrame(list(rows), columns=["tid", "frame", "x", "y"])


class EpisodeTests(SimpleTestCase):
    def test_a_continuous_stay_is_one_episode(self):
        tidy = tracks(*[(1, f, 0.1, 0.1) for f in range(10)])

        episodes = ops.compute_episodes(tidy, REFS)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["reference"], "tube_1")
        self.assertEqual((episodes[0]["start_frame"], episodes[0]["end_frame"]), (0, 9))
        self.assertEqual(episodes[0]["frames"], 10)

    def test_a_long_gap_splits_the_stay_in_two(self):
        tidy = tracks(*[(1, f, 0.1, 0.1) for f in list(range(5)) + list(range(50, 55))])

        episodes = ops.compute_episodes(tidy, REFS, gap_frames=15)

        self.assertEqual(len(episodes), 2)

    def test_moving_between_references_ends_the_episode(self):
        tidy = tracks(*([(1, f, 0.1, 0.1) for f in range(5)]
                        + [(1, f, 0.9, 0.9) for f in range(5, 10)]))

        episodes = ops.compute_episodes(tidy, REFS)

        self.assertEqual([e["reference"] for e in episodes], ["tube_1", "tube_2"])

    def test_frames_outside_every_reference_are_not_an_episode(self):
        tidy = tracks(*[(1, f, 0.5, 0.5) for f in range(10)])

        self.assertEqual(ops.compute_episodes(tidy, REFS), [])

    def test_two_tracks_in_one_reference_are_two_episodes(self):
        tidy = tracks(*([(1, f, 0.1, 0.1) for f in range(5)]
                        + [(2, f, 0.11, 0.11) for f in range(5)]))

        episodes = ops.compute_episodes(tidy, REFS)

        self.assertEqual(len(episodes), 2)
        self.assertEqual({e["track"] for e in episodes}, {1, 2})


class ProjectionTests(SimpleTestCase):
    def setUp(self):
        tidy = tracks(*([(1, f, 0.1, 0.1) for f in range(0, 25)]
                        + [(1, f, 0.9, 0.9) for f in range(25, 50)]))
        self.episodes = ops.compute_episodes(tidy, REFS)

    def test_every_episode_yields_exactly_one_enter_and_one_exit(self):
        events = primitives.events_from_episodes(self.episodes, fps=25.0)

        self.assertEqual(len(events), 2 * len(self.episodes))
        self.assertEqual(sum(1 for e in events if e["action"] == "enter"),
                         len(self.episodes))
        self.assertEqual(sum(1 for e in events if e["action"] == "exit"),
                         len(self.episodes))

    def test_the_two_tables_describe_the_same_spans(self):
        events = primitives.events_from_episodes(self.episodes, fps=25.0)
        interactions = primitives.interactions_from_episodes(self.episodes, fps=25.0)

        enters = sorted(e["frame"] for e in events if e["action"] == "enter")
        exits = sorted(e["frame"] for e in events if e["action"] == "exit")
        self.assertEqual(enters, sorted(i["start_frame"] for i in interactions))
        self.assertEqual(exits, sorted(i["end_frame"] for i in interactions))

    def test_visit_counts_are_just_the_reference_interactions(self):
        # The old analyzer and the new primitive must not disagree.
        tidy = tracks(*([(1, f, 0.1, 0.1) for f in range(0, 25)]
                        + [(1, f, 0.9, 0.9) for f in range(25, 50)]))
        visitation = ops.compute_visitation(tidy, REFS, fps=25.0)
        interactions = primitives.interactions_from_episodes(
            ops.compute_episodes(tidy, REFS), fps=25.0)
        summary = primitives.summarize_interactions(interactions)

        self.assertEqual(summary["organism_reference"], visitation["total_visits"])

    def test_dwell_time_counts_frames_held_not_the_span(self):
        # Frames 0-4 and 20-24 inside, nothing between: 10 frames at 25 fps.
        # Spanning start to end would claim 1.0s; only 0.4s was observed.
        tidy = tracks(*[(1, f, 0.1, 0.1) for f in list(range(5)) + list(range(20, 25))])
        episodes = ops.compute_episodes(tidy, REFS, gap_frames=30)  # one episode

        interactions = primitives.interactions_from_episodes(episodes, fps=25.0)

        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]["duration_sec"], 0.4)

    def test_seconds_are_none_rather_than_wrong_without_a_rate(self):
        events = primitives.events_from_episodes(self.episodes, fps=0)

        self.assertTrue(all(e["time_sec"] is None for e in events))

    def test_derived_rows_say_they_are_derived(self):
        events = primitives.events_from_episodes(self.episodes, fps=25.0)
        interactions = primitives.interactions_from_episodes(self.episodes, fps=25.0)

        self.assertTrue(all(r["source"] == "derived" for r in events + interactions))


class GpuNormalisationTests(SimpleTestCase):
    def test_worker_entry_exit_rows_become_events(self):
        df = pd.DataFrame([
            {"frame": 100, "action": "Entry", "nest": "nest_1", "track_id": 7},
            {"frame": 220, "action": "Exit", "nest": "nest_1", "track_id": 7},
        ])

        rows = primitives.events_from_gpu(df, fps=25.0)

        self.assertEqual([r["action"] for r in rows], ["enter", "exit"])
        self.assertEqual(rows[0]["time_sec"], 4.0)
        self.assertTrue(all(r["source"] == "gpu" for r in rows))

    def test_worker_targets_stay_distinct_from_roi_references(self):
        # A nest the worker found is not a reference the user drew; a count of
        # ROI visits must not be able to absorb it silently.
        df = pd.DataFrame([{"frame": 1, "action": "Entry", "nest": "nest_1"}])

        self.assertEqual(primitives.events_from_gpu(df, 25.0)[0]["target_kind"], "nest")

    def test_unrecognised_actions_are_dropped_not_guessed(self):
        df = pd.DataFrame([{"frame": 1, "action": "Hover", "nest": "n1"}])

        self.assertEqual(primitives.events_from_gpu(df, 25.0), [])

    def test_organism_to_organism_interactions_keep_both_partners(self):
        df = pd.DataFrame([{
            "interaction_type": "organism-to-organism", "entity1_id": 3,
            "entity2_id": 9, "start_frame": 10, "end_frame": 40,
            "duration_seconds": 1.2,
        }])

        rows = primitives.interactions_from_gpu(df, fps=25.0)

        self.assertEqual((rows[0]["a"], rows[0]["b"]), (3, 9))
        self.assertEqual(rows[0]["b_kind"], "organism")
        self.assertEqual(rows[0]["relation"], "proximity")

    def test_organism_to_reference_interactions_point_at_the_reference(self):
        df = pd.DataFrame([{
            "interaction_type": "organism-to-reference", "organism_track_id": 3,
            "reference_id": "nest_2", "start_frame": 10, "end_frame": 40,
            "duration_seconds": 1.2,
        }])

        rows = primitives.interactions_from_gpu(df, fps=25.0)

        self.assertEqual(rows[0]["b"], "nest_2")
        self.assertEqual(rows[0]["b_kind"], "reference")

    def test_a_missing_duration_is_recovered_from_the_frames(self):
        df = pd.DataFrame([{
            "interaction_type": "organism-to-organism", "entity1_id": 1,
            "entity2_id": 2, "start_frame": 0, "end_frame": 50,
        }])

        self.assertEqual(primitives.interactions_from_gpu(df, fps=25.0)[0]["duration_sec"], 2.0)

    def test_empty_inputs_yield_empty_tables(self):
        self.assertEqual(primitives.events_from_gpu(None, 25.0), [])
        self.assertEqual(primitives.interactions_from_gpu(None, 25.0), [])
        self.assertEqual(primitives.events_from_episodes([], 25.0), [])


class RollupTests(SimpleTestCase):
    def test_event_summary_splits_enters_from_exits_per_target(self):
        rows = primitives.events_from_episodes(
            [{"track": 1, "reference": "tube_1", "reference_label": "Tube 1",
              "start_frame": 0, "end_frame": 10, "frames": 11}], fps=25.0)

        summary = primitives.summarize_events(rows)

        self.assertEqual((summary["enter_count"], summary["exit_count"]), (1, 1))
        self.assertEqual(summary["per_reference"][0]["subjects"], 1)

    def test_interaction_summary_reports_the_visitation_view(self):
        rows = [
            {"a": 1, "a_kind": "organism", "b": "tube_1", "b_kind": "reference",
             "duration_sec": 2.0, "start_frame": 0},
            {"a": 1, "a_kind": "organism", "b": 2, "b_kind": "organism",
             "duration_sec": 0.5, "start_frame": 5},
        ]

        summary = primitives.summarize_interactions(rows)

        self.assertEqual(summary["organism_reference"], 1)
        self.assertEqual(summary["organism_organism"], 1)
        self.assertEqual(summary["total_duration_sec"], 2.5)
        self.assertEqual(summary["per_reference"][0]["id"], "tube_1")


class ProximityTests(SimpleTestCase):
    """Insect-to-insect is the one place a distance threshold IS the model.

    The worker used a flat 50 px, which means a different real distance at
    every resolution — and after normalisation there are no pixels left to
    compare against. These pin the frame-relative replacement.
    """

    def test_two_tracks_side_by_side_are_one_episode(self):
        tidy = tracks(*([(1, f, 0.50, 0.50) for f in range(10)]
                        + [(2, f, 0.52, 0.50) for f in range(10)]))

        episodes = ops.compute_proximity_episodes(tidy, radius=0.05)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["frames"], 10)

    def test_tracks_further_apart_than_the_radius_never_meet(self):
        tidy = tracks(*([(1, f, 0.1, 0.5) for f in range(10)]
                        + [(2, f, 0.9, 0.5) for f in range(10)]))

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius=0.05), [])

    def test_a_lone_track_has_nobody_to_interact_with(self):
        tidy = tracks(*[(1, f, 0.5, 0.5) for f in range(10)])

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius=0.5), [])

    def test_each_unordered_pair_is_counted_once_not_twice(self):
        tidy = tracks((1, 0, 0.50, 0.50), (2, 0, 0.51, 0.50))

        episodes = ops.compute_proximity_episodes(tidy, radius=0.05)

        self.assertEqual(len(episodes), 1)

    def test_three_tracks_together_give_three_pairs(self):
        tidy = tracks((1, 0, 0.50, 0.50), (2, 0, 0.51, 0.50), (3, 0, 0.50, 0.51))

        self.assertEqual(len(ops.compute_proximity_episodes(tidy, radius=0.05)), 3)

    def test_the_radius_is_a_circle_not_an_ellipse(self):
        """Normalisation divides x by width and y by height separately, so
        without the aspect correction a vertical gap counts as much smaller
        than the same real distance horizontally."""
        # 0.04 apart in normalised y on a 16:9 frame is 0.0225 of frame width.
        vertical = tracks((1, 0, 0.5, 0.50), (2, 0, 0.5, 0.54))

        near = ops.compute_proximity_episodes(vertical, radius=0.03, aspect=16 / 9)
        far = ops.compute_proximity_episodes(vertical, radius=0.02, aspect=16 / 9)

        self.assertEqual(len(near), 1)   # 0.0225 < 0.03
        self.assertEqual(far, [])        # 0.0225 > 0.02

    def test_a_separation_splits_the_encounter_in_two(self):
        tidy = tracks(*([(1, f, 0.50, 0.5) for f in range(60)]
                        + [(2, f, 0.51, 0.5) for f in range(5)]
                        + [(2, f, 0.90, 0.5) for f in range(5, 50)]
                        + [(2, f, 0.51, 0.5) for f in range(50, 60)]))

        episodes = ops.compute_proximity_episodes(tidy, radius=0.05, gap_frames=15)

        self.assertEqual(len(episodes), 2)

    def test_the_closest_approach_is_reported(self):
        tidy = tracks((1, 0, 0.50, 0.5), (2, 0, 0.54, 0.5),
                      (1, 1, 0.50, 0.5), (2, 1, 0.51, 0.5))

        episodes = ops.compute_proximity_episodes(tidy, radius=0.05)

        self.assertAlmostEqual(episodes[0]["min_distance"], 0.01, places=6)

    def test_the_rows_are_organism_to_organism_with_a_distance(self):
        tidy = tracks((1, 0, 0.50, 0.5), (2, 0, 0.51, 0.5))
        episodes = ops.compute_proximity_episodes(tidy, radius=0.05)

        rows = primitives.interactions_from_proximity(episodes, fps=25.0)

        self.assertEqual(rows[0]["b_kind"], "organism")
        self.assertEqual(rows[0]["relation"], "proximity")
        self.assertIn("min_distance", rows[0])

    def test_a_zero_radius_measures_nothing_rather_than_everything(self):
        tidy = tracks((1, 0, 0.5, 0.5), (2, 0, 0.5, 0.5))

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius=0), [])


class FrameAspectTests(SimpleTestCase):
    def test_dimensions_from_the_summary_are_used(self):
        self.assertAlmostEqual(ops.frame_aspect({"frame_width": 1920,
                                                 "frame_height": 1080}), 16 / 9)

    def test_the_alternative_key_spellings_are_understood(self):
        self.assertAlmostEqual(ops.frame_aspect({"width": 640, "height": 480}), 4 / 3)

    def test_a_summary_without_dimensions_falls_back_to_widescreen(self):
        self.assertAlmostEqual(ops.frame_aspect({}), 16 / 9)
        self.assertAlmostEqual(ops.frame_aspect(None), 16 / 9)

    def test_a_zero_height_does_not_divide_by_zero(self):
        self.assertAlmostEqual(ops.frame_aspect({"width": 100, "height": 0}), 16 / 9)
