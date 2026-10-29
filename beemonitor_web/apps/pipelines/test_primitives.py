"""The episode pass, the interaction projections, proximity and the rollups
(memory/35, rules from memory/60).

Events have their own file (tests/test_event_rule.py); these pin what the
interactions table is made of and how both tables are summarised and exported.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import ops, primitives

# One tube on the left, one on the right, in normalised coords.
REFS = [
    {"id": "tube_1", "label": "Tube 1", "box": (0.0, 0.0, 0.2, 0.2), "points": None},
    {"id": "tube_2", "label": "Tube 2", "box": (0.8, 0.8, 1.0, 1.0), "points": None},
]


def tracks(*rows, bl=None):
    """rows of (tid, frame, x, y[, detected]) as the tidy frame ops produces.
    ``bl`` gives every row that body length (frame widths)."""
    rows = [r if len(r) == 5 else (*r, 1) for r in rows]
    df = pd.DataFrame(rows, columns=["tid", "frame", "x", "y", "detected"])
    if bl is not None:
        df["bl"] = bl
    return df


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

        self.assertEqual(len(ops.compute_episodes(tidy, REFS, gap_frames=15)), 2)

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

        self.assertEqual({e["track"] for e in episodes}, {1, 2})

    def test_predicted_rows_are_not_time_in_the_reference(self):
        # 5 detected frames in the tube, then 5 the tracker only predicted.
        tidy = tracks(*([(1, f, 0.1, 0.1, 1) for f in range(5)]
                        + [(1, f, 0.1, 0.1, 0) for f in range(5, 10)]))

        episodes = ops.compute_episodes(tidy, REFS)

        self.assertEqual(episodes[0]["frames"], 5)
        self.assertEqual(episodes[0]["end_frame"], 4)

    def test_padding_in_body_lengths_reaches_a_centre_just_outside(self):
        # Centre 0.01 frame widths right of tube_1's edge; body length 0.02.
        tidy = tracks(*[(1, f, 0.21, 0.1) for f in range(5)], bl=0.02)

        self.assertEqual(ops.compute_episodes(tidy, REFS), [])
        self.assertEqual(len(ops.compute_episodes(tidy, REFS, padding_bl=1.0)), 1)

    def test_no_body_length_means_no_padding(self):
        tidy = tracks(*[(1, f, 0.21, 0.1) for f in range(5)])

        self.assertEqual(ops.compute_episodes(tidy, REFS, padding_bl=5.0), [])


class InteractionProjectionTests(SimpleTestCase):
    def test_each_episode_is_one_in_reference_interaction(self):
        tidy = tracks(*([(1, f, 0.1, 0.1) for f in range(0, 25)]
                        + [(1, f, 0.9, 0.9) for f in range(25, 50)]))

        rows = primitives.interactions_from_episodes(ops.compute_episodes(tidy, REFS), fps=25.0)

        self.assertEqual([(r["track_id"], r["with"], r["relation"]) for r in rows],
                         [(1, "tube_1", "in_reference"), (1, "tube_2", "in_reference")])
        self.assertEqual(rows[0]["duration_sec"], 1.0)

    def test_dwell_time_counts_frames_held_not_the_span(self):
        # Frames 0-4 and 20-24 inside, nothing between: 10 frames at 25 fps.
        tidy = tracks(*[(1, f, 0.1, 0.1) for f in list(range(5)) + list(range(20, 25))])
        episodes = ops.compute_episodes(tidy, REFS, gap_frames=30)  # one episode

        rows = primitives.interactions_from_episodes(episodes, fps=25.0)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["duration_sec"], 0.4)

    def test_seconds_are_none_rather_than_wrong_without_a_rate(self):
        tidy = tracks(*[(1, f, 0.1, 0.1) for f in range(5)])

        rows = primitives.interactions_from_episodes(ops.compute_episodes(tidy, REFS), fps=0)

        self.assertIsNone(rows[0]["duration_sec"])

    def test_no_source_column(self):
        tidy = tracks(*[(1, f, 0.1, 0.1) for f in range(5)])
        rows = primitives.interactions_from_episodes(ops.compute_episodes(tidy, REFS), 25.0)

        self.assertNotIn("source", rows[0])


class RollupTests(SimpleTestCase):
    def test_event_summary_splits_exits_entries_and_stays_per_reference(self):
        rows = [
            {"frame": 0, "track_id": 1, "action": "exit", "reference": "tube_1"},
            {"frame": 9, "track_id": 1, "action": "enter", "reference": "tube_2"},
            {"frame": 3, "track_id": 2, "action": "stay", "reference": "tube_1"},
        ]

        summary = primitives.summarize_events(rows)

        self.assertEqual((summary["exit_count"], summary["enter_count"], summary["stay_count"]),
                         (1, 1, 1))
        self.assertEqual(summary["subjects"], 2)
        tube_1 = next(r for r in summary["per_reference"] if r["id"] == "tube_1")
        self.assertEqual((tube_1["exit"], tube_1["stay"], tube_1["subjects"]), (1, 1, 2))

    def test_interaction_summary_reports_time_in_each_reference(self):
        rows = [
            {"track_id": 1, "with": "tube_1", "relation": "in_reference",
             "duration_sec": 2.0, "start_frame": 0},
            {"track_id": 1, "with": 2, "relation": "near",
             "duration_sec": 0.5, "start_frame": 5},
        ]

        summary = primitives.summarize_interactions(rows)

        self.assertEqual(summary["organism_reference"], 1)
        self.assertEqual(summary["organism_organism"], 1)
        self.assertEqual(summary["total_duration_sec"], 2.5)
        self.assertEqual(summary["per_reference"][0]["id"], "tube_1")


class ProximityTests(SimpleTestCase):
    """Insect-to-insect closeness in BODY LENGTHS (memory/60): the long side of
    the insects' own boxes, averaged over the pair."""

    def test_two_tracks_side_by_side_are_one_episode(self):
        tidy = tracks(*([(1, f, 0.50, 0.50) for f in range(10)]
                        + [(2, f, 0.52, 0.50) for f in range(10)]), bl=0.02)

        episodes = ops.compute_proximity_episodes(tidy, radius_bl=2)

        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["frames"], 10)
        self.assertAlmostEqual(episodes[0]["min_distance_bl"], 1.0, places=6)

    def test_tracks_further_apart_than_the_radius_never_meet(self):
        tidy = tracks(*([(1, f, 0.1, 0.5) for f in range(10)]
                        + [(2, f, 0.9, 0.5) for f in range(10)]), bl=0.02)

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius_bl=2), [])

    def test_the_same_gap_is_near_for_big_insects_and_far_for_small_ones(self):
        rows = [(1, 0, 0.50, 0.5), (2, 0, 0.56, 0.5)]   # 0.06 frame widths apart

        self.assertEqual(len(ops.compute_proximity_episodes(tracks(*rows, bl=0.04), radius_bl=2)), 1)
        self.assertEqual(ops.compute_proximity_episodes(tracks(*rows, bl=0.01), radius_bl=2), [])

    def test_the_pair_uses_the_mean_of_both_body_lengths(self):
        tidy = tracks((1, 0, 0.50, 0.5), (2, 0, 0.53, 0.5))
        tidy["bl"] = [0.01, 0.03]                      # mean 0.02 -> 1.5 BL apart

        episodes = ops.compute_proximity_episodes(tidy, radius_bl=2)

        self.assertAlmostEqual(episodes[0]["min_distance_bl"], 1.5, places=6)

    def test_insects_without_a_box_are_not_measured(self):
        tidy = tracks((1, 0, 0.50, 0.5), (2, 0, 0.51, 0.5))   # no bl column

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius_bl=2), [])

    def test_predicted_rows_are_not_measured(self):
        tidy = tracks((1, 0, 0.50, 0.5, 1), (2, 0, 0.51, 0.5, 0), bl=0.02)

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius_bl=2), [])

    def test_a_lone_track_has_nobody_to_interact_with(self):
        tidy = tracks(*[(1, f, 0.5, 0.5) for f in range(10)], bl=0.02)

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius_bl=5), [])

    def test_three_tracks_together_give_three_pairs(self):
        tidy = tracks((1, 0, 0.50, 0.50), (2, 0, 0.51, 0.50), (3, 0, 0.50, 0.51), bl=0.02)

        self.assertEqual(len(ops.compute_proximity_episodes(tidy, radius_bl=2)), 3)

    def test_a_crowded_frame_is_still_measured(self):
        # The old pass skipped any frame with more than 200 tracks.
        rows = [(t, 0, 0.001 * t, 0.5) for t in range(250)]

        episodes = ops.compute_proximity_episodes(tracks(*rows, bl=0.001), radius_bl=1.5)

        self.assertEqual(len(episodes), 249)            # each neighbour pair

    def test_the_radius_is_a_circle_not_an_ellipse(self):
        # 0.04 apart in normalised y on a 16:9 frame is 0.0225 of frame width,
        # 2.25 body lengths of 0.01.
        vertical = tracks((1, 0, 0.5, 0.50), (2, 0, 0.5, 0.54), bl=0.01)

        self.assertEqual(len(ops.compute_proximity_episodes(vertical, radius_bl=3, aspect=16 / 9)), 1)
        self.assertEqual(ops.compute_proximity_episodes(vertical, radius_bl=2, aspect=16 / 9), [])

    def test_a_separation_splits_the_encounter_in_two(self):
        tidy = tracks(*([(1, f, 0.50, 0.5) for f in range(60)]
                        + [(2, f, 0.51, 0.5) for f in range(5)]
                        + [(2, f, 0.90, 0.5) for f in range(5, 50)]
                        + [(2, f, 0.51, 0.5) for f in range(50, 60)]), bl=0.02)

        self.assertEqual(len(ops.compute_proximity_episodes(tidy, radius_bl=2, gap_frames=15)), 2)

    def test_the_rows_are_near_with_a_distance_in_body_lengths(self):
        tidy = tracks((1, 0, 0.50, 0.5), (2, 0, 0.51, 0.5), bl=0.02)

        rows = primitives.interactions_from_proximity(
            ops.compute_proximity_episodes(tidy, radius_bl=2), fps=25.0)

        self.assertEqual(rows[0]["relation"], "near")
        self.assertEqual(rows[0]["with"], 2)
        self.assertEqual(rows[0]["min_distance_bl"], 0.5)

    def test_a_zero_radius_measures_nothing_rather_than_everything(self):
        tidy = tracks((1, 0, 0.5, 0.5), (2, 0, 0.5, 0.5), bl=0.02)

        self.assertEqual(ops.compute_proximity_episodes(tidy, radius_bl=0), [])


class ExportTests(SimpleTestCase):
    def test_events_export_in_time_not_frames(self):
        rows = [{"frame": 50, "track_id": 3, "action": "exit", "reference": "reference_1",
                 "at_edge": False}]

        out = primitives.export_rows("events", rows, fps=25.0, start=None)

        self.assertEqual(list(out[0]), list(primitives.EVENT_FIELDS))
        self.assertEqual(out[0]["clip_sec"], 2.0)
        self.assertEqual(out[0]["timestamp"], "")          # start unknown
        self.assertEqual(out[0]["at_edge"], "false")

    def test_an_unknown_edge_is_blank_not_false(self):
        rows = [{"frame": 1, "track_id": 3, "action": "enter", "reference": "r", "at_edge": None}]

        self.assertEqual(primitives.export_rows("events", rows, 25.0)[0]["at_edge"], "")

    def test_interactions_export_both_ends(self):
        rows = [{"start_frame": 25, "end_frame": 75, "duration_sec": 2.0, "track_id": 1,
                 "with": 2, "relation": "near", "min_distance_bl": 0.8}]

        out = primitives.export_rows("interactions", rows, fps=25.0)

        self.assertEqual(list(out[0]), list(primitives.INTERACTION_FIELDS))
        self.assertEqual((out[0]["start_clip_sec"], out[0]["end_clip_sec"]), (1.0, 3.0))

    def test_rows_saved_under_the_old_rules_export_in_the_new_columns(self):
        old_events = [{"frame": 25, "time_sec": 1.0, "subject": 7, "subject_kind": "organism",
                       "action": "enter", "target": "nest_2", "target_kind": "reference",
                       "source": "derived"}]
        old_inter = [{"start_frame": 0, "end_frame": 25, "duration_sec": 1.0, "a": 7,
                      "a_kind": "organism", "b": "nest_2", "b_kind": "reference",
                      "relation": "inside", "source": "derived"},
                     {"start_frame": 0, "end_frame": 25, "duration_sec": 1.0, "a": 7,
                      "a_kind": "organism", "b": 8, "b_kind": "organism",
                      "relation": "proximity", "min_distance": 0.01, "source": "derived"}]

        ev = primitives.export_rows("events", old_events, 25.0)[0]
        inter = primitives.export_rows("interactions", old_inter, 25.0)

        self.assertEqual((ev["track_id"], ev["reference"], ev["action"]), (7, "reference_2", "enter"))
        self.assertEqual([(r["with"], r["relation"]) for r in inter],
                         [("reference_2", "in_reference"), (8, "near")])
        # A fraction of frame width is not a body length: dropped, not relabelled.
        self.assertEqual(inter[1]["min_distance_bl"], "")


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
