"""The event rule (memory/60): one test per worked case.

For each track, its first and last DETECTED row: starts in R and is later
elsewhere -> exit; ends in R and was elsewhere before -> enter; starts and ends
in R without leaving -> stay; neither end in a reference -> nothing. No
duration, hand-off or edge filters; edges are flagged.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import ops, primitives

R3 = {"id": "reference_3", "label": "Reference 3", "box": (0.10, 0.10, 0.20, 0.20), "points": None}
R5 = {"id": "reference_5", "label": "Reference 5", "box": (0.70, 0.10, 0.80, 0.20), "points": None}
R1 = {"id": "reference_1", "label": "Reference 1", "box": (0.30, 0.40, 0.35, 0.45), "points": None}
REFS = [R3, R5, R1]

IN_R3, IN_R5, IN_R1, AWAY = (0.15, 0.15), (0.75, 0.15), (0.32, 0.42), (0.5, 0.8)


def track(*points, tid=1, start=10, detected=None):
    """One track through ``points``, one per frame from ``start``."""
    rows = []
    for i, (x, y) in enumerate(points):
        rows.append({"tid": tid, "frame": start + i, "x": x, "y": y,
                     "detected": 1 if detected is None else detected[i]})
    return pd.DataFrame(rows)


def events(tidy, refs=REFS, **kw):
    kw.setdefault("last_frame", 1000)
    return [(e["action"], e["reference"], e["frame"])
            for e in ops.compute_events(tidy, refs, 25.0, **kw)]


class WorkedCases(SimpleTestCase):
    def test_starts_in_r3_and_flies_off_is_an_exit(self):
        self.assertEqual(events(track(IN_R3, IN_R3, AWAY, AWAY)),
                         [("exit", "reference_3", 10)])

    def test_comes_in_and_ends_in_r3_is_an_entry(self):
        self.assertEqual(events(track(AWAY, AWAY, IN_R3, IN_R3)),
                         [("enter", "reference_3", 13)])

    def test_out_and_back_into_r3_is_an_exit_and_an_entry(self):
        self.assertEqual(events(track(IN_R3, AWAY, AWAY, IN_R3)),
                         [("exit", "reference_3", 10), ("enter", "reference_3", 13)])

    def test_from_r3_into_r5_is_an_exit_and_an_entry(self):
        self.assertEqual(events(track(IN_R3, AWAY, IN_R5)),
                         [("exit", "reference_3", 10), ("enter", "reference_5", 12)])

    def test_never_leaving_r3_is_a_stay(self):
        rows = ops.compute_events(track(IN_R3, IN_R3, IN_R3, IN_R3), REFS, 25.0, last_frame=1000)

        self.assertEqual([(e["action"], e["reference"], e["frame"], e["end_frame"]) for e in rows],
                         [("stay", "reference_3", 10, 13)])
        saved = primitives.events_from_computed(rows, 25.0)
        self.assertEqual(saved[0]["duration_sec"], 0.12)

    def test_passing_over_references_is_not_an_event(self):
        self.assertEqual(events(track(AWAY, IN_R1, IN_R5, IN_R3, AWAY)), [])

    def test_a_predicted_tail_drifting_off_r3_is_still_an_entry_at_the_last_detection(self):
        tidy = track(AWAY, IN_R3, IN_R3, AWAY, AWAY, detected=[1, 1, 1, 0, 0])

        self.assertEqual(events(tidy), [("enter", "reference_3", 12)])

    def test_a_fast_exit_counts_however_short_the_time_in_the_tube(self):
        # One detected frame on the tube: no minimum duration any more.
        self.assertEqual(events(track(IN_R3, AWAY)), [("exit", "reference_3", 10)])

    def test_a_tracker_handoff_is_not_filtered(self):
        # Track 1 ends on the tube, track 2 starts there a frame later. The
        # rule reports what each track did; the old hand-off filter is gone.
        tidy = pd.concat([track(AWAY, IN_R3, tid=1, start=10),
                          track(IN_R3, AWAY, tid=2, start=12)])

        self.assertEqual(events(tidy), [("enter", "reference_3", 11), ("exit", "reference_3", 12)])

    def test_no_references_means_no_events(self):
        self.assertEqual(events(track(IN_R3, AWAY), refs=[]), [])

    def test_a_table_without_the_detected_column_uses_every_row(self):
        tidy = track(AWAY, IN_R3).drop(columns=["detected"])

        self.assertFalse(ops.has_detected(tidy))
        self.assertEqual(events(tidy), [("enter", "reference_3", 11)])


class EdgeTests(SimpleTestCase):
    def test_an_exit_on_the_first_frame_analysed_is_flagged(self):
        rows = ops.compute_events(track(IN_R3, AWAY, start=0), REFS, 25.0,
                                  first_frame=0, last_frame=100)

        self.assertIs(rows[0]["at_edge"], True)

    def test_an_entry_on_the_last_frame_analysed_is_flagged(self):
        rows = ops.compute_events(track(AWAY, IN_R3, start=99), REFS, 25.0,
                                  first_frame=0, last_frame=100)

        self.assertIs(rows[0]["at_edge"], True)

    def test_an_event_inside_the_clip_is_not_at_the_edge(self):
        rows = ops.compute_events(track(IN_R3, AWAY, start=5), REFS, 25.0,
                                  first_frame=0, last_frame=100)

        self.assertIs(rows[0]["at_edge"], False)

    def test_a_chosen_window_start_is_an_edge(self):
        rows = ops.compute_events(track(IN_R3, AWAY, start=250), REFS, 25.0,
                                  first_frame=250, last_frame=500)

        self.assertIs(rows[0]["at_edge"], True)

    def test_an_unknown_clip_length_leaves_the_end_edge_unknown(self):
        rows = ops.compute_events(track(AWAY, IN_R3, start=40), REFS, 25.0,
                                  first_frame=0, last_frame=None)

        self.assertIsNone(rows[0]["at_edge"])
        self.assertEqual(primitives.export_rows("events",
                                                primitives.events_from_computed(rows, 25.0),
                                                25.0)[0]["at_edge"], "")


    def test_an_exit_is_always_known_even_without_the_clip_length(self):
        rows = ops.compute_events(track(IN_R3, AWAY, start=5), REFS, 25.0,
                                  first_frame=0, last_frame=None)

        self.assertIs(rows[0]["at_edge"], False)

    def test_a_stay_from_the_first_frame_is_at_the_edge(self):
        rows = ops.compute_events(track(IN_R3, IN_R3, start=0), REFS, 25.0,
                                  first_frame=0, last_frame=None)

        self.assertIs(rows[0]["at_edge"], True)

    def test_a_stay_inside_the_clip_with_unknown_length_is_unknown(self):
        rows = ops.compute_events(track(IN_R3, IN_R3, start=5), REFS, 25.0,
                                  first_frame=0, last_frame=None)

        self.assertIsNone(rows[0]["at_edge"])

    def test_a_stay_to_the_last_frame_is_at_the_edge(self):
        rows = ops.compute_events(track(IN_R3, IN_R3, start=99), REFS, 25.0,
                                  first_frame=0, last_frame=100)

        self.assertIs(rows[0]["at_edge"], True)

    def test_a_stay_inside_a_known_clip_is_not_at_the_edge(self):
        rows = ops.compute_events(track(IN_R3, IN_R3, start=5), REFS, 25.0,
                                  first_frame=0, last_frame=100)

        self.assertIs(rows[0]["at_edge"], False)


class PaddingAndOverlapTests(SimpleTestCase):
    def test_padding_in_body_lengths_reaches_a_centre_just_outside(self):
        tidy = track((0.205, 0.15), AWAY)        # 0.005 frame widths right of R3
        tidy["bl"] = 0.01

        self.assertEqual(events(tidy), [])
        self.assertEqual(events(tidy, padding_bl=1.0), [("exit", "reference_3", 10)])

    def test_overlapping_references_are_named(self):
        a = {"id": "a", "label": "a", "box": (0.0, 0.0, 0.5, 0.5), "points": None}
        b = {"id": "b", "label": "b", "box": (0.4, 0.4, 0.9, 0.9), "points": None}
        c = {"id": "c", "label": "c", "box": (0.95, 0.95, 1.0, 1.0), "points": None}

        self.assertEqual(ops.overlapping_references([a, b, c]), [("a", "b")])

    def test_touching_is_not_overlapping(self):
        a = {"id": "a", "label": "a", "box": (0.0, 0.0, 0.5, 0.5), "points": None}
        b = {"id": "b", "label": "b", "box": (0.5, 0.0, 1.0, 0.5), "points": None}

        self.assertEqual(ops.overlapping_references([a, b]), [])

    def test_a_point_in_two_references_belongs_to_the_nearest_centre(self):
        a = {"id": "a", "label": "a", "box": (0.0, 0.0, 0.5, 0.5), "points": None}
        b = {"id": "b", "label": "b", "box": (0.4, 0.4, 0.9, 0.9), "points": None}

        self.assertEqual(ops.reference_at(0.42, 0.42, [a, b])["id"], "a")
        self.assertEqual(ops.reference_at(0.48, 0.48, [a, b])["id"], "b")

    def test_a_point_in_an_overlap_counts_once(self):
        a = {"id": "a", "label": "a", "box": (0.0, 0.0, 0.5, 0.5), "points": None}
        b = {"id": "b", "label": "b", "box": (0.4, 0.4, 0.9, 0.9), "points": None}

        self.assertEqual(events(track((0.42, 0.42), (0.95, 0.2)), refs=[a, b]), [("exit", "a", 10)])


class HotelIsNotAReferenceTests(SimpleTestCase):
    def test_the_hotel_roi_is_dropped(self):
        refs = ops.roi_references({"hotel_roi": [0, 0, 1, 1],
                                   "nest_layout": [{"id": 3, "box": [0.1, 0.1, 0.2, 0.2]}]})

        self.assertEqual([r["id"] for r in refs], ["reference_3"])

    def test_a_hotel_with_no_tubes_gives_no_references(self):
        self.assertEqual(ops.roi_references({"hotel_roi": [0, 0, 1, 1]}), [])
