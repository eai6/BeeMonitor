"""Events are going in and coming out, not every tube a passing bee touched.

A bee crossing a 60-tube hotel from the front passed over five tube boxes on
its way to the one it went into, and the table listed 12 events: an enter and
an exit for every box. What happened was one thing — it went into tube 46.
"""

import pandas as pd
from django.test import SimpleTestCase

from apps.pipelines import ops, primitives

FPS = 25


def ref(n, x):
    """A tube box 0.04 wide at height 0.5."""
    return {"id": f"reference_{n}", "label": f"reference {n}",
            "box": (x, 0.48, x + 0.04, 0.52), "points": None}


REFS = [ref(10, 0.10), ref(20, 0.20), ref(50, 0.30), ref(60, 0.40), ref(46, 0.60)]
ON = {r["id"]: r["box"][0] + 0.02 for r in REFS}   # each tube's centre x
OFF = 0.9                                           # nowhere near a tube


def track(tid, points):
    """points: [(frame, x)] at y=0.5 on the tube row, or y=0.1 when x is OFF."""
    return [{"tid": tid, "frame": f, "x": x, "y": 0.5 if x != OFF else 0.1} for f, x in points]


def tidy(*tracks):
    return pd.DataFrame([row for t in tracks for row in t])


def events(df, **kw):
    return ops.compute_nest_events(df, REFS, FPS, gap_frames=15, **kw)


class NestEventTests(SimpleTestCase):
    def test_the_pasted_clip_is_one_entry(self):
        # Passes over 10, 20, 50, 60 for a frame or three, then sits on 46
        # from frame 108 and vanishes at 171 (the clip ends there too).
        path = ([(f, OFF) for f in range(40, 60)]
                + [(61, ON["reference_10"]), (64, ON["reference_20"]), (65, ON["reference_20"]),
                   (73, ON["reference_50"]), (82, ON["reference_60"]), (84, ON["reference_60"])]
                + [(f, 0.5) for f in range(86, 108)]
                + [(f, ON["reference_46"]) for f in range(108, 172)])
        out = events(tidy(track(1, path)))

        self.assertEqual([(e["action"], e["reference"], e["frame"]) for e in out],
                         [("enter", "reference_46", 171)])

    def test_appearing_on_a_tube_is_coming_out(self):
        path = [(f, ON["reference_20"]) for f in range(100, 120)] + [(f, OFF) for f in range(120, 150)]
        out = events(tidy(track(1, path)))

        self.assertEqual([(e["action"], e["reference"], e["frame"]) for e in out],
                         [("exit", "reference_20", 100)])

    def test_tracker_losing_the_bee_is_not_an_entry(self):
        a = track(1, [(f, OFF) for f in range(60, 100)] + [(f, ON["reference_20"]) for f in range(100, 120)])
        b = track(2, [(f, ON["reference_20"]) for f in range(125, 130)] + [(f, OFF) for f in range(130, 160)])
        self.assertEqual(events(tidy(a, b)), [])

    def test_a_brush_shorter_than_the_minimum_is_not_an_entry(self):
        path = [(f, OFF) for f in range(0, 50)] + [(50, ON["reference_20"]), (51, ON["reference_20"])]
        self.assertEqual(events(tidy(track(1, path))), [])

    def test_a_chosen_window_cuts_tracks_and_those_are_not_events(self):
        path = [(f, ON["reference_20"]) for f in range(1000, 1500)]
        self.assertEqual(events(tidy(track(1, path)), start_frame=1000, end_frame=1500), [])
        # The same track in a whole clip: it came out and went back in.
        self.assertEqual([e["action"] for e in events(tidy(track(1, path)))], ["exit", "enter"])

    def test_rows_are_in_the_event_schema(self):
        path = [(f, OFF) for f in range(0, 50)] + [(f, ON["reference_46"]) for f in range(50, 80)]
        rows = primitives.events_from_nest_events(events(tidy(track(7, path))), FPS)
        self.assertEqual(rows[0]["subject"], 7)
        self.assertEqual(rows[0]["action"], "enter")
        self.assertEqual(rows[0]["target"], "reference_46")
        self.assertAlmostEqual(rows[0]["time_sec"], 79 / FPS, places=2)
