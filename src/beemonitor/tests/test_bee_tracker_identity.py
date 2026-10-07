"""BeeTracker keeps one id per bee when bees crowd a nest entrance."""
import unittest

from beemonitor.tracking.mot.bee_tracker import BeeTracker, Track


def box(cx, cy, size=50, conf=0.8):
    h = size / 2
    return (cx - h, cy - h, cx + h, cy + h, conf, "yolo", "bee")


def tracker():
    Track.reset_id_counter()
    return BeeTracker(fps=25.0, frame_height=1080, max_age_seconds=2.0,
                      min_hits_seconds=0.0, max_resurrection_seconds=1.0,
                      resurrection_search_multiplier=2.5, bee_size=50)


class BeeTrackerIdentityTests(unittest.TestCase):
    def test_two_close_bees_both_keep_their_ids(self):
        t = tracker()
        for f in range(20):  # 40 px apart: under the 1.2 x bee-size "duplicate" radius
            out = t.update([box(500, 500), box(540, 500)], f)
        self.assertEqual(sorted(r["track_id"] for r in out), [1, 2])

    def test_lost_bee_keeps_its_id_when_another_walks_past(self):
        t = tracker()
        for f in range(20):  # bee A waits at the nest; bee B is far away
            t.update([box(500, 500), box(900, 500)], f)
        for f in range(20, 40):  # A unseen; B walks over A's spot and on
            t.update([box(900 - (f - 19) * 20, 500)], f)
        for f in range(40, 45):  # A seen again where it was; B still walking
            out = t.update([box(500, 500), box(900 - (f - 19) * 20, 500)], f)
        ids = {abs(r["cx"] - 500) < 10: r["track_id"] for r in out}
        self.assertEqual(ids, {True: 1, False: 2})

    def test_lost_track_does_not_coast_off_at_flight_speed(self):
        t = tracker()
        for f in range(10):  # flying right at 30 px/frame
            t.update([box(300 + 30 * f, 500)], f)
        for f in range(10, 40):  # then missed for 1.2 s
            t.update([], f)
        cx = t.tracks[0].centroid[0]
        self.assertLess(cx - 570, 150)  # undamped it would be ~900 px further on

    def test_ids_have_no_gaps_when_duplicate_boxes_are_dropped(self):
        t = tracker()
        seen = set()
        for f in range(30):
            # Every few frames the detector puts a second box on bee A: a new
            # track, removed as a duplicate. It must not use up an id.
            dets = [box(300, 300), box(700, 300)]
            if f % 3 == 0:
                dets.append(box(302, 301))
            seen.update(r["track_id"] for r in t.update(dets, f))
        self.assertEqual(sorted(seen), list(range(1, len(seen) + 1)))
        # And none was handed out to a track nobody ever saw.
        self.assertEqual(Track._next_confirmed_id - 1, len(seen))


if __name__ == "__main__":
    unittest.main()
