"""A trip paired on timing alone is not the same claim as an observed one.

aggregate_trips pairs Exit -> Entry per nest on the absolute timeline using
only the most recent Exit. It records exit_track_id and entry_track_id and
never compares them, so a cross-video trip asserts the returning bee is the one
that left. For a hotel with one active tube that is fair; with several it
manufactures trips. The pairing is unchanged — what changes is that the page
can now tell the two apart.
"""

from datetime import datetime, timedelta, timezone as dt_timezone

from django.test import SimpleTestCase

from apps.pipelines import aggregate

T0 = datetime(2026, 6, 1, 10, 0, tzinfo=dt_timezone.utc)


def event(action, nest, seconds, video, track_id):
    return {"action": action, "nest": nest, "time": T0 + timedelta(seconds=seconds),
            "video": video, "video_pk": 1, "track_id": track_id}


class TripIdentityTests(SimpleTestCase):
    def test_the_same_track_leaving_and_returning_is_confirmed(self):
        events = [event("Exit", "n1", 0, "clip-a", "7"),
                  event("Entry", "n1", 60, "clip-a", "7")]

        trips, summary = aggregate.aggregate_trips([], events=events)

        self.assertEqual(len(trips), 1)
        self.assertTrue(trips[0]["same_track"])
        self.assertEqual((summary["confirmed_trips"], summary["inferred_trips"]), (1, 0))

    def test_a_different_track_in_the_same_clip_is_only_inferred(self):
        events = [event("Exit", "n1", 0, "clip-a", "7"),
                  event("Entry", "n1", 60, "clip-a", "9")]

        trips, summary = aggregate.aggregate_trips([], events=events)

        self.assertFalse(trips[0]["same_track"])
        self.assertEqual((summary["confirmed_trips"], summary["inferred_trips"]), (0, 1))

    def test_a_cross_video_trip_can_never_be_confirmed(self):
        # Track ids are unique only within a clip, so a matching id across two
        # clips is a coincidence of numbering, not evidence of identity.
        events = [event("Exit", "n1", 0, "clip-a", "7"),
                  event("Entry", "n1", 600, "clip-b", "7")]

        trips, summary = aggregate.aggregate_trips([], events=events)

        self.assertTrue(trips[0]["is_cross_video"])
        self.assertFalse(trips[0]["same_track"])
        self.assertEqual(summary["inferred_trips"], 1)

    def test_a_blank_track_id_on_both_sides_is_not_a_match(self):
        events = [event("Exit", "n1", 0, "clip-a", ""),
                  event("Entry", "n1", 60, "clip-a", "")]

        trips, _summary = aggregate.aggregate_trips([], events=events)

        self.assertFalse(trips[0]["same_track"])

    def test_the_split_always_adds_up_to_the_total(self):
        events = [event("Exit", "n1", 0, "clip-a", "7"),
                  event("Entry", "n1", 60, "clip-a", "7"),
                  event("Exit", "n2", 0, "clip-a", "3"),
                  event("Entry", "n2", 90, "clip-b", "3")]

        _trips, summary = aggregate.aggregate_trips([], events=events)

        self.assertEqual(summary["confirmed_trips"] + summary["inferred_trips"],
                         summary["total_trips"])

    def test_the_csv_export_carries_the_distinction(self):
        events = [event("Exit", "n1", 0, "clip-a", "7"),
                  event("Entry", "n1", 60, "clip-a", "7")]
        trips, _summary = aggregate.aggregate_trips([], events=events)

        fieldnames, rows = aggregate.trips_csv_rows(trips)

        self.assertIn("same_track", fieldnames)
        self.assertTrue(rows[0]["same_track"])
