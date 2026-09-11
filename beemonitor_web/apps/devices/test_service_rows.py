"""A green dot beside "Cellular" is not "this device is on cellular".

It only ever meant cellular.service is up, which is the NORMAL state on a
device sitting on WiFi with a modem fitted: the unit stays running so the link
is there the moment WiFi is not. Read as a transport indicator it says the
opposite of the truth, and it is the modem that costs money.
"""

from django.test import SimpleTestCase

from apps.devices.views import _service_rows


def _cell(rows):
    return next(r for r in rows if r["label"] == "Cellular")


class ServiceRowTests(SimpleTestCase):
    def test_the_unit_running_on_wifi_reads_as_standby(self):
        row = _cell(_service_rows({"cellular_active": True, "active_transport": "wifi"}))

        self.assertTrue(row["ok"])
        self.assertEqual(row["note"], "standby")
        self.assertFalse(row["warn"])

    def test_traffic_actually_on_cellular_is_flagged(self):
        """This is the state worth noticing — it is the one that costs money."""
        row = _cell(_service_rows({"cellular_active": True,
                                   "active_transport": "cellular"}))

        self.assertEqual(row["note"], "carrying traffic")
        self.assertTrue(row["warn"])

    def test_the_unit_stopped_reads_as_off(self):
        row = _cell(_service_rows({"cellular_active": False,
                                   "active_transport": "wifi"}))

        self.assertFalse(row["ok"])
        self.assertEqual(row["note"], "off")

    def test_no_route_reported_yet_does_not_guess(self):
        """Before the first beat, "standby" would be a claim we cannot make."""
        row = _cell(_service_rows({"cellular_active": True}))

        self.assertEqual(row["note"], "ready")
        self.assertFalse(row["warn"])

    def test_an_unrecognised_interface_still_counts_as_not_cellular(self):
        """A USB adapter routes as wlan1/eth0 — anything but cellular is fine."""
        row = _cell(_service_rows({"cellular_active": True, "active_transport": "eth0"}))

        self.assertEqual(row["note"], "standby")

    def test_the_other_two_rows_carry_no_note(self):
        rows = _service_rows({"recorder_active": True, "uploader_active": True})

        for label in ("Recorder", "Uploader"):
            row = next(r for r in rows if r["label"] == label)
            self.assertTrue(row["ok"])
            self.assertNotIn("note", row)
