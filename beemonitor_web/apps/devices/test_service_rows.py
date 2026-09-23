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
            self.assertFalse(row.get("note"))


def _unit(active, sub, result="success", load="loaded"):
    return {"load": load, "active": active, "sub": sub, "result": result}


class AllServicesTests(SimpleTestCase):
    """Newer firmware reports every unit; each gets a row with a plain state."""

    def _rows(self, units, **metrics):
        return {r["label"]: r for r in _service_rows({"services": units, **metrics})}

    def test_every_reported_unit_gets_a_row(self):
        rows = self._rows({
            "beemonitor-recorder.service": _unit("active", "running"),
            "beemonitor-telemetry.service": _unit("active", "running"),
            "beemonitor-camera-detect.service": _unit("active", "exited"),
            "beemonitor-calibrate.timer": _unit("active", "waiting"),
            "beemonitor-calibrate.service": _unit("inactive", "dead"),
            "beemonitor-update.service": _unit("inactive", "dead"),
            "beemonitor-tailscale.service": _unit("inactive", "dead", load="not-found"),
        })
        self.assertEqual(rows["Recorder"]["state"], "running")
        self.assertEqual(rows["Camera detect"]["state"], "done")
        self.assertEqual(rows["Motion calibration"]["state"], "scheduled")
        self.assertEqual(rows["Updater"]["level"], "idle")
        self.assertEqual(rows["Remote access"]["state"], "not installed")
        self.assertNotIn("Uploader", rows)  # not reported -> no guess

    def test_a_stopped_core_service_is_a_fault(self):
        row = self._rows({"beemonitor-uploader.service": _unit("inactive", "dead")})["Uploader"]
        self.assertEqual((row["state"], row["level"]), ("stopped", "fail"))

    def test_failed_units_and_failed_last_runs_are_flagged(self):
        rows = self._rows({
            "beemonitor-enroll.service": _unit("failed", "failed", "exit-code"),
            "beemonitor-update.service": _unit("inactive", "dead", "exit-code"),
            "beemonitor-calibrate.timer": _unit("active", "waiting"),
            "beemonitor-calibrate.service": _unit("failed", "failed", "exit-code"),
        })
        self.assertEqual(rows["Enrollment"]["level"], "fail")
        self.assertEqual(rows["Updater"]["state"], "last run failed")
        self.assertEqual(rows["Motion calibration"]["level"], "fail")

    def test_recorder_restarts_and_cellular_traffic_still_warn(self):
        rows = self._rows({
            "beemonitor-recorder.service": _unit("active", "running"),
            "cellular.service": _unit("active", "running"),
        }, recorder_restarts=2, recorder_error="boom",
            cellular_active=True, active_transport="cellular")
        self.assertEqual(rows["Recorder"]["level"], "warn")
        self.assertIn("restarted 2", rows["Recorder"]["note"])
        self.assertEqual(rows["Cellular"]["note"], "carrying traffic")
        self.assertEqual(rows["Cellular"]["level"], "warn")
