#!/usr/bin/env python3
"""Enrollment sends the card's own id, and the card takes its unit's name.

    python3 hardware/test_enroll.py

Stdlib only, no network or root: the platform and hostnamectl are faked.
Exits non-zero on failure.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import enroll  # noqa: E402


class _Reply:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class EnrollTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.env = d / "uploader.env"
        self.env.write_text("BEEMONITOR_API_BASE=https://example.test\n")
        self.boot = d / "beemonitor.conf"
        self.boot.write_text("BEEMONITOR_ENROLL_TOKEN=bmk_enroll_x\n")
        self.card = d / "card_id"
        self.patches = [
            mock.patch.object(enroll, "ENV_FILE", str(self.env)),
            mock.patch.object(enroll, "CARD_ID_FILE", str(self.card)),
            mock.patch.object(enroll, "BOOT_CONF_CANDIDATES", [str(self.boot)]),
            mock.patch.object(enroll, "_hw_id", return_value="10000000f6da9015"),
            mock.patch.object(enroll, "_configure_wifi"),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _run(self, hostname="beemonitor"):
        sent = {}

        def urlopen(req, timeout=0):
            sent.update(json.loads(req.data.decode()))
            return _Reply({"device_key": "bmk_device_k", "device_id": 7,
                           "name": "Amber Osmia", "hostname": "amber-osmia"})

        with mock.patch.object(enroll.urllib.request, "urlopen", urlopen), \
                mock.patch.object(enroll.socket, "gethostname", return_value=hostname), \
                mock.patch.object(enroll, "_adopt_hostname") as adopt:
            self.assertEqual(enroll.main(), 0)
        return sent, adopt

    def test_the_card_sends_its_own_id_and_keeps_it(self):
        sent, _ = self._run()

        self.assertTrue(sent["card_id"])
        self.assertEqual(self.card.read_text().strip(), sent["card_id"])
        self.assertEqual(sent["hw_id"], "10000000f6da9015")
        self.assertIn("BEEMONITOR_DEVICE_KEY=bmk_device_k", self.env.read_text())

    def test_the_card_id_survives_a_second_read(self):
        first = enroll._card_id()
        self.assertEqual(enroll._card_id(), first)

    def test_the_card_takes_its_units_name(self):
        _, adopt = self._run()
        adopt.assert_called_once_with("amber-osmia")

    def test_a_hostname_set_by_hand_is_not_replaced(self):
        with mock.patch.object(enroll.socket, "gethostname", return_value="north-meadow"), \
                mock.patch.object(enroll.subprocess, "run") as run:
            enroll._adopt_hostname("amber-osmia")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
