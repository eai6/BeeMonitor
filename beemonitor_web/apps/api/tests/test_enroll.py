"""Zero-touch enrollment: the SD card is the unit, and new units get fun names.

Matching on the Pi serial alone made every card booted in the same Pi one
Device, and each enrollment invalidated the previous card's key. That is what
happened checking a stack of new cards on one Pi.
"""
import hashlib

from django.contrib.auth import get_user_model
from django.test import TestCase

from apps.devices import names
from apps.devices.models import Device, EnrollmentToken

User = get_user_model()
URL = "/api/v1/devices/enroll"


class EnrollTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user("en", password="x")
        _, self.token = EnrollmentToken.create_token(self.user)

    def _enroll(self, **data):
        body = {"token": self.token, "hw_id": "10000000f6da9015", "hostname": "beemonitor", **data}
        return self.client.post(URL, body, content_type="application/json")

    def test_cards_checked_on_one_pi_become_separate_units(self):
        a = self._enroll(card_id="card-a").json()
        b = self._enroll(card_id="card-b").json()

        self.assertNotEqual(a["device_id"], b["device_id"])
        self.assertEqual(Device.objects.filter(owner=self.user).count(), 2)
        # The first card's key still works: the second never touched it.
        dev_a = Device.objects.get(pk=a["device_id"])
        self.assertEqual(dev_a.key_hash, hashlib.sha256(a["device_key"].encode()).hexdigest())

    def test_the_same_card_enrolling_again_is_the_same_unit(self):
        first = self._enroll(card_id="card-a")
        again = self._enroll(card_id="card-a", hw_id="10000000aaaa0001")

        self.assertEqual(first.status_code, 201)
        self.assertEqual(again.status_code, 200)
        self.assertEqual(first.json()["device_id"], again.json()["device_id"])
        # The serial follows the Pi the card is in now.
        self.assertEqual(Device.objects.get(pk=again.json()["device_id"]).hw_id, "10000000aaaa0001")

    def test_a_card_without_an_id_keeps_the_old_pi_serial_matching(self):
        a = self._enroll().json()
        b = self._enroll().json()

        self.assertEqual(a["device_id"], b["device_id"])

    def test_a_new_unit_gets_a_fun_name_and_hostname(self):
        r = self._enroll(card_id="card-a").json()

        word, genus = r["name"].split(" ")[:2]
        self.assertIn(word, names.WORDS)
        self.assertIn(genus, names.GENERA)
        self.assertEqual(r["hostname"], names.hostname_for(r["name"]))
        self.assertRegex(r["hostname"], r"^[a-z0-9-]+$")

    def test_names_are_unique_per_owner(self):
        got = {self._enroll(card_id=f"card-{i}").json()["name"] for i in range(30)}

        self.assertEqual(len(got), 30)

    def test_a_clash_moves_to_the_next_name(self):
        first = next(names.candidates("card-x"))
        Device.create_with_key(owner=self.user, name=first)

        r = self._enroll(card_id="card-x").json()

        self.assertNotEqual(r["name"].lower(), first.lower())

    def test_a_hostname_set_by_hand_is_kept(self):
        r = self._enroll(card_id="card-a", hostname="North-Meadow-1").json()

        self.assertEqual(r["name"], "North-Meadow-1")

    def test_a_bad_token_is_refused(self):
        r = self.client.post(URL, {"token": "bmk_enroll_nope", "hw_id": "x", "card_id": "c"},
                             content_type="application/json")
        self.assertEqual(r.status_code, 401)
        self.assertFalse(Device.objects.exists())


class NameTests(TestCase):
    def test_the_same_seed_always_gives_the_same_first_name(self):
        self.assertEqual(next(names.candidates("abc")), next(names.candidates("abc")))

    def test_generic_hostnames(self):
        for h in ("", "beemonitor", "BeeMonitor", "raspberrypi"):
            self.assertTrue(names.is_generic(h), h)
        self.assertFalse(names.is_generic("north-meadow"))
