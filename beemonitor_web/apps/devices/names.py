"""Fun, unique names for new units: an English word and a bee genus.

Every card is flashed from the same golden image, so every unit used to enroll
as "beemonitor" and a page of devices was a column of identical names. A new
unit is now named from its card's random id ("Amber Osmia"), which reads on a
device list, a map pin and a hostname, and is unique among the owner's units.
"""
from __future__ import annotations

import hashlib
import random
import re

WORDS = [
    "Amber", "Ash", "Autumn", "Bramble", "Breezy", "Bright", "Brisk", "Bronze",
    "Busy", "Clover", "Copper", "Coral", "Dandelion", "Dawn", "Dewy", "Drowsy",
    "Dusky", "Ember", "Fennel", "Fern", "Fuzzy", "Gentle", "Ginger", "Golden",
    "Hazel", "Heather", "Honey", "Humble", "Indigo", "Jolly", "Juniper", "Lavender",
    "Lilac", "Linden", "Lucky", "Maple", "Marigold", "Meadow", "Mellow", "Misty",
    "Mossy", "Nimble", "Oak", "Pebble", "Peppy", "Plucky", "Poppy", "Quiet",
    "Rosy", "Russet", "Rustic", "Saffron", "Sage", "Sandy", "Sorrel", "Sparrow",
    "Speckled", "Spry", "Sunny", "Swift", "Thistle", "Thyme", "Tidy", "Velvet",
    "Violet", "Willow", "Windy", "Wren", "Zesty", "Zippy",
]

GENERA = [
    "Agapostemon", "Andrena", "Anthidium", "Anthophora", "Apis", "Augochlora",
    "Bombus", "Calliopsis", "Centris", "Ceratina", "Chelostoma", "Coelioxys",
    "Colletes", "Dasypoda", "Dianthidium", "Epeolus", "Eucera", "Euglossa",
    "Habropoda", "Halictus", "Heriades", "Hoplitis", "Hylaeus", "Lasioglossum",
    "Macropis", "Megachile", "Melipona", "Melissodes", "Nomada", "Nomia",
    "Osmia", "Panurgus", "Peponapis", "Perdita", "Sphecodes", "Stelis",
    "Tetragonula", "Trachusa", "Trigona", "Xylocopa",
]

# Hostnames every clone of the golden image shares: replaced by a fun name.
# A hostname someone chose (Raspberry Pi Imager lets you set one) is kept.
GENERIC_HOSTNAMES = {"", "beemonitor", "raspberrypi", "localhost", "pi"}


def is_generic(hostname: str) -> bool:
    return (hostname or "").strip().lower() in GENERIC_HOSTNAMES


def hostname_for(name: str) -> str:
    """A name as a valid hostname: "Amber Osmia" -> "amber-osmia"."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return slug[:63] or "beemonitor"


def candidates(seed: str):
    """Names in a fixed order for this seed: the same card always gets the same
    first choice, and the next ones only matter on a clash."""
    rng = random.Random(hashlib.sha256((seed or "").encode()).hexdigest())
    for _ in range(25):
        yield f"{rng.choice(WORDS)} {rng.choice(GENERA)}"
    n = 2
    base = f"{rng.choice(WORDS)} {rng.choice(GENERA)}"
    while True:
        yield f"{base} {n}"
        n += 1


def unique_name(owner, seed: str) -> str:
    """The first name for ``seed`` that none of ``owner``'s units already has."""
    from .models import Device

    taken = {n.lower() for n in Device.objects.filter(owner=owner).values_list("name", flat=True)}
    for name in candidates(seed):
        if name.lower() not in taken:
            return name
    raise AssertionError("unreachable: candidates() never ends")
