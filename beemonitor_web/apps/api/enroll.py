"""Zero-touch device enrollment.

A field unit flashed from a pre-baked image carries only a per-user *enrollment
token* (no device key). On first boot it POSTs that token + its hardware id here;
we create (or re-bind) a Device for the token's owner and hand back a fresh
device key. No copy-paste, no per-device image.

  POST /api/v1/devices/enroll   (JSON, no session)
      token   : bmk_enroll_*   — the user's enrollment token
      hw_id   : str            — the Pi's serial; required, shown for information
      card_id : str            — the SD card's own random id (newer cards)
      hostname, tz, tz_offset_min, lat, lon : optional metadata
  -> { device_key, device_id, name, hostname }

The CARD is the unit. A card that sends ``card_id`` is matched on (owner,
card_id): one Pi can check many cards without merging them, and each card keeps
its own key. Matching on the Pi serial alone (older cards, no card_id) made
every card booted in the same Pi the same Device and invalidated the previous
card's key on each enrollment.

A new unit is named from its card ("Amber Osmia") unless the card's hostname
was set by hand; ``hostname`` in the reply is that name as a hostname, for the
card to adopt.
"""

from __future__ import annotations

import hashlib
import logging

from django.utils import timezone
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.devices import names
from apps.devices.models import Device, EnrollmentToken

logger = logging.getLogger(__name__)


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class DeviceEnrollView(APIView):
    """Self-enroll a device using a user enrollment token (no device key yet)."""

    authentication_classes: list = []   # the enrollment token IS the auth
    permission_classes: list = []
    throttle_classes: list = []

    def post(self, request):
        data = request.data if isinstance(request.data, dict) else {}
        raw = (data.get("token") or "").strip()
        # Allow the token via Authorization: Bearer too.
        if not raw:
            auth = request.META.get("HTTP_AUTHORIZATION", "")
            if auth.startswith("Bearer "):
                raw = auth[7:].strip()
        hw_id = (data.get("hw_id") or "").strip()[:64]
        if not raw or not hw_id:
            return Response({"detail": "token and hw_id are required."}, status=400)

        key_hash = hashlib.sha256(raw.encode()).hexdigest()
        tok = EnrollmentToken.objects.filter(key_hash=key_hash, revoked=False).first()
        if tok is None:
            return Response({"detail": "Invalid or revoked enrollment token."}, status=401)

        hostname = (data.get("hostname") or "").strip()[:100]
        card_id = (data.get("card_id") or "").strip()[:64]
        if card_id:
            existing = Device.objects.filter(owner=tok.user, card_id=card_id).first()
        else:
            # A card from before card ids: the old behaviour, keyed on the Pi.
            existing = Device.objects.filter(owner=tok.user, hw_id=hw_id, card_id="").first()
        if existing is not None:
            device = existing
            raw_key = device.rotate_key()
            created = False
        else:
            name = (names.unique_name(tok.user, card_id or hw_id)
                    if names.is_generic(hostname) else hostname)
            device, raw_key = Device.create_with_key(owner=tok.user, name=name)
            device.hw_id = hw_id
            device.card_id = card_id
            device.save(update_fields=["hw_id", "card_id"])
            created = True
        # The Pi serial is the box the card is in now; it can change.
        if device.hw_id != hw_id:
            device.hw_id = hw_id
            device.save(update_fields=["hw_id"])

        # Opportunistically capture reported metadata (timezone for the charts).
        updates = []
        tz_name = (data.get("tz") or "")[:64]
        if tz_name and tz_name != device.tz_name:
            device.tz_name = tz_name
            updates.append("tz_name")
        tz_off = data.get("tz_offset_min")
        if isinstance(tz_off, (int, float)):
            device.tz_offset_min = int(tz_off)
            updates.append("tz_offset_min")
        lat, lon = _as_float(data.get("lat")), _as_float(data.get("lon"))
        if lat is not None and lon is not None:
            device.lat, device.lon = lat, lon
            updates += ["lat", "lon"]
        if updates:
            device.save(update_fields=updates)

        tok.last_used_at = timezone.now()
        tok.save(update_fields=["last_used_at"])
        logger.info("enroll: user=%s hw=%s card=%s device=%s %s",
                    tok.user_id, hw_id, card_id or "-", device.id,
                    "created" if created else "rebound")
        return Response(
            {"device_key": raw_key, "device_id": device.id, "name": device.name,
             "hostname": names.hostname_for(device.name)},
            status=201 if created else 200,
        )
