"""Data source credentials, encrypted at rest with the platform's Fernet key.

Every way a source is created (web form, API) encrypts with ``encrypt``; every
reader uses ``decrypt``. Sources the API created before it encrypted hold plain
JSON, which ``decrypt`` still reads.
"""

import json
import os

from cryptography.fernet import Fernet, InvalidToken
from django.conf import settings


def fernet() -> Fernet:
    """SOC2 CC6.1: Encryption of sensitive data at rest."""
    key = getattr(settings, "BEEMONITOR_CREDENTIAL_KEY", "") or os.environ.get("BEEMONITOR_CREDENTIAL_KEY", "")
    if not key:
        raise ValueError("BEEMONITOR_CREDENTIAL_KEY is not set.")
    return Fernet(key.encode() if isinstance(key, str) else key)


def encrypt(credentials: dict) -> bytes:
    return fernet().encrypt(json.dumps(credentials).encode())


def decrypt(source) -> dict:
    raw = bytes(source.config_encrypted or b"")
    if not raw:
        raise ValueError("No credentials stored for this source.")
    try:
        return json.loads(fernet().decrypt(raw).decode())
    except InvalidToken:
        # Legacy: stored as plain JSON by the API.
        return json.loads(raw.decode("utf-8"))
