"""Encrypt and decrypt cloud source credentials using Fernet symmetric encryption."""

import json
import os
from typing import Optional

from cryptography.fernet import Fernet


def get_or_create_key() -> bytes:
    """Get encryption key from env var or generate one.

    In production, BEEMONITOR_CREDENTIAL_KEY must be set.
    """
    key = os.environ.get("BEEMONITOR_CREDENTIAL_KEY")
    if key:
        return key.encode()
    return Fernet.generate_key()


class CredentialManager:
    """Encrypt/decrypt JSON credential blobs for cloud source storage."""

    def __init__(self, key: Optional[bytes] = None):
        self._key = key or get_or_create_key()
        self._fernet = Fernet(self._key)

    def encrypt(self, credentials: dict) -> bytes:
        """Encrypt a credentials dict to bytes."""
        payload = json.dumps(credentials).encode()
        return self._fernet.encrypt(payload)

    def decrypt(self, encrypted: bytes) -> dict:
        """Decrypt bytes back to a credentials dict."""
        payload = self._fernet.decrypt(encrypted)
        return json.loads(payload)

    @property
    def key(self) -> bytes:
        return self._key
