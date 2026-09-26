"""Sealing helpers for model/data protection in enclave workflows.

Production: replace local Fernet key with Nitro attestation + KMS unwrap of a DEK.
Local/dev: Fernet key file under artifacts/sealed (never ship to prod).
"""

from __future__ import annotations

from pathlib import Path

from cryptography.fernet import Fernet


def ensure_local_key(key_path: str | Path) -> bytes:
    path = Path(key_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_bytes().strip()
    key = Fernet.generate_key()
    path.write_bytes(key)
    path.chmod(0o600)
    return key


def seal_bytes(payload: bytes, key_path: str | Path) -> bytes:
    key = ensure_local_key(key_path)
    return Fernet(key).encrypt(payload)


def unseal_bytes(token: bytes, key_path: str | Path) -> bytes:
    key = ensure_local_key(key_path)
    return Fernet(key).decrypt(token)
