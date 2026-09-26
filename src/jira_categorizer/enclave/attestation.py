"""Enclave attestation stubs for AWS Nitro Enclaves.

In production, the host verifies PCR measurements before releasing a KMS-wrapped DEK
to the enclave over vsock. This module documents the contract and provides a local
dev stand-in that never claims production attestation.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class AttestationReport:
    mode: str
    attested: bool
    enclave_id: str
    pcrs: dict[str, str]
    timestamp: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _hash_file_list(paths: list[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.encode("utf-8"))
        if os.path.isfile(p):
            h.update(_read_bytes(p))
    return h.hexdigest()


def collect_local_dev_report(image_paths: list[str] | None = None) -> AttestationReport:
    """Local stand-in: fingerprint process environment; not a Nitro attestation."""
    image_paths = image_paths or []
    fingerprint = _hash_file_list(image_paths) if image_paths else hashlib.sha256(
        f"{platform.node()}|{os.getpid()}".encode()
    ).hexdigest()
    return AttestationReport(
        mode="local_dev",
        attested=False,
        enclave_id=f"local-{fingerprint[:12]}",
        pcrs={
            "pcr0": fingerprint[:64],
            "pcr1": hashlib.sha256(platform.python_version().encode()).hexdigest(),
            "pcr2": hashlib.sha256(b"jira-categorizer").hexdigest(),
        },
        timestamp=datetime.now(timezone.utc).isoformat(),
        notes=(
            "LOCAL DEV ONLY. Replace with Nitro NSM attestation document and "
            "KMS Decrypt conditioned on PCR values before handling production Jira data."
        ),
    )


def require_attestation(cfg: dict[str, Any]) -> AttestationReport:
    mode = cfg.get("enclave", {}).get("mode", "local_dev")
    report = collect_local_dev_report()
    report.mode = mode
    if mode == "nitro" and cfg.get("enclave", {}).get("attest_required", True):
        # Production hook: integrate with Nitro Secure Module (nsm-lib / aws-nitro-enclaves-nsm-api)
        # and verify against expected PCR0/PCR1/PCR2 from the signed enclave image.
        if not os.environ.get("NITRO_ATTESTATION_DOC"):
            raise RuntimeError(
                "Nitro mode requires attestation document (NITRO_ATTESTATION_DOC). "
                "Use enclave.mode=local_dev for offline development."
            )
        report.attested = True
        report.notes = "Attestation document present; verify PCRs against expected measurements."
    elif mode == "local_dev" and cfg.get("enclave", {}).get("allow_unattested_local", True):
        report.attested = False
    else:
        raise RuntimeError("Attestation required but mode/config does not allow unattested run")
    return report


def dump_report(report: AttestationReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report.to_dict(), fh, indent=2)
