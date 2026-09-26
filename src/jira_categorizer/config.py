"""Configuration loading for the categorizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else project_root() / "config" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # Resolve relative paths against project root
    root = project_root()
    for key, value in list(cfg.get("paths", {}).items()):
        p = Path(value)
        if not p.is_absolute():
            cfg["paths"][key] = str(root / p)
    if "enclave" in cfg and "local_key_path" in cfg["enclave"]:
        p = Path(cfg["enclave"]["local_key_path"])
        if not p.is_absolute():
            cfg["enclave"]["local_key_path"] = str(root / p)
    return cfg
