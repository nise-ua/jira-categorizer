"""Import Jira Excel / CSV exports for offline learning and triage."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from jira_categorizer.data.ingest import load_jira_export


SUPPORTED_SUFFIXES = {".xlsx", ".xls", ".csv", ".json", ".jsonl"}


def load_spreadsheet(path: str | Path) -> pd.DataFrame:
    """Load a Jira export spreadsheet into the normalized ticket schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Export not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".csv", ".json", ".jsonl"}:
        return load_jira_export(path)
    if suffix not in {".xlsx", ".xls"}:
        raise ValueError(f"Unsupported spreadsheet type: {suffix}")

    # Jira Cloud/Server Excel exports are usually the first sheet
    raw = pd.read_excel(path, sheet_name=0, dtype=str)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        raw.to_csv(tmp_path, index=False)
        return load_jira_export(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def save_upload(content: bytes, dest_dir: str | Path, filename: str) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(filename).name
    if Path(safe).suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type. Use one of: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )
    path = dest_dir / safe
    path.write_bytes(content)
    return path


def import_to_cache(path: str | Path, cache) -> dict[str, Any]:
    df = load_spreadsheet(path)
    cache.save(df, source=f"spreadsheet:{Path(path).name}")
    return {"rows": int(len(df)), "source": f"spreadsheet:{Path(path).name}", "columns": list(df.columns)}
