"""Local ticket cache so UI/JQL fetches are not repeated every request."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from jira_categorizer.data.ingest import load_jira_export


class TicketCache:
    def __init__(self, cache_dir: str | Path, ttl_seconds: int = 300):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self._path = self.cache_dir / "tickets.csv"
        self._meta = self.cache_dir / "meta.json"

    def _meta_read(self) -> dict[str, Any]:
        if not self._meta.exists():
            return {}
        return json.loads(self._meta.read_text(encoding="utf-8"))

    def _meta_write(self, payload: dict[str, Any]) -> None:
        self._meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_fresh(self) -> bool:
        meta = self._meta_read()
        fetched_at = float(meta.get("fetched_at", 0))
        return self._path.exists() and (time.time() - fetched_at) < self.ttl_seconds

    def load(self) -> pd.DataFrame | None:
        if not self._path.exists():
            return None
        return load_jira_export(self._path)

    def save(self, df: pd.DataFrame, *, source: str) -> None:
        df.to_csv(self._path, index=False)
        self._meta_write({"fetched_at": time.time(), "source": source, "rows": int(len(df))})

    def get_or_load_file(self, path: str | Path, *, force: bool = False) -> pd.DataFrame:
        if not force and self.is_fresh():
            cached = self.load()
            if cached is not None:
                return cached
        df = load_jira_export(path)
        self.save(df, source=str(path))
        return df
