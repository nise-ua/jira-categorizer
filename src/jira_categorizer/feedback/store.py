"""Persist triage corrections and merge them into training exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from jira_categorizer.data.ingest import load_jira_export


class FeedbackStore:
    def __init__(self, feedback_dir: str | Path):
        self.feedback_dir = Path(feedback_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.feedback_dir / "corrections.jsonl"

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        payload = dict(record)
        payload.setdefault("saved_at", datetime.now(timezone.utc).isoformat())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
        return payload

    def list_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def disagreements(self) -> list[dict[str, Any]]:
        return [r for r in self.list_all() if r.get("disagreed")]

    def merge_into_training_csv(
        self,
        source_csv: str | Path,
        output_csv: str | Path,
    ) -> dict[str, Any]:
        """Overwrite labels/impacted_area for corrected issue keys; append unknowns."""
        src = Path(source_csv)
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        base = load_jira_export(src) if src.exists() else pd.DataFrame()
        feedback = self.list_all()
        if not feedback:
            if src.exists():
                base.to_csv(out, index=False)
            return {"merged": 0, "output": str(out)}

        by_key = {r["issue_key"]: r for r in feedback if r.get("issue_key")}
        if base.empty:
            rows = []
            for r in by_key.values():
                rows.append(
                    {
                        "issue_key": r["issue_key"],
                        "summary": r.get("summary", ""),
                        "description": r.get("description", ""),
                        "labels": "|".join(r.get("selected_labels") or []),
                        "impacted_area": "|".join(r.get("selected_areas") or []),
                        "created": r.get("created", ""),
                    }
                )
            pd.DataFrame(rows).to_csv(out, index=False)
            return {"merged": len(rows), "output": str(out)}

        updated = 0
        keys = set(base["issue_key"].astype(str))
        for idx, row in base.iterrows():
            key = str(row["issue_key"])
            if key not in by_key:
                continue
            fb = by_key[key]
            base.at[idx, "labels"] = "|".join(fb.get("selected_labels") or [])
            base.at[idx, "impacted_area"] = "|".join(fb.get("selected_areas") or [])
            updated += 1

        extras = []
        for key, fb in by_key.items():
            if key in keys:
                continue
            extras.append(
                {
                    "issue_key": key,
                    "summary": fb.get("summary", ""),
                    "description": fb.get("description", ""),
                    "labels": "|".join(fb.get("selected_labels") or []),
                    "impacted_area": "|".join(fb.get("selected_areas") or []),
                    "created": fb.get("created", ""),
                }
            )
        if extras:
            base = pd.concat([base, pd.DataFrame(extras)], ignore_index=True)
        base.to_csv(out, index=False)
        return {"merged": updated + len(extras), "updated": updated, "appended": len(extras), "output": str(out)}
