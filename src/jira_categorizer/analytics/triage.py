"""Build triage rows: ML suggestions + dropdown options for human review."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from jira_categorizer.data.preprocess import clean_text
from jira_categorizer.data.schema import split_multi_value


def _snippet(text: str, n: int = 180) -> str:
    text = clean_text(text or "", strip_html_tags=True)
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def build_triage_queue(
    tickets: pd.DataFrame,
    categorizer: Any,
    *,
    days: int = 1,
    limit: int = 50,
    suggested_new_labels: list[str] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    frame = tickets.copy()
    if "created" in frame.columns:
        frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
        as_of_ts = pd.Timestamp(as_of or frame["created"].max())
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        start = as_of_ts - pd.Timedelta(days=days)
        frame = frame[(frame["created"] >= start) & (frame["created"] <= as_of_ts)]
        frame = frame.sort_values("created", ascending=False)
    else:
        as_of_ts = pd.Timestamp(datetime.now(timezone.utc))
        start = None

    frame = frame.head(limit).reset_index(drop=True)
    if frame.empty:
        return {
            "days": days,
            "count": 0,
            "items": [],
            "label_options": [],
            "area_options": [],
            "window_start": start.isoformat() if start is not None else None,
            "window_end": as_of_ts.isoformat(),
        }

    preds = categorizer.predict_texts(
        [
            f"{str(r.get('summary', ''))} {str(r.get('description', ''))}".strip()
            for _, r in frame.iterrows()
        ]
    )

    # Known vocabulary from model + current data + suggested new labels
    label_options = sorted(set(map(str, getattr(categorizer.labels_mlb, "classes_", []))))
    area_options = sorted(set(map(str, getattr(categorizer.area_encoder, "classes_", []))))
    for _, row in frame.iterrows():
        label_options.extend(split_multi_value(row.get("labels")))
        area_options.extend(split_multi_value(row.get("impacted_area")))
    if suggested_new_labels:
        label_options.extend(suggested_new_labels)
    label_options = sorted({x for x in label_options if x and x != "__none__"})
    area_options = sorted({x for x in area_options if x and x != "__none__"})

    items = []
    for i, row in frame.iterrows():
        pred = preds[i]
        suggested_labels = list(pred.get("labels") or [])
        # Include top scored even if below threshold for dropdown default
        if not suggested_labels and pred.get("labels_scored"):
            suggested_labels = [pred["labels_scored"][0]["label"]]
        suggested_area = pred.get("impacted_area")
        novelty = pred.get("novelty") or {}
        items.append(
            {
                "issue_key": str(row.get("issue_key", "")),
                "summary": str(row.get("summary", "")),
                "description": str(row.get("description", "")),
                "description_snippet": _snippet(str(row.get("description", ""))),
                "created": row["created"].isoformat() if pd.notna(row.get("created")) else None,
                "current_labels": split_multi_value(row.get("labels")),
                "current_area": (split_multi_value(row.get("impacted_area")) or [None])[0],
                "suggested_labels": suggested_labels,
                "suggested_label": suggested_labels[0] if suggested_labels else None,
                "suggested_labels_scored": pred.get("labels_scored") or [],
                "suggested_area": suggested_area,
                "suggested_area_scored": pred.get("impacted_area_scored") or [],
                "novelty": novelty,
                "is_novel": bool(novelty.get("is_novel")),
            }
        )

    return {
        "days": days,
        "count": len(items),
        "items": items,
        "label_options": label_options,
        "area_options": area_options,
        "window_start": start.isoformat() if start is not None else None,
        "window_end": as_of_ts.isoformat(),
    }
