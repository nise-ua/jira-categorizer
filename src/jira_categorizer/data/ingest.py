"""Ingest historical Jira exports for training inside the enclave."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from jira_categorizer.data.preprocess import combine_summary_description
from jira_categorizer.data.schema import TicketRecord, split_multi_value


REQUIRED_COLUMNS = {"summary", "description"}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {c: c.strip().lower().replace(" ", "_") for c in df.columns}
    # Common Jira export aliases
    aliases = {
        "issue_key": "issue_key",
        "key": "issue_key",
        "issue": "issue_key",
        "summary": "summary",
        "description": "description",
        "labels": "labels",
        "label": "labels",
        "impacted_area": "impacted_area",
        "impacted_areas": "impacted_area",
        "components": "impacted_area",
        "component/s": "impacted_area",
        "component": "impacted_area",
        "custom_field_impacted_area": "impacted_area",
        "created": "created",
        "created_date": "created",
        "updated": "updated",
    }
    renamed = {}
    for original, lowered in mapping.items():
        renamed[original] = aliases.get(lowered, lowered)
    out = df.rename(columns=renamed)
    return out


def load_jira_export(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Jira export not found: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".json", ".jsonl"}:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            df = pd.DataFrame(rows)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "issues" in payload:
                rows = []
                for issue in payload["issues"]:
                    fields = issue.get("fields", {})
                    rows.append(
                        {
                            "issue_key": issue.get("key"),
                            "summary": fields.get("summary"),
                            "description": fields.get("description"),
                            "labels": fields.get("labels"),
                            "impacted_area": fields.get("customfield_impacted_area")
                            or fields.get("components"),
                            "created": fields.get("created"),
                            "updated": fields.get("updated"),
                        }
                    )
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame(payload)
    else:
        raise ValueError(f"Unsupported export format: {path.suffix}")

    df = _normalize_columns(df)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Export missing required columns: {sorted(missing)}")

    if "issue_key" not in df.columns:
        df["issue_key"] = [f"ROW-{i}" for i in range(len(df))]
    if "labels" not in df.columns:
        df["labels"] = ""
    if "impacted_area" not in df.columns:
        df["impacted_area"] = ""

    df["summary"] = df["summary"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    if "created" not in df.columns:
        df["created"] = pd.NaT
    return df


def build_training_frame(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Produce a clean frame with text + parsed targets."""
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    strip_html = bool(feat_cfg.get("strip_html", True))

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        labels = split_multi_value(row.get(data_cfg["label_field"]))
        areas = split_multi_value(row.get(data_cfg["area_field"]))
        text = combine_summary_description(
            row.get("summary", ""),
            row.get("description", ""),
            strip_html_tags=strip_html,
        )
        if not text:
            continue
        records.append(
            {
                "issue_key": str(row.get(data_cfg.get("issue_key_field", "issue_key"), "")),
                "text": text,
                "labels": labels,
                "impacted_areas": areas,
                # Primary area for multiclass path (first value)
                "primary_area": areas[0] if areas else "",
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("No usable training rows after preprocessing")

    # Filter rare targets
    min_label = int(data_cfg.get("min_label_support", 2))
    min_area = int(data_cfg.get("min_area_support", 2))

    label_counts: dict[str, int] = {}
    for labs in frame["labels"]:
        for lab in labs:
            label_counts[lab] = label_counts.get(lab, 0) + 1
    keep_labels = {k for k, v in label_counts.items() if v >= min_label}
    frame["labels"] = frame["labels"].apply(lambda xs: [x for x in xs if x in keep_labels])

    area_counts = frame["primary_area"].value_counts().to_dict()
    keep_areas = {k for k, v in area_counts.items() if k and v >= min_area}
    frame.loc[~frame["primary_area"].isin(keep_areas), "primary_area"] = ""
    frame["impacted_areas"] = frame["impacted_areas"].apply(
        lambda xs: [x for x in xs if x in keep_areas]
    )

    return frame.reset_index(drop=True)


def to_ticket_records(frame: pd.DataFrame) -> list[TicketRecord]:
    out: list[TicketRecord] = []
    for _, row in frame.iterrows():
        out.append(
            TicketRecord(
                issue_key=str(row["issue_key"]),
                summary="",
                description=str(row["text"]),
                labels=list(row["labels"]),
                impacted_areas=list(row["impacted_areas"]),
            )
        )
    return out
