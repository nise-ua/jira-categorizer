"""Historical category timeseries for UI charts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from jira_categorizer.data.schema import split_multi_value


WINDOW_DAYS = {"1d": 1, "1w": 7, "1m": 30, "3m": 90}


def _bucket_freq(window: str) -> str:
    if window == "1d":
        return "h"  # hourly for one day
    return "D"


def build_timeseries(
    tickets: pd.DataFrame,
    *,
    dimension: str = "labels",
    window: str = "1m",
    top_n: int = 8,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Return Chart.js-friendly series for a category dimension over a window."""
    if window not in WINDOW_DAYS:
        raise ValueError(f"Unsupported window: {window}")
    if dimension not in {"labels", "impacted_area", "issuetype", "status"}:
        raise ValueError(f"Unsupported dimension: {dimension}")

    frame = tickets.copy()
    if "created" not in frame.columns:
        return {
            "dimension": dimension,
            "window": window,
            "labels": [],
            "datasets": [],
            "note": "No created timestamps available",
        }

    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created"])
    if frame.empty:
        return {"dimension": dimension, "window": window, "labels": [], "datasets": []}

    as_of_ts = pd.Timestamp(as_of or frame["created"].max())
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    start = as_of_ts - pd.Timedelta(days=WINDOW_DAYS[window])
    frame = frame[(frame["created"] >= start) & (frame["created"] <= as_of_ts)]

    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        if dimension in {"labels", "impacted_area"}:
            values = split_multi_value(row.get(dimension))
        else:
            raw = row.get(dimension)
            values = [str(raw).strip().lower()] if raw and str(raw).strip() else []
        for v in values:
            rows.append({"created": row["created"], "category": v, "issue_key": row.get("issue_key")})

    if not rows:
        return {
            "dimension": dimension,
            "window": window,
            "start": start.isoformat(),
            "end": as_of_ts.isoformat(),
            "labels": [],
            "datasets": [],
            "totals": {},
        }

    long = pd.DataFrame(rows)
    totals = long["category"].value_counts().to_dict()
    keep = [c for c, _ in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    long = long[long["category"].isin(keep)]

    freq = _bucket_freq(window)
    long["bucket"] = long["created"].dt.floor(freq)
    pivot = (
        long.groupby(["bucket", "category"]).size().unstack(fill_value=0).reindex(columns=keep, fill_value=0)
    )
    # Full index for continuity
    full_idx = pd.date_range(start=start.floor(freq), end=as_of_ts.floor(freq), freq=freq)
    pivot = pivot.reindex(full_idx, fill_value=0)

    if freq == "h":
        labels = [ts.strftime("%m-%d %H:%M") for ts in pivot.index]
    else:
        labels = [ts.strftime("%Y-%m-%d") for ts in pivot.index]

    palette = [
        "#0F6E56",
        "#C45C26",
        "#1F4B7A",
        "#8B3A4A",
        "#5B6B2F",
        "#6B4C9A",
        "#A67C2A",
        "#2F6F8F",
    ]
    datasets = []
    for i, cat in enumerate(keep):
        datasets.append(
            {
                "label": cat,
                "data": [int(v) for v in pivot[cat].tolist()],
                "borderColor": palette[i % len(palette)],
                "backgroundColor": palette[i % len(palette)] + "33",
                "tension": 0.25,
                "fill": False,
            }
        )

    return {
        "dimension": dimension,
        "window": window,
        "start": start.isoformat(),
        "end": as_of_ts.isoformat(),
        "labels": labels,
        "datasets": datasets,
        "totals": {k: int(v) for k, v in totals.items() if k in keep},
        "bucket": "hour" if freq == "h" else "day",
    }


def build_novelty_timeseries(
    tickets: pd.DataFrame,
    predictions: list[dict[str, Any]],
    *,
    window: str = "1m",
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Line series of novel vs known tickets over time."""
    if window not in WINDOW_DAYS:
        raise ValueError(f"Unsupported window: {window}")
    frame = tickets.copy().reset_index(drop=True)
    if "created" not in frame.columns or len(predictions) != len(frame):
        # Align by predicting caller responsibility; if mismatch, trim
        n = min(len(frame), len(predictions))
        frame = frame.iloc[:n].copy()
        predictions = predictions[:n]

    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    frame["is_novel"] = [bool((p.get("novelty") or {}).get("is_novel")) for p in predictions]
    frame = frame.dropna(subset=["created"])
    if frame.empty:
        return {"window": window, "labels": [], "datasets": []}

    as_of_ts = pd.Timestamp(as_of or frame["created"].max())
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    start = as_of_ts - pd.Timedelta(days=WINDOW_DAYS[window])
    frame = frame[(frame["created"] >= start) & (frame["created"] <= as_of_ts)]
    freq = _bucket_freq(window)
    frame["bucket"] = frame["created"].dt.floor(freq)
    known = frame.loc[~frame["is_novel"]].groupby("bucket").size()
    novel = frame.loc[frame["is_novel"]].groupby("bucket").size()
    full_idx = pd.date_range(start=start.floor(freq), end=as_of_ts.floor(freq), freq=freq)
    known = known.reindex(full_idx, fill_value=0)
    novel = novel.reindex(full_idx, fill_value=0)
    labels = [
        ts.strftime("%m-%d %H:%M") if freq == "h" else ts.strftime("%Y-%m-%d") for ts in full_idx
    ]
    return {
        "window": window,
        "start": start.isoformat(),
        "end": as_of_ts.isoformat(),
        "labels": labels,
        "datasets": [
            {
                "label": "known categories",
                "data": [int(v) for v in known.tolist()],
                "borderColor": "#1F4B7A",
                "backgroundColor": "#1F4B7A33",
                "tension": 0.25,
                "fill": False,
            },
            {
                "label": "novel / new-type",
                "data": [int(v) for v in novel.tolist()],
                "borderColor": "#C45C26",
                "backgroundColor": "#C45C2633",
                "tension": 0.25,
                "fill": False,
            },
        ],
    }
