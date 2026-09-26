"""Detect category surges (labels / impacted areas) vs a baseline window.

Classical stats only: rolling counts, rate ratio, and z-score against baseline.
No LLM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jira_categorizer.data.schema import split_multi_value


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _explode_categories(df: pd.DataFrame, field: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        created = row.get("created")
        values = split_multi_value(row.get(field))
        if not values:
            continue
        for v in values:
            rows.append({"created": created, "category": v, "issue_key": row.get("issue_key")})
    return pd.DataFrame(rows)


def _window_counts(
    cat_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, int]:
    if cat_df.empty:
        return {}
    mask = (cat_df["created"] >= start) & (cat_df["created"] < end)
    return cat_df.loc[mask, "category"].value_counts().to_dict()


def detect_category_surges(
    tickets: pd.DataFrame,
    *,
    cfg: dict[str, Any],
    label_field: str = "labels",
    area_field: str = "impacted_area",
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Compare recent window rates to baseline for labels and impacted areas.

    Alerts when:
      - recent_count >= min_recent_count, AND
      - rate_ratio >= min_rate_ratio OR z_score >= min_z_score
    """
    mon = cfg.get("monitoring", {}).get("surge", {})
    if not mon.get("enabled", True):
        return {"enabled": False, "surges": []}

    recent_days = int(mon.get("recent_days", 7))
    baseline_days = int(mon.get("baseline_days", 28))
    min_recent = int(mon.get("min_recent_count", 3))
    min_ratio = float(mon.get("min_rate_ratio", 2.5))
    min_z = float(mon.get("min_z_score", 2.0))
    epsilon = float(mon.get("epsilon", 0.5))  # Laplace-style for empty baseline

    if "created" not in tickets.columns:
        return {
            "enabled": True,
            "status": "skip",
            "reason": "no created timestamp column; refetch with --include-created",
            "surges": [],
        }

    frame = tickets.copy()
    frame["created"] = _parse_ts(frame["created"])
    frame = frame.dropna(subset=["created"])
    if frame.empty:
        return {"enabled": True, "status": "skip", "reason": "no parseable created timestamps", "surges": []}

    as_of_ts = pd.Timestamp(as_of or datetime.now(timezone.utc))
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    # Prefer data-relative "now" so offline exports still detect surges
    data_max = frame["created"].max()
    if data_max < as_of_ts:
        as_of_ts = data_max + pd.Timedelta(seconds=1)

    recent_start = as_of_ts - pd.Timedelta(days=recent_days)
    baseline_start = recent_start - pd.Timedelta(days=baseline_days)
    baseline_end = recent_start

    surges: list[dict[str, Any]] = []
    for kind, field in (("label", label_field), ("impacted_area", area_field)):
        cat_df = _explode_categories(frame, field)
        if cat_df.empty:
            continue
        cat_df["created"] = _parse_ts(cat_df["created"])
        recent = _window_counts(cat_df, recent_start, as_of_ts)
        baseline = _window_counts(cat_df, baseline_start, baseline_end)

        # Daily rates
        for category, r_count in recent.items():
            if r_count < min_recent:
                continue
            b_count = baseline.get(category, 0)
            recent_rate = r_count / max(recent_days, 1)
            baseline_rate = (b_count + epsilon) / max(baseline_days, 1)
            rate_ratio = recent_rate / baseline_rate

            # Z-score treating baseline daily counts as Poisson-ish: std ~= sqrt(mean)
            # Compare recent daily mean to baseline daily mean.
            baseline_daily_mean = b_count / max(baseline_days, 1)
            recent_daily_mean = r_count / max(recent_days, 1)
            std = float(np.sqrt(baseline_daily_mean + epsilon))
            z_score = (recent_daily_mean - baseline_daily_mean) / std if std > 0 else 0.0

            if rate_ratio >= min_ratio or z_score >= min_z:
                severity = "high" if rate_ratio >= min_ratio * 1.5 or z_score >= min_z * 1.5 else "medium"
                surges.append(
                    {
                        "kind": kind,
                        "category": category,
                        "severity": severity,
                        "recent_count": int(r_count),
                        "baseline_count": int(b_count),
                        "recent_days": recent_days,
                        "baseline_days": baseline_days,
                        "recent_daily_rate": round(recent_rate, 4),
                        "baseline_daily_rate": round(baseline_rate, 4),
                        "rate_ratio": round(float(rate_ratio), 3),
                        "z_score": round(float(z_score), 3),
                    }
                )

    surges.sort(key=lambda x: (x["rate_ratio"], x["z_score"]), reverse=True)
    return {
        "enabled": True,
        "status": "alert" if surges else "ok",
        "as_of": as_of_ts.isoformat(),
        "recent_window": {"start": recent_start.isoformat(), "end": as_of_ts.isoformat()},
        "baseline_window": {"start": baseline_start.isoformat(), "end": baseline_end.isoformat()},
        "thresholds": {
            "min_recent_count": min_recent,
            "min_rate_ratio": min_ratio,
            "min_z_score": min_z,
        },
        "surge_count": len(surges),
        "surges": surges,
    }


def write_surge_report(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path
