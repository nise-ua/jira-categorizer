"""Hierarchical ticket-income tree for executive live view.

Layers:
  1) total
  2) impacted area (larger category)
  3) label (within area)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from jira_categorizer.data.schema import split_multi_value

PERIOD_DAYS = {"1d": 1, "7d": 7, "1m": 30, "3m": 90}


def build_income_tree(
    tickets: pd.DataFrame,
    *,
    period: str = "1d",
    as_of: datetime | None = None,
    top_areas: int = 8,
    top_labels_per_area: int = 6,
) -> dict[str, Any]:
    if period not in PERIOD_DAYS:
        raise ValueError(f"Unsupported period: {period}")

    frame = tickets.copy()
    if "created" not in frame.columns or frame.empty:
        return {
            "period": period,
            "total": 0,
            "tree": {"name": "Tickets", "value": 0, "children": []},
            "sunburst": [],
            "headline": "No ticket timestamps available",
        }

    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created"])
    if frame.empty:
        return {
            "period": period,
            "total": 0,
            "tree": {"name": "Tickets", "value": 0, "children": []},
            "sunburst": [],
            "headline": "No usable dates",
        }

    as_of_ts = pd.Timestamp(as_of or frame["created"].max())
    if as_of_ts.tzinfo is None:
        as_of_ts = as_of_ts.tz_localize("UTC")
    start = as_of_ts - pd.Timedelta(days=PERIOD_DAYS[period])
    frame = frame[(frame["created"] >= start) & (frame["created"] <= as_of_ts)]
    total = int(len(frame))

    # Aggregate area -> label counts (a ticket can contribute to multiple labels)
    area_totals: dict[str, int] = {}
    area_label: dict[str, dict[str, int]] = {}
    for _, row in frame.iterrows():
        areas = split_multi_value(row.get("impacted_area")) or ["(unassigned area)"]
        labels = split_multi_value(row.get("labels")) or ["(unlabeled)"]
        # Count ticket once per area for area totals; labels nested under each area
        for area in areas:
            area_totals[area] = area_totals.get(area, 0) + 1
            bucket = area_label.setdefault(area, {})
            for lab in labels:
                bucket[lab] = bucket.get(lab, 0) + 1

    top_area_names = [
        a for a, _ in sorted(area_totals.items(), key=lambda x: x[1], reverse=True)[:top_areas]
    ]
    other_areas = [a for a in area_totals if a not in top_area_names]

    children: list[dict[str, Any]] = []
    sunburst: list[dict[str, Any]] = [{"name": "Tickets", "children": []}]

    def _label_children(area: str) -> list[dict[str, Any]]:
        labs = area_label.get(area, {})
        ranked = sorted(labs.items(), key=lambda x: x[1], reverse=True)
        keep = ranked[:top_labels_per_area]
        rest = ranked[top_labels_per_area:]
        out = [{"name": lab, "value": int(n)} for lab, n in keep]
        if rest:
            out.append({"name": "other labels", "value": int(sum(n for _, n in rest))})
        return out

    sun_children = []
    for area in top_area_names:
        labs = _label_children(area)
        node = {
            "name": area,
            "value": int(area_totals[area]),
            "children": labs,
        }
        children.append(node)
        sun_children.append(
            {
                "name": area,
                "value": int(area_totals[area]),
                "children": [{"name": c["name"], "value": c["value"]} for c in labs],
            }
        )

    if other_areas:
        other_val = int(sum(area_totals[a] for a in other_areas))
        children.append({"name": "other areas", "value": other_val, "children": []})
        sun_children.append({"name": "other areas", "value": other_val})

    tree = {"name": "Tickets", "value": total, "children": children}
    sunburst = [{"name": "Tickets", "children": sun_children}]

    # Executive one-liner
    if total == 0:
        headline = f"No ticket income in the selected {period} window."
    else:
        lead = top_area_names[0] if top_area_names else "n/a"
        lead_n = area_totals.get(lead, 0)
        pct = 100.0 * lead_n / total if total else 0
        headline = (
            f"{total} tickets in {period} · largest branch “{lead}” ({lead_n}, {pct:.0f}% of income)"
        )

    return {
        "period": period,
        "start": start.isoformat(),
        "end": as_of_ts.isoformat(),
        "total": total,
        "tree": tree,
        "sunburst": sunburst[0],
        "headline": headline,
        "layers": ["total", "impacted_area", "label"],
        "top_areas": [{"name": a, "count": int(area_totals[a])} for a in top_area_names],
    }
