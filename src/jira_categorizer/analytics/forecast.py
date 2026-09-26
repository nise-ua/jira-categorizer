"""Hourly forecast vs actual for today + tomorrow, with management summary.

Classical baseline: same weekday/hour median from prior weeks (no LLM).
When working offline on historical exports, "today" = max(created) date.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from jira_categorizer.data.schema import split_multi_value


def _as_of(tickets: pd.DataFrame, as_of: datetime | None) -> pd.Timestamp:
    if as_of is not None:
        ts = pd.Timestamp(as_of)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    if "created" in tickets.columns:
        created = pd.to_datetime(tickets["created"], utc=True, errors="coerce").dropna()
        if not created.empty:
            return created.max()
    return pd.Timestamp(datetime.now(timezone.utc))


def _hourly_counts(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    s = frame.copy()
    s["hour"] = s["created"].dt.floor("h")
    return s.groupby("hour").size().astype(float)


def build_forecast_actual(
    tickets: pd.DataFrame,
    *,
    lookback_weeks: int = 8,
    as_of: datetime | None = None,
    predictions: list[dict[str, Any]] | None = None,
    surges: list[dict[str, Any]] | None = None,
    suggested_labels: list[dict[str, Any]] | None = None,
    dimension: str = "labels",
) -> dict[str, Any]:
    """Return Chart.js series for today+tomorrow forecast/actual + summary."""
    frame = tickets.copy()
    if "created" not in frame.columns:
        return {
            "labels": [],
            "datasets": [],
            "summary": {
                "headline": "No timestamps in dataset",
                "status": "unknown",
                "bullets": ["Import a Jira Excel/CSV with Created dates, or connect Jira API."],
            },
        }

    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created"]).sort_values("created")
    if frame.empty:
        return {
            "labels": [],
            "datasets": [],
            "summary": {"headline": "No usable dates", "status": "unknown", "bullets": []},
        }

    now = _as_of(frame, as_of)
    today = now.normalize()
    tomorrow = today + pd.Timedelta(days=1)
    day_after = tomorrow + pd.Timedelta(days=1)
    current_hour = now.floor("h")

    # History for baseline: prior lookback_weeks, excluding today forward
    hist_start = today - pd.Timedelta(weeks=lookback_weeks)
    hist = frame[(frame["created"] >= hist_start) & (frame["created"] < today)]

    # Baseline: median count for (day_of_week, hour_of_day)
    if not hist.empty:
        tmp = hist.copy()
        tmp["dow"] = tmp["created"].dt.dayofweek
        tmp["hod"] = tmp["created"].dt.hour
        tmp["day"] = tmp["created"].dt.normalize()
        daily_hour = tmp.groupby(["day", "dow", "hod"]).size().reset_index(name="n")
        baseline = daily_hour.groupby(["dow", "hod"])["n"].median()
    else:
        baseline = pd.Series(dtype=float)

    hours = pd.date_range(today, day_after, freq="h", inclusive="left")
    forecast = []
    for h in hours:
        key = (h.dayofweek, h.hour)
        forecast.append(float(baseline.get(key, baseline.mean() if len(baseline) else 0.0)))

    # Actuals: today up to current hour (inclusive), tomorrow empty
    today_tickets = frame[(frame["created"] >= today) & (frame["created"] < tomorrow)]
    actual_counts = _hourly_counts(today_tickets)
    actual = []
    for h in hours:
        if h.normalize() == today and h <= current_hour:
            actual.append(float(actual_counts.get(h, 0.0)))
        else:
            actual.append(None)  # future / unknown

    labels = [h.strftime("%a %H:%M") for h in hours]
    # Split marker index between today and tomorrow
    tomorrow_start_idx = int(np.where(hours.normalize() == tomorrow)[0][0]) if len(hours) else 0

    cumulative_actual = float(np.nansum([v for v in actual if v is not None]))
    cumulative_forecast_to_now = float(
        sum(forecast[i] for i, h in enumerate(hours) if h <= current_hour and h.normalize() == today)
    )
    day_forecast_total = float(sum(forecast[:tomorrow_start_idx]))
    next_day_forecast_total = float(sum(forecast[tomorrow_start_idx:]))

    # Category breakdown for today
    cat_counts: dict[str, int] = {}
    for _, row in today_tickets.iterrows():
        values = split_multi_value(row.get(dimension)) if dimension in row else []
        if not values and dimension == "labels":
            values = split_multi_value(row.get("labels"))
        if not values and dimension == "impacted_area":
            values = split_multi_value(row.get("impacted_area"))
        for v in values or ["(uncategorized)"]:
            cat_counts[v] = cat_counts.get(v, 0) + 1
    top_cats = sorted(cat_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Novelty among today's tickets if predictions aligned by caller
    novel_today = 0
    if predictions is not None and len(predictions) == len(tickets):
        # map by issue_key
        pred_by_key = {}
        base = tickets.reset_index(drop=True)
        for i, row in base.iterrows():
            pred_by_key[str(row.get("issue_key"))] = predictions[i]
        for _, row in today_tickets.iterrows():
            p = pred_by_key.get(str(row.get("issue_key")), {})
            if (p.get("novelty") or {}).get("is_novel"):
                novel_today += 1
    elif predictions is not None:
        # best-effort: count novel flags if same length as today slice unavailable
        novel_today = sum(1 for p in predictions if (p.get("novelty") or {}).get("is_novel"))

    ratio = (
        cumulative_actual / cumulative_forecast_to_now
        if cumulative_forecast_to_now > 0
        else (2.0 if cumulative_actual > 0 else 1.0)
    )
    if cumulative_forecast_to_now < 0.5 and cumulative_actual <= 1:
        status = "quiet"
        headline = "Volume is quiet versus the usual pattern for this hour."
    elif ratio >= 1.75:
        status = "surge"
        headline = f"Surge vs forecast: {cumulative_actual:.0f} tickets so far (~{ratio:.1f}× expected)."
    elif ratio <= 0.55 and cumulative_forecast_to_now >= 2:
        status = "below"
        headline = f"Below forecast: {cumulative_actual:.0f} tickets so far (~{ratio:.1f}× expected)."
    else:
        status = "normal"
        headline = f"On track: {cumulative_actual:.0f} tickets so far (~{ratio:.1f}× expected for this hour)."

    bullets: list[str] = [
        f"Today forecast total ≈ {day_forecast_total:.0f}; next-day forecast ≈ {next_day_forecast_total:.0f}.",
        f"Actual through {current_hour.strftime('%H:%M')} UTC: {cumulative_actual:.0f} (expected ≈ {cumulative_forecast_to_now:.0f}).",
    ]
    if top_cats:
        tops = ", ".join(f"{c} ({n})" for c, n in top_cats[:3])
        bullets.append(f"Top {dimension} today: {tops}.")

    surge_bits = []
    for s in (surges or [])[:4]:
        surge_bits.append(f"{s.get('category')} ({s.get('kind')}, ×{s.get('rate_ratio')})")
    if surge_bits:
        bullets.append("Known-category spikes: " + "; ".join(surge_bits) + ".")
    else:
        bullets.append("No sustained known-category surge alert in the monitor window.")

    if suggested_labels:
        names = ", ".join(s.get("suggested_label", "?") for s in suggested_labels[:3])
        bullets.append(f"Brand-new problem signals / suggested labels: {names}.")
    elif novel_today:
        bullets.append(f"{novel_today} ticket(s) today look novel (low confidence vs known categories).")
    else:
        bullets.append("No brand-new category cluster detected for the current slice.")

    return {
        "as_of": now.isoformat(),
        "today": today.date().isoformat(),
        "tomorrow": tomorrow.date().isoformat(),
        "current_hour": current_hour.isoformat(),
        "tomorrow_start_index": tomorrow_start_idx,
        "labels": labels,
        "datasets": [
            {
                "label": "Forecast",
                "data": forecast,
                "borderColor": "#1F4B7A",
                "backgroundColor": "#1F4B7A22",
                "borderDash": [6, 4],
                "tension": 0.25,
                "fill": False,
                "spanGaps": True,
            },
            {
                "label": "Actual",
                "data": actual,
                "borderColor": "#C45C26",
                "backgroundColor": "#C45C2633",
                "tension": 0.2,
                "fill": False,
                "spanGaps": False,
            },
        ],
        "metrics": {
            "cumulative_actual": cumulative_actual,
            "cumulative_forecast_to_now": cumulative_forecast_to_now,
            "ratio_to_forecast": round(ratio, 3),
            "day_forecast_total": day_forecast_total,
            "next_day_forecast_total": next_day_forecast_total,
            "today_ticket_count": int(len(today_tickets)),
            "novel_today": int(novel_today),
            "top_categories": [{"category": c, "count": n} for c, n in top_cats],
        },
        "summary": {
            "headline": headline,
            "status": status,
            "bullets": bullets,
        },
    }
