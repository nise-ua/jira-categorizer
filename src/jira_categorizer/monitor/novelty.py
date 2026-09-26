"""Per-ticket novelty scoring from classifier confidences."""

from __future__ import annotations

from typing import Any


def _max_score(scored: list[dict[str, Any]], key: str = "score") -> float:
    if not scored:
        return 0.0
    return float(max(item.get(key, 0.0) for item in scored))


def score_novelty(
    predictions: list[dict[str, Any]],
    *,
    label_conf_threshold: float,
    area_conf_threshold: float,
) -> list[dict[str, Any]]:
    """Attach novelty flags when label and area confidence are both weak."""
    out: list[dict[str, Any]] = []
    for pred in predictions:
        label_conf = _max_score(pred.get("labels_scored") or [])
        area_conf = _max_score(pred.get("impacted_area_scored") or [])
        weak_labels = (not pred.get("labels")) or label_conf < label_conf_threshold
        weak_area = area_conf < area_conf_threshold
        is_novel = weak_labels and weak_area
        reasons = []
        if weak_labels:
            reasons.append("low_label_confidence")
        if weak_area:
            reasons.append("low_area_confidence")
        item = dict(pred)
        item["novelty"] = {
            "is_novel": bool(is_novel),
            "label_confidence": round(label_conf, 4),
            "area_confidence": round(area_conf, 4),
            "reasons": reasons if is_novel else [],
        }
        out.append(item)
    return out
