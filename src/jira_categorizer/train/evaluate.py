"""Evaluation metrics for labels (multi-label) and impacted area (multiclass)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def _safe_classification_report(y_true, y_pred, target_names: list[str] | None):
    if not target_names:
        return None
    try:
        labels = list(range(len(target_names)))
        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        )
    except Exception:
        return classification_report(y_true, y_pred, zero_division=0, output_dict=True)


def evaluate_labels(y_true, y_pred, label_names: list[str] | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"f1_macro": 0.0, "f1_micro": 0.0, "hamming_loss": 1.0}
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "report": _safe_classification_report(y_true, y_pred, label_names),
    }


def evaluate_area(y_true, y_pred, area_names: list[str] | None = None) -> dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return {"f1_macro": 0.0, "accuracy": 0.0}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "report": _safe_classification_report(y_true, y_pred, area_names),
    }


def passes_promotion_gates(metrics: dict[str, Any], cfg: dict[str, Any]) -> bool:
    train_cfg = cfg.get("training", {})
    labels_ok = metrics.get("labels", {}).get("f1_macro", 0.0) >= float(
        train_cfg.get("min_f1_macro_labels", 0.0)
    )
    area_ok = metrics.get("area", {}).get("f1_macro", 0.0) >= float(
        train_cfg.get("min_f1_macro_area", 0.0)
    )
    return labels_ok and area_ok
