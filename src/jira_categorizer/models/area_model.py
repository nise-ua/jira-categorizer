"""Impacted-area classifier learned from historical tickets."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


def build_area_estimator(cfg: dict[str, Any]):
    mcfg = cfg["models"]["area"]
    algo = mcfg.get("algorithm", "logistic_regression")
    if algo == "linear_svc":
        return LinearSVC(
            C=float(mcfg.get("C", 1.0)),
            class_weight=mcfg.get("class_weight", "balanced"),
            max_iter=int(mcfg.get("max_iter", 2000)),
        )
    return LogisticRegression(
        C=float(mcfg.get("C", 1.0)),
        class_weight=mcfg.get("class_weight", "balanced"),
        max_iter=int(mcfg.get("max_iter", 2000)),
        solver="lbfgs",
    )


def fit_area_encoder(y_areas: list[str]) -> LabelEncoder:
    enc = LabelEncoder()
    enc.fit([a for a in y_areas if a])
    return enc


def predict_areas(
    estimator,
    encoder: LabelEncoder,
    X,
    *,
    top_k: int = 3,
) -> list[list[dict[str, Any]]]:
    classes = list(encoder.classes_)
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X))
    elif hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X))
        if decision.ndim == 1:
            # Binary
            p1 = 1.0 / (1.0 + np.exp(-decision))
            proba = np.column_stack([1 - p1, p1])
        else:
            # Softmax over decision
            e = np.exp(decision - decision.max(axis=1, keepdims=True))
            proba = e / e.sum(axis=1, keepdims=True)
    else:
        pred = estimator.predict(X)
        return [[{"area": classes[int(i)], "score": 1.0}] for i in pred]

    results: list[list[dict[str, Any]]] = []
    for row in proba:
        scored = sorted(
            [{"area": classes[i], "score": float(row[i])} for i in range(len(classes))],
            key=lambda x: x["score"],
            reverse=True,
        )
        results.append(scored[:top_k])
    return results
