"""Multi-label classifier for Jira Labels learned from history."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC


def build_labels_estimator(cfg: dict[str, Any]) -> OneVsRestClassifier:
    mcfg = cfg["models"]["labels"]
    algo = mcfg.get("algorithm", "logistic_regression")
    if algo == "linear_svc":
        base = LinearSVC(
            C=float(mcfg.get("C", 1.0)),
            class_weight=mcfg.get("class_weight", "balanced"),
            max_iter=int(mcfg.get("max_iter", 2000)),
        )
    else:
        base = LogisticRegression(
            C=float(mcfg.get("C", 1.0)),
            class_weight=mcfg.get("class_weight", "balanced"),
            max_iter=int(mcfg.get("max_iter", 2000)),
            solver="liblinear",
        )
    return OneVsRestClassifier(base, n_jobs=1)


def fit_label_binarizer(y_labels: list[list[str]]) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer()
    mlb.fit(y_labels)
    return mlb


def predict_labels(
    estimator: OneVsRestClassifier,
    mlb: MultiLabelBinarizer,
    X,
    *,
    threshold: float = 0.35,
    top_k: int = 5,
) -> list[list[dict[str, Any]]]:
    """Return scored label predictions per row."""
    classes = list(mlb.classes_)
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
        except Exception:
            # Some OneVsRest setups return list of arrays
            if isinstance(estimator.predict_proba(X), list):
                proba = np.column_stack(
                    [p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p.ravel() for p in estimator.predict_proba(X)]
                )
            else:
                decision = estimator.decision_function(X)
                proba = 1.0 / (1.0 + np.exp(-decision))
    elif hasattr(estimator, "decision_function"):
        decision = estimator.decision_function(X)
        proba = 1.0 / (1.0 + np.exp(-np.asarray(decision)))
    else:
        pred = estimator.predict(X)
        return [
            [{"label": classes[i], "score": 1.0} for i, v in enumerate(row) if v]
            for row in np.asarray(pred)
        ]

    proba = np.asarray(proba)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)

    results: list[list[dict[str, Any]]] = []
    for row in proba:
        scored = sorted(
            [{"label": classes[i], "score": float(row[i])} for i in range(len(classes))],
            key=lambda x: x["score"],
            reverse=True,
        )
        selected = [s for s in scored if s["score"] >= threshold][:top_k]
        if not selected and scored:
            selected = scored[:1]
        results.append(selected)
    return results
