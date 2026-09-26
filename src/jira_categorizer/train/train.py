"""Train labels + impacted-area models from historical Jira tickets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from jira_categorizer.config import load_config
from jira_categorizer.data.ingest import build_training_frame, load_jira_export
from jira_categorizer.enclave.attestation import dump_report, require_attestation
from jira_categorizer.features.text_features import build_text_vectorizer
from jira_categorizer.models.area_model import build_area_estimator, fit_area_encoder
from jira_categorizer.models.labels_model import build_labels_estimator, fit_label_binarizer
from jira_categorizer.registry.model_store import ModelStore
from jira_categorizer.train.evaluate import evaluate_area, evaluate_labels, passes_promotion_gates


def _threshold_predict(estimator, X, threshold: float):
    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            proba = np.asarray(proba)
            if proba.ndim == 1:
                proba = proba.reshape(-1, 1)
            return (proba >= threshold).astype(int)
        except Exception:
            pass
    if hasattr(estimator, "decision_function"):
        decision = np.asarray(estimator.decision_function(X))
        # Map decision margin to soft threshold around 0
        return (decision >= 0).astype(int)
    return np.asarray(estimator.predict(X))


def train(cfg: dict[str, Any], data_path: str | None = None) -> dict[str, Any]:
    report = require_attestation(cfg)
    paths = cfg["paths"]
    for key in ("processed_dir", "model_dir", "metrics_dir", "reports_dir", "sealed_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)

    dump_report(report, str(Path(paths["reports_dir"]) / "attestation.json"))

    export_path = data_path or paths["raw_data"]
    raw = load_jira_export(export_path)
    frame = build_training_frame(raw, cfg)

    processed_path = Path(paths["processed_dir"]) / "train_frame.parquet"
    try:
        frame.to_parquet(processed_path, index=False)
    except Exception:
        # parquet optional; always keep csv
        frame.to_csv(Path(paths["processed_dir"]) / "train_frame.csv", index=False)

    texts = frame["text"].tolist()
    y_labels = frame["labels"].tolist()
    y_area = frame["primary_area"].tolist()

    # Rows with at least one label and/or a known area
    has_labels = [bool(labs) for labs in y_labels]
    has_area = [bool(a) for a in y_area]

    data_cfg = cfg["data"]
    rs = int(data_cfg.get("random_state", 42))
    test_size = float(data_cfg.get("test_size", 0.2))

    idx = np.arange(len(frame))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=rs)

    vectorizer = build_text_vectorizer(cfg)
    X_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    X_test = vectorizer.transform([texts[i] for i in test_idx])

    # --- Labels (multi-label) ---
    label_train_mask = [has_labels[i] for i in train_idx]
    label_test_mask = [has_labels[i] for i in test_idx]
    mlb = fit_label_binarizer([y_labels[i] for i in train_idx if has_labels[i]])
    labels_estimator = build_labels_estimator(cfg)

    metrics: dict[str, Any] = {"labels": {}, "area": {}, "counts": {}}
    metrics["counts"] = {
        "rows": int(len(frame)),
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
        "label_classes": int(len(mlb.classes_)),
    }

    if len(mlb.classes_) == 0 or not any(label_train_mask):
        # Fit a dummy-safe empty path
        from sklearn.dummy import DummyClassifier
        from sklearn.multiclass import OneVsRestClassifier

        labels_estimator = OneVsRestClassifier(DummyClassifier(strategy="prior"))
        # Single fake class so pipeline stays loadable
        mlb.fit([["__none__"]])
        Y_dummy = mlb.transform([["__none__"]] * max(1, sum(label_train_mask) or 1))
        X_dummy = X_train[: Y_dummy.shape[0]]
        labels_estimator.fit(X_dummy, Y_dummy)
        metrics["labels"] = {"f1_macro": 0.0, "note": "insufficient label data"}
    else:
        Y_train = mlb.transform([y_labels[i] for i in train_idx if has_labels[i]])
        X_lab_train = X_train[np.array(label_train_mask)]
        labels_estimator.fit(X_lab_train, Y_train)

        if any(label_test_mask):
            Y_test = mlb.transform([y_labels[i] for i in test_idx if has_labels[i]])
            X_lab_test = X_test[np.array(label_test_mask)]
            thr = float(cfg["models"]["labels"].get("decision_threshold", 0.35))
            Y_hat = _threshold_predict(labels_estimator, X_lab_test, thr)
            # Align shapes if needed
            if Y_hat.shape != Y_test.shape:
                Y_hat = labels_estimator.predict(X_lab_test)
            metrics["labels"] = evaluate_labels(Y_test, Y_hat, list(mlb.classes_))
        else:
            metrics["labels"] = {"f1_macro": 0.0, "note": "no labeled test rows"}

    # --- Impacted area (multiclass) ---
    area_train_idx = [i for i in train_idx if has_area[i]]
    area_test_idx = [i for i in test_idx if has_area[i]]
    area_encoder = fit_area_encoder([y_area[i] for i in area_train_idx])
    area_estimator = build_area_estimator(cfg)
    metrics["counts"]["area_classes"] = int(len(getattr(area_encoder, "classes_", [])))

    if len(getattr(area_encoder, "classes_", [])) < 2 or not area_train_idx:
        from sklearn.dummy import DummyClassifier

        area_estimator = DummyClassifier(strategy="most_frequent")
        if not area_train_idx:
            area_encoder.fit(["__none__"])
            area_estimator.fit(X_train[:1], [0])
        else:
            y_enc = area_encoder.transform([y_area[i] for i in area_train_idx])
            # Map train_idx positions
            pos = {i: n for n, i in enumerate(train_idx)}
            X_a = X_train[[pos[i] for i in area_train_idx]]
            area_estimator.fit(X_a, y_enc)
        metrics["area"] = {"f1_macro": 0.0, "note": "insufficient area data"}
    else:
        pos_train = {i: n for n, i in enumerate(train_idx)}
        X_a_train = X_train[[pos_train[i] for i in area_train_idx]]
        y_a_train = area_encoder.transform([y_area[i] for i in area_train_idx])
        area_estimator.fit(X_a_train, y_a_train)

        if area_test_idx:
            pos_test = {i: n for n, i in enumerate(test_idx)}
            X_a_test = X_test[[pos_test[i] for i in area_test_idx]]
            y_a_test = area_encoder.transform([y_area[i] for i in area_test_idx])
            y_hat = area_estimator.predict(X_a_test)
            metrics["area"] = evaluate_area(y_a_test, y_hat, list(area_encoder.classes_))
        else:
            metrics["area"] = {"f1_macro": 0.0, "note": "no area test rows"}

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = ModelStore(paths["model_dir"], paths.get("sealed_dir"))
    promote = None
    if cfg.get("training", {}).get("promote_if_better", True) and passes_promotion_gates(metrics, cfg):
        promote = cfg.get("mlops", {}).get("registry_name", "production")
    elif not (Path(paths["model_dir"]) / "production").exists():
        # Always bootstrap first production bundle for local/dev
        promote = "production"

    seal = bool(cfg.get("enclave", {}).get("seal_models", False))
    key_path = cfg.get("enclave", {}).get("local_key_path")
    manifest = store.save_bundle(
        version=version,
        vectorizer=vectorizer,
        labels_estimator=labels_estimator,
        labels_mlb=mlb,
        area_estimator=area_estimator,
        area_encoder=area_encoder,
        metrics=metrics,
        config=cfg,
        seal=seal,
        key_path=key_path,
        promote_as=promote,
    )

    metrics_path = Path(paths["metrics_dir"]) / f"metrics_{version}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (Path(paths["metrics_dir"]) / "latest.json").write_text(
        json.dumps({"version": version, "promoted": promote, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )

    return {
        "version": version,
        "promoted": promote,
        "metrics": metrics,
        "manifest": manifest.to_dict(),
        "attestation": report.to_dict(),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Jira categorizer (no LLM)")
    parser.add_argument("--config", default=None, help="Path to config YAML")
    parser.add_argument("--data", default=None, help="Path to Jira CSV/JSON export")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = train(cfg, data_path=args.data)
    print(json.dumps({"version": result["version"], "promoted": result["promoted"], "metrics": {
        "labels_f1_macro": result["metrics"].get("labels", {}).get("f1_macro"),
        "area_f1_macro": result["metrics"].get("area", {}).get("f1_macro"),
        "counts": result["metrics"].get("counts"),
    }}, indent=2))


if __name__ == "__main__":
    main()
