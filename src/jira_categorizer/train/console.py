"""Training console: windowed train, live logs, holdout-day validation (no LLM)."""

from __future__ import annotations

import json
import threading
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from jira_categorizer.data.ingest import build_training_frame
from jira_categorizer.data.schema import split_multi_value
from jira_categorizer.enclave.attestation import dump_report, require_attestation
from jira_categorizer.features.text_features import build_text_vectorizer
from jira_categorizer.infer.predict import Categorizer
from jira_categorizer.models.area_model import build_area_estimator, fit_area_encoder, predict_areas
from jira_categorizer.models.labels_model import (
    build_labels_estimator,
    fit_label_binarizer,
    predict_labels,
)
from jira_categorizer.registry.model_store import ModelStore
from jira_categorizer.train.evaluate import evaluate_area, evaluate_labels, passes_promotion_gates
from jira_categorizer.train.train import _threshold_predict


WINDOW_DAYS = {"1m": 30, "1y": 365, "all": None}


class TrainJobState:
    def __init__(self, reports_dir: str | Path):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.reports_dir / "train_job.json"
        self.log_path = self.reports_dir / "train_console.log"
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def status(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"state": "idle", "logs": self.tail_logs()}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        data["logs"] = self.tail_logs()
        return data

    def tail_logs(self, n: int = 200) -> list[str]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def start(self, fn: Callable[[], None]) -> dict[str, Any]:
        with self._lock:
            cur = self.status()
            if cur.get("state") == "running":
                return cur
            self.log_path.write_text("", encoding="utf-8")
            self._write(
                {
                    "state": "running",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "finished_at": None,
                    "error": None,
                    "result": None,
                }
            )

            def runner() -> None:
                try:
                    fn()
                except Exception as exc:
                    self.log(f"ERROR: {exc}")
                    self.log(traceback.format_exc())
                    self._write(
                        {
                            "state": "failed",
                            "started_at": cur.get("started_at"),
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "error": str(exc),
                            "result": None,
                        }
                    )

            self._thread = threading.Thread(target=runner, daemon=True)
            self._thread.start()
            return self.status()


def apply_train_params(cfg: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(cfg)
    if "C" in params:
        out["models"]["labels"]["C"] = float(params["C"])
        out["models"]["area"]["C"] = float(params["C"])
    if "decision_threshold" in params:
        out["models"]["labels"]["decision_threshold"] = float(params["decision_threshold"])
    if "max_word_features" in params:
        out["features"]["max_word_features"] = int(params["max_word_features"])
    if "max_char_features" in params:
        out["features"]["max_char_features"] = int(params["max_char_features"])
    if "min_df" in params:
        out["features"]["min_df"] = int(params["min_df"])
    if "test_size" in params:
        out["data"]["test_size"] = float(params["test_size"])
    if "min_label_support" in params:
        out["data"]["min_label_support"] = int(params["min_label_support"])
    if "min_area_support" in params:
        out["data"]["min_area_support"] = int(params["min_area_support"])
    if "algorithm" in params:
        out["models"]["labels"]["algorithm"] = str(params["algorithm"])
        out["models"]["area"]["algorithm"] = str(params["algorithm"])
    return out


def filter_by_window(raw: pd.DataFrame, window: str, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    if window not in WINDOW_DAYS:
        raise ValueError(f"window must be one of {list(WINDOW_DAYS)}")
    days = WINDOW_DAYS[window]
    if days is None or "created" not in raw.columns:
        return raw.copy()
    frame = raw.copy()
    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created"])
    end = as_of or frame["created"].max()
    start = end - pd.Timedelta(days=days)
    return frame[(frame["created"] >= start) & (frame["created"] <= end)].copy()


def available_validation_days(raw: pd.DataFrame, train_window: str) -> list[str]:
    """Days that fall outside the training window (or all days if window=all — use last 20%)."""
    if "created" not in raw.columns:
        return []
    frame = raw.copy()
    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["created"])
    if frame.empty:
        return []
    end = frame["created"].max()
    days = WINDOW_DAYS.get(train_window)
    if days is None:
        # Suggest days from last 20% chronologically
        cutoff = frame["created"].quantile(0.8)
        hold = frame[frame["created"] >= cutoff]
    else:
        start = end - pd.Timedelta(days=days)
        hold = frame[frame["created"] < start]
        if hold.empty:
            # not enough history — use last calendar day as validation candidate
            hold = frame[frame["created"].dt.normalize() == end.normalize()]
    days_list = sorted({d.date().isoformat() for d in hold["created"]})
    return days_list[-30:]


def run_windowed_train(
    *,
    cfg: dict[str, Any],
    tickets: pd.DataFrame,
    window: str,
    params: dict[str, Any],
    job: TrainJobState,
) -> dict[str, Any]:
    cfg = apply_train_params(cfg, params)
    job.log(f"Start training window={window} params={json.dumps(params)}")
    report = require_attestation(cfg)
    paths = cfg["paths"]
    for key in ("processed_dir", "model_dir", "metrics_dir", "reports_dir", "sealed_dir"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    dump_report(report, str(Path(paths["reports_dir"]) / "attestation.json"))

    raw = filter_by_window(tickets, window)
    job.log(f"Rows in window: {len(raw)} / {len(tickets)}")
    if raw.empty:
        raise ValueError("No tickets in selected training window")

    frame = build_training_frame(raw, cfg)
    job.log(f"Usable training rows after preprocess: {len(frame)}")
    frame.to_csv(Path(paths["processed_dir"]) / "train_frame.csv", index=False)

    texts = frame["text"].tolist()
    y_labels = frame["labels"].tolist()
    y_area = frame["primary_area"].tolist()
    has_labels = [bool(labs) for labs in y_labels]
    has_area = [bool(a) for a in y_area]
    rs = int(cfg["data"].get("random_state", 42))
    test_size = float(cfg["data"].get("test_size", 0.2))
    idx = np.arange(len(frame))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=rs)
    job.log(f"Split train={len(train_idx)} holdout={len(test_idx)}")

    job.log("Fitting TF-IDF features…")
    vectorizer = build_text_vectorizer(cfg)
    X_train = vectorizer.fit_transform([texts[i] for i in train_idx])
    X_test = vectorizer.transform([texts[i] for i in test_idx])

    metrics: dict[str, Any] = {"labels": {}, "area": {}, "counts": {}}
    label_train_mask = [has_labels[i] for i in train_idx]
    label_test_mask = [has_labels[i] for i in test_idx]
    mlb = fit_label_binarizer([y_labels[i] for i in train_idx if has_labels[i]])
    labels_estimator = build_labels_estimator(cfg)
    metrics["counts"] = {
        "rows": int(len(frame)),
        "train": int(len(train_idx)),
        "test": int(len(test_idx)),
        "label_classes": int(len(mlb.classes_)),
        "window": window,
    }

    job.log(f"Training labels model ({len(mlb.classes_)} classes)…")
    if len(mlb.classes_) == 0 or not any(label_train_mask):
        from sklearn.dummy import DummyClassifier
        from sklearn.multiclass import OneVsRestClassifier

        labels_estimator = OneVsRestClassifier(DummyClassifier(strategy="prior"))
        mlb.fit([["__none__"]])
        Y_dummy = mlb.transform([["__none__"]] * max(1, sum(label_train_mask) or 1))
        labels_estimator.fit(X_train[: Y_dummy.shape[0]], Y_dummy)
        metrics["labels"] = {"f1_macro": 0.0, "note": "insufficient label data"}
    else:
        Y_train = mlb.transform([y_labels[i] for i in train_idx if has_labels[i]])
        labels_estimator.fit(X_train[np.array(label_train_mask)], Y_train)
        if any(label_test_mask):
            Y_test = mlb.transform([y_labels[i] for i in test_idx if has_labels[i]])
            thr = float(cfg["models"]["labels"].get("decision_threshold", 0.35))
            Y_hat = _threshold_predict(labels_estimator, X_test[np.array(label_test_mask)], thr)
            if Y_hat.shape != Y_test.shape:
                Y_hat = labels_estimator.predict(X_test[np.array(label_test_mask)])
            metrics["labels"] = evaluate_labels(Y_test, Y_hat, list(mlb.classes_))
        else:
            metrics["labels"] = {"f1_macro": 0.0, "note": "no labeled test rows"}
    job.log(
        f"Labels F1 macro={metrics['labels'].get('f1_macro')} micro={metrics['labels'].get('f1_micro')}"
    )

    area_train_idx = [i for i in train_idx if has_area[i]]
    area_test_idx = [i for i in test_idx if has_area[i]]
    area_encoder = fit_area_encoder([y_area[i] for i in area_train_idx])
    area_estimator = build_area_estimator(cfg)
    metrics["counts"]["area_classes"] = int(len(getattr(area_encoder, "classes_", [])))
    job.log(f"Training impacted-area model ({metrics['counts']['area_classes']} classes)…")
    if len(getattr(area_encoder, "classes_", [])) < 2 or not area_train_idx:
        from sklearn.dummy import DummyClassifier

        area_estimator = DummyClassifier(strategy="most_frequent")
        if not area_train_idx:
            area_encoder.fit(["__none__"])
            area_estimator.fit(X_train[:1], [0])
        else:
            pos = {i: n for n, i in enumerate(train_idx)}
            y_enc = area_encoder.transform([y_area[i] for i in area_train_idx])
            area_estimator.fit(X_train[[pos[i] for i in area_train_idx]], y_enc)
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
    job.log(
        f"Area accuracy={metrics['area'].get('accuracy')} F1 macro={metrics['area'].get('f1_macro')}"
    )

    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    store = ModelStore(paths["model_dir"], paths.get("sealed_dir"))
    promote = None
    if cfg.get("training", {}).get("promote_if_better", True) and passes_promotion_gates(metrics, cfg):
        promote = cfg.get("mlops", {}).get("registry_name", "production")
    elif not (Path(paths["model_dir"]) / "production").exists():
        promote = "production"
    job.log(f"Saving model version={version} promote={promote}")
    manifest = store.save_bundle(
        version=version,
        vectorizer=vectorizer,
        labels_estimator=labels_estimator,
        labels_mlb=mlb,
        area_estimator=area_estimator,
        area_encoder=area_encoder,
        metrics=metrics,
        config=cfg,
        seal=bool(cfg.get("enclave", {}).get("seal_models", False)),
        key_path=cfg.get("enclave", {}).get("local_key_path"),
        promote_as=promote,
    )
    (Path(paths["metrics_dir"]) / f"metrics_{version}.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    result = {
        "version": version,
        "promoted": promote,
        "metrics": metrics,
        "manifest": manifest.to_dict(),
        "window": window,
        "params": params,
        "chart": {
            "labels": ["Labels F1 macro", "Labels F1 micro", "Area accuracy", "Area F1 macro"],
            "values": [
                float(metrics.get("labels", {}).get("f1_macro") or 0),
                float(metrics.get("labels", {}).get("f1_micro") or 0),
                float(metrics.get("area", {}).get("accuracy") or 0),
                float(metrics.get("area", {}).get("f1_macro") or 0),
            ],
        },
    }
    job.log("Training complete.")
    job._write(
        {
            "state": "completed",
            "started_at": job.status().get("started_at"),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error": None,
            "result": result,
        }
    )
    return result


def validate_on_day(
    *,
    cfg: dict[str, Any],
    tickets: pd.DataFrame,
    day: str,
    categorizer: Categorizer | None = None,
) -> dict[str, Any]:
    """Score production model against tickets created on a calendar day with known labels/areas."""
    frame = tickets.copy()
    if "created" not in frame.columns:
        raise ValueError("created timestamps required for day validation")
    frame["created"] = pd.to_datetime(frame["created"], utc=True, errors="coerce")
    day_ts = pd.Timestamp(day, tz="UTC")
    day_rows = frame[frame["created"].dt.normalize() == day_ts.normalize()].copy()
    if day_rows.empty:
        raise ValueError(f"No tickets on {day}")

    model = categorizer or Categorizer.load(cfg, version="production")
    texts = []
    true_labels = []
    true_areas = []
    keys = []
    for _, row in day_rows.iterrows():
        texts.append(f"{row.get('summary', '')} {row.get('description', '')}".strip())
        true_labels.append(split_multi_value(row.get("labels")))
        areas = split_multi_value(row.get("impacted_area"))
        true_areas.append(areas[0] if areas else "")
        keys.append(str(row.get("issue_key")))

    preds = model.predict_texts(texts)
    pred_labels = [p.get("labels") or [] for p in preds]
    pred_areas = [p.get("impacted_area") or "" for p in preds]

    # Multi-label metrics on union of classes seen that day
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(true_labels)
    y_pred = mlb.transform(pred_labels)
    label_metrics = evaluate_labels(y_true, y_pred, list(mlb.classes_)) if len(mlb.classes_) else {
        "f1_macro": 0.0,
        "f1_micro": 0.0,
    }

    area_pairs = [(t, p) for t, p in zip(true_areas, pred_areas) if t]
    if area_pairs:
        from sklearn.preprocessing import LabelEncoder

        enc = LabelEncoder()
        yt = enc.fit_transform([t for t, _ in area_pairs])
        # map preds; unknown -> -1 then filtered
        yp = []
        yt2 = []
        for (t, p), yti in zip(area_pairs, yt):
            if p in enc.classes_:
                yp.append(enc.transform([p])[0])
                yt2.append(yti)
        if yp:
            area_metrics = evaluate_area(np.array(yt2), np.array(yp), list(enc.classes_))
        else:
            area_metrics = {"accuracy": 0.0, "f1_macro": 0.0}
        area_acc_exact = float(np.mean([t == p for t, p in area_pairs]))
        area_metrics["exact_match_accuracy"] = area_acc_exact
    else:
        area_metrics = {"accuracy": 0.0, "f1_macro": 0.0, "exact_match_accuracy": 0.0}

    rows = []
    for key, tl, pl, ta, pa, pred in zip(keys, true_labels, pred_labels, true_areas, pred_areas, preds):
        rows.append(
            {
                "issue_key": key,
                "true_labels": tl,
                "pred_labels": pl,
                "label_hit": bool(set(tl) & set(pl)) if tl else None,
                "true_area": ta,
                "pred_area": pa,
                "area_hit": (ta == pa) if ta else None,
                "novelty": pred.get("novelty"),
            }
        )

    return {
        "day": day,
        "ticket_count": len(day_rows),
        "label_metrics": label_metrics,
        "area_metrics": area_metrics,
        "chart": {
            "labels": ["Label F1 macro", "Label F1 micro", "Area exact match", "Area F1 macro"],
            "values": [
                float(label_metrics.get("f1_macro") or 0),
                float(label_metrics.get("f1_micro") or 0),
                float(area_metrics.get("exact_match_accuracy") or 0),
                float(area_metrics.get("f1_macro") or 0),
            ],
        },
        "rows": rows[:50],
        "headline": (
            f"Validation {day}: labels F1µ={float(label_metrics.get('f1_micro') or 0):.2f}, "
            f"area match={float(area_metrics.get('exact_match_accuracy') or 0):.0%} "
            f"on {len(day_rows)} tickets"
        ),
    }


def current_params(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "C": float(cfg["models"]["labels"].get("C", 2.0)),
        "decision_threshold": float(cfg["models"]["labels"].get("decision_threshold", 0.35)),
        "max_word_features": int(cfg["features"].get("max_word_features", 15000)),
        "max_char_features": int(cfg["features"].get("max_char_features", 10000)),
        "min_df": int(cfg["features"].get("min_df", 1)),
        "test_size": float(cfg["data"].get("test_size", 0.2)),
        "min_label_support": int(cfg["data"].get("min_label_support", 2)),
        "min_area_support": int(cfg["data"].get("min_area_support", 2)),
        "algorithm": cfg["models"]["labels"].get("algorithm", "logistic_regression"),
    }
