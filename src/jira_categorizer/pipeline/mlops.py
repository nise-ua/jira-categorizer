"""End-to-end MLOps loop: ingest → train → gate → register → monitor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jira_categorizer.config import load_config
from jira_categorizer.data.ingest import build_training_frame, load_jira_export
from jira_categorizer.infer.predict import Categorizer
from jira_categorizer.monitor.new_labels import suggest_new_labels, write_new_label_report
from jira_categorizer.monitor.surge import detect_category_surges, write_surge_report
from jira_categorizer.train.train import train


def check_simple_drift(cfg: dict[str, Any], data_path: str | None = None) -> dict[str, Any]:
    """Lightweight drift signal: mean cleaned-text length shift vs training frame."""
    if not cfg.get("mlops", {}).get("drift_check", {}).get("enabled", True):
        return {"enabled": False}

    paths = cfg["paths"]
    train_csv = Path(paths["processed_dir"]) / "train_frame.csv"
    if not train_csv.exists():
        return {"enabled": True, "status": "skip", "reason": "no processed training frame"}

    import pandas as pd

    hist = pd.read_csv(train_csv)
    hist_len = hist["text"].fillna("").astype(str).str.len().mean()

    raw = load_jira_export(data_path or paths["raw_data"])
    current = build_training_frame(raw, cfg)
    cur_len = current["text"].str.len().mean()
    if hist_len <= 0:
        shift = 0.0
    else:
        shift = abs(cur_len - hist_len) / hist_len
    max_shift = float(cfg["mlops"]["drift_check"].get("max_mean_text_len_shift", 0.35))
    return {
        "enabled": True,
        "status": "drift" if shift > max_shift else "ok",
        "hist_mean_len": float(hist_len),
        "current_mean_len": float(cur_len),
        "relative_shift": float(shift),
        "threshold": max_shift,
    }


def smoke_infer(cfg: dict[str, Any]) -> dict[str, Any]:
    model = Categorizer.load(cfg, version="production")
    sample = model.predict_one(
        "Payment API timeout in checkout",
        "Customers see 504 when submitting card. Affects EU region only.",
    )
    return sample


def run_monitoring(cfg: dict[str, Any], data_path: str | None = None) -> dict[str, Any]:
    tickets = load_jira_export(data_path or cfg["paths"]["raw_data"])
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    surge = detect_category_surges(
        tickets,
        cfg=cfg,
        label_field=cfg["data"]["label_field"],
        area_field=cfg["data"]["area_field"],
    )
    write_surge_report(surge, reports / "category_surges.json")

    model = Categorizer.load(cfg, version="production")
    new_labels = suggest_new_labels(tickets, model, cfg=cfg)
    write_new_label_report(new_labels, reports / "new_label_suggestions.json")

    return {"surge": surge, "new_labels": new_labels}


def run_pipeline(cfg: dict[str, Any], data_path: str | None = None) -> dict[str, Any]:
    train_result = train(cfg, data_path=data_path)
    drift = check_simple_drift(cfg, data_path=data_path)
    smoke = smoke_infer(cfg)
    monitoring = run_monitoring(cfg, data_path=data_path)
    result = {
        "train": {
            "version": train_result["version"],
            "promoted": train_result["promoted"],
            "metrics": train_result["metrics"],
        },
        "drift": drift,
        "smoke_prediction": smoke,
        "monitoring": {
            "surge_status": monitoring["surge"].get("status"),
            "surge_count": monitoring["surge"].get("surge_count", 0),
            "top_surges": (monitoring["surge"].get("surges") or [])[:5],
            "new_label_status": monitoring["new_labels"].get("status"),
            "novel_ticket_count": monitoring["new_labels"].get("novel_ticket_count"),
            "suggested_labels": [
                {
                    "suggested_label": s["suggested_label"],
                    "cluster_size": s["cluster_size"],
                    "top_terms": s["top_terms"],
                }
                for s in (monitoring["new_labels"].get("suggestions") or [])
            ],
        },
    }
    out = Path(cfg["paths"]["reports_dir"]) / "mlops_last_run.json"
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Jira categorizer MLOps pipeline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    result = run_pipeline(cfg, data_path=args.data)
    print(
        json.dumps(
            {
                "version": result["train"]["version"],
                "promoted": result["train"]["promoted"],
                "labels_f1_macro": result["train"]["metrics"].get("labels", {}).get("f1_macro"),
                "area_f1_macro": result["train"]["metrics"].get("area", {}).get("f1_macro"),
                "drift": result["drift"].get("status"),
                "surge_status": result["monitoring"].get("surge_status"),
                "surge_count": result["monitoring"].get("surge_count"),
                "new_label_status": result["monitoring"].get("new_label_status"),
                "suggested_labels": result["monitoring"].get("suggested_labels"),
                "smoke_labels": result["smoke_prediction"].get("labels"),
                "smoke_area": result["smoke_prediction"].get("impacted_area"),
                "smoke_novelty": result["smoke_prediction"].get("novelty"),
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
