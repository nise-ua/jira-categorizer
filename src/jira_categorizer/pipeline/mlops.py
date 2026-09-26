"""End-to-end MLOps loop: ingest → train → gate → register → optional serve check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from jira_categorizer.config import load_config
from jira_categorizer.data.ingest import build_training_frame, load_jira_export
from jira_categorizer.infer.predict import Categorizer
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


def run_pipeline(cfg: dict[str, Any], data_path: str | None = None) -> dict[str, Any]:
    train_result = train(cfg, data_path=data_path)
    drift = check_simple_drift(cfg, data_path=data_path)
    smoke = smoke_infer(cfg)
    result = {
        "train": {
            "version": train_result["version"],
            "promoted": train_result["promoted"],
            "metrics": train_result["metrics"],
        },
        "drift": drift,
        "smoke_prediction": smoke,
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
    # Compact stdout
    print(
        json.dumps(
            {
                "version": result["train"]["version"],
                "promoted": result["train"]["promoted"],
                "labels_f1_macro": result["train"]["metrics"].get("labels", {}).get("f1_macro"),
                "area_f1_macro": result["train"]["metrics"].get("area", {}).get("f1_macro"),
                "drift": result["drift"].get("status"),
                "smoke_labels": result["smoke_prediction"].get("labels"),
                "smoke_area": result["smoke_prediction"].get("impacted_area"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
