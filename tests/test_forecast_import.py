from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from jira_categorizer.analytics.forecast import build_forecast_actual
from jira_categorizer.config import load_config, project_root
from jira_categorizer.data.excel_import import load_spreadsheet
from jira_categorizer.enclave import server as server_mod
from jira_categorizer.pipeline.mlops import run_pipeline


def test_forecast_has_today_tomorrow_and_summary():
    df = pd.read_csv(project_root() / "sample_data" / "public_jira" / "kafka_100.csv")
    out = build_forecast_actual(df, dimension="labels")
    assert len(out["labels"]) >= 24
    assert len(out["datasets"]) == 2
    assert out["datasets"][0]["label"] == "Forecast"
    assert out["datasets"][1]["label"] == "Actual"
    assert out["summary"]["headline"]
    assert out["summary"]["bullets"]


def test_excel_roundtrip(tmp_path):
    src = pd.read_csv(project_root() / "sample_data" / "public_jira" / "kafka_100.csv")
    xlsx = tmp_path / "jira_export.xlsx"
    src.to_excel(xlsx, index=False)
    loaded = load_spreadsheet(xlsx)
    assert len(loaded) == len(src)
    assert "summary" in loaded.columns


def test_import_and_forecast_api(tmp_path):
    cfg = load_config("config/kafka_public.yaml")
    for key in (
        "processed_dir",
        "model_dir",
        "metrics_dir",
        "reports_dir",
        "sealed_dir",
        "cache_dir",
        "feedback_dir",
        "uploads_dir",
    ):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["enclave"]["mode"] = "local_dev"
    cfg["enclave"]["seal_models"] = False
    cfg["enclave"]["local_key_path"] = str(tmp_path / "sealed" / "local.dev.key")
    cfg["training"]["min_f1_macro_labels"] = 0.0
    cfg["training"]["min_f1_macro_area"] = 0.0
    run_pipeline(cfg)
    server_mod._STATE.clear()
    server_mod._STATE["cfg"] = cfg
    client = TestClient(server_mod.app)

    xlsx = tmp_path / "upload.xlsx"
    pd.read_csv(project_root() / "sample_data" / "public_jira" / "kafka_100.csv").to_excel(xlsx, index=False)
    with xlsx.open("rb") as fh:
        resp = client.post(
            "/api/import/spreadsheet",
            files={"file": ("upload.xlsx", fh, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )
    assert resp.status_code == 200, resp.text
    assert resp.json()["source_mode"] == "spreadsheet"

    forecast = client.get("/api/forecast?dimension=labels")
    assert forecast.status_code == 200
    body = forecast.json()
    assert body["summary"]["status"] in {"surge", "normal", "below", "quiet", "unknown"}
    assert "Forecast" in [d["label"] for d in body["datasets"]]
