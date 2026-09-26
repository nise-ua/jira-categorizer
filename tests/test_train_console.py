import time

from fastapi.testclient import TestClient

from jira_categorizer.config import load_config
from jira_categorizer.enclave import server as server_mod
from jira_categorizer.pipeline.mlops import run_pipeline


def _client(tmp_path):
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
    return TestClient(server_mod.app)


def test_train_and_validate_loop(tmp_path):
    client = _client(tmp_path)
    params = client.get("/api/train/params").json()["params"]
    assert "C" in params

    start = client.post("/api/train/start?sync=true", json={"window": "1y", "params": params})
    assert start.status_code == 200
    st = start.json()
    assert st["state"] == "completed", st
    assert st["result"]["chart"]["values"]

    days = client.get("/api/validate/days?train_window=1y").json()["days"]
    # With 1y window on recent kafka slice, holdout may be empty — fall back to any day via all
    if not days:
        days = client.get("/api/validate/days?train_window=all").json()["days"]
    assert days
    val = client.post("/api/validate/day", json={"day": days[-1], "train_window": "1y"})
    assert val.status_code == 200, val.text
    body = val.json()
    assert body["ticket_count"] >= 1
    assert "chart" in body
