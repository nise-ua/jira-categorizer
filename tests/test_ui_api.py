from fastapi.testclient import TestClient

from jira_categorizer.config import load_config
from jira_categorizer.enclave import server as server_mod
from jira_categorizer.pipeline.mlops import run_pipeline


def _boot_kafka_app(tmp_path):
    cfg = load_config("config/kafka_public.yaml")
    for key in (
        "processed_dir",
        "model_dir",
        "metrics_dir",
        "reports_dir",
        "sealed_dir",
        "cache_dir",
        "feedback_dir",
    ):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["enclave"]["mode"] = "local_dev"
    cfg["enclave"]["seal_models"] = False
    cfg["enclave"]["local_key_path"] = str(tmp_path / "sealed" / "local.dev.key")
    cfg["training"]["min_f1_macro_labels"] = 0.0
    cfg["training"]["min_f1_macro_area"] = 0.0
    cfg["monitoring"]["surge"]["enabled"] = True
    cfg["monitoring"]["new_labels"]["min_cluster_size"] = 2
    run_pipeline(cfg)
    server_mod._STATE.clear()
    server_mod._STATE["cfg"] = cfg
    return TestClient(server_mod.app)


def test_timeseries_and_triage_and_feedback(tmp_path):
    client = _boot_kafka_app(tmp_path)
    meta = client.get("/api/meta")
    assert meta.status_code == 200
    assert meta.json()["rows"] >= 50

    ts = client.get("/api/timeseries?dimension=labels&window=3m")
    assert ts.status_code == 200
    body = ts.json()
    assert "labels" in body and "datasets" in body

    nov = client.get("/api/timeseries/novelty?window=3m")
    assert nov.status_code == 200
    assert len(nov.json()["datasets"]) == 2

    triage = client.get("/api/triage?days=90&limit=10")
    assert triage.status_code == 200
    tbody = triage.json()
    assert tbody["count"] >= 1
    item = tbody["items"][0]

    save = client.post(
        "/api/triage/save",
        json={
            "issue_key": item["issue_key"],
            "summary": item["summary"],
            "description": item["description"],
            "created": item["created"],
            "selected_labels": ["human-corrected-label"],
            "selected_areas": [item.get("suggested_area") or "streams"],
            "suggested_labels": item.get("suggested_labels") or [],
            "suggested_area": item.get("suggested_area"),
            "push_to_jira": True,
        },
    )
    assert save.status_code == 200
    saved = save.json()
    assert saved["ok"] is True
    assert saved["disagreed"] is True
    assert saved["jira"]["dry_run"] is True

    retrain = client.post("/api/retrain-with-feedback")
    assert retrain.status_code == 200
    assert retrain.json()["merge"]["merged"] >= 1
