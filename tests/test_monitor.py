from datetime import datetime, timedelta, timezone

import pandas as pd

from jira_categorizer.config import load_config, project_root
from jira_categorizer.infer.predict import Categorizer
from jira_categorizer.monitor.new_labels import suggest_new_labels
from jira_categorizer.monitor.novelty import score_novelty
from jira_categorizer.monitor.surge import detect_category_surges
from jira_categorizer.pipeline.mlops import run_pipeline


def test_score_novelty_flags_weak_predictions():
    preds = [
        {
            "labels": ["payments"],
            "labels_scored": [{"label": "payments", "score": 0.9}],
            "impacted_area": "payments",
            "impacted_area_scored": [{"area": "payments", "score": 0.8}],
        },
        {
            "labels": [],
            "labels_scored": [{"label": "auth", "score": 0.1}],
            "impacted_area": "api-gateway",
            "impacted_area_scored": [{"area": "api-gateway", "score": 0.2}],
        },
    ]
    out = score_novelty(preds, label_conf_threshold=0.35, area_conf_threshold=0.4)
    assert out[0]["novelty"]["is_novel"] is False
    assert out[1]["novelty"]["is_novel"] is True


def test_surge_detects_spike():
    now = datetime(2026, 9, 1, tzinfo=timezone.utc)
    rows = []
    # Baseline: occasional "payments"
    for i in range(10):
        rows.append(
            {
                "issue_key": f"B-{i}",
                "summary": "baseline",
                "description": "normal",
                "labels": "ops",
                "impacted_area": "payments",
                "created": (now - timedelta(days=40 + i)).isoformat(),
            }
        )
    # Recent surge on "checkout-latency"
    for i in range(8):
        rows.append(
            {
                "issue_key": f"S-{i}",
                "summary": "latency",
                "description": "p99 up",
                "labels": "checkout-latency",
                "impacted_area": "payments",
                "created": (now - timedelta(days=i)).isoformat(),
            }
        )
    df = pd.DataFrame(rows)
    cfg = {
        "monitoring": {
            "surge": {
                "enabled": True,
                "recent_days": 7,
                "baseline_days": 28,
                "min_recent_count": 3,
                "min_rate_ratio": 2.0,
                "min_z_score": 1.5,
                "epsilon": 0.5,
            }
        }
    }
    result = detect_category_surges(df, cfg=cfg, as_of=now)
    assert result["status"] == "alert"
    cats = {s["category"] for s in result["surges"]}
    assert "checkout-latency" in cats


def test_new_label_suggestions_on_trained_model(tmp_path):
    cfg = load_config()
    for key in ("processed_dir", "model_dir", "metrics_dir", "reports_dir", "sealed_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["enclave"]["mode"] = "local_dev"
    cfg["enclave"]["seal_models"] = False
    cfg["enclave"]["local_key_path"] = str(tmp_path / "sealed" / "local.dev.key")
    cfg["training"]["min_f1_macro_labels"] = 0.0
    cfg["training"]["min_f1_macro_area"] = 0.0
    cfg["paths"]["raw_data"] = str(project_root() / "sample_data" / "jira_export.csv")
    cfg["monitoring"] = {
        "surge": {"enabled": False},
        "new_labels": {
            "enabled": True,
            "label_confidence_threshold": 0.99,  # force novelty
            "area_confidence_threshold": 0.99,
            "min_cluster_size": 3,
            "max_clusters": 4,
            "max_suggestions": 3,
            "top_terms": 4,
        },
    }
    run_pipeline(cfg)
    model = Categorizer.load(cfg, version="production")
    # Synthetic novel tickets about a brand-new theme
    tickets = pd.DataFrame(
        [
            {
                "issue_key": f"N-{i}",
                "summary": f"Quantum ledger sync failure shard {i}",
                "description": "Entanglement queue overflow in quantum ledger bridge",
                "labels": "",
                "impacted_area": "",
            }
            for i in range(9)
        ]
    )
    result = suggest_new_labels(tickets, model, cfg=cfg)
    assert result["novel_ticket_count"] >= 3
    assert result["suggestions"]
    assert result["suggestions"][0]["suggested_label"]
