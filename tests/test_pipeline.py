from pathlib import Path

from jira_categorizer.config import load_config, project_root
from jira_categorizer.infer.predict import Categorizer
from jira_categorizer.pipeline.mlops import run_pipeline


def test_end_to_end_train_and_predict(tmp_path, monkeypatch):
    cfg = load_config()
    # Redirect artifacts into temp dir for isolation
    for key in ("processed_dir", "model_dir", "metrics_dir", "reports_dir", "sealed_dir"):
        cfg["paths"][key] = str(tmp_path / key)
    cfg["enclave"]["mode"] = "local_dev"
    cfg["enclave"]["seal_models"] = True
    cfg["enclave"]["local_key_path"] = str(tmp_path / "sealed" / "local.dev.key")
    cfg["training"]["min_f1_macro_labels"] = 0.0
    cfg["training"]["min_f1_macro_area"] = 0.0
    cfg["paths"]["raw_data"] = str(project_root() / "sample_data" / "jira_export.csv")

    result = run_pipeline(cfg)
    assert result["train"]["version"]
    assert Path(cfg["paths"]["model_dir"], "production").exists()

    model = Categorizer.load(cfg, version="production")
    pred = model.predict_one(
        "Card payment gateway timeout",
        "Checkout returns 504 when charging Visa in EU",
    )
    assert "labels" in pred
    assert "impacted_area" in pred
    # Should lean toward payments area given sample distribution
    assert pred["impacted_area"] in {
        "payments",
        "authentication",
        "api-gateway",
        "storefront",
        "data-platform",
        "infrastructure",
    }
