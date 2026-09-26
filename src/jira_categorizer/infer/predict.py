"""Batch and single-ticket inference for labels + impacted area."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jira_categorizer.config import load_config
from jira_categorizer.data.preprocess import combine_summary_description
from jira_categorizer.models.area_model import predict_areas
from jira_categorizer.models.labels_model import predict_labels
from jira_categorizer.monitor.novelty import score_novelty
from jira_categorizer.registry.model_store import ModelStore


class Categorizer:
    def __init__(self, bundle: dict[str, Any], cfg: dict[str, Any]):
        self.vectorizer = bundle["vectorizer"]
        self.labels_estimator = bundle["labels_estimator"]
        self.labels_mlb = bundle["labels_mlb"]
        self.area_estimator = bundle["area_estimator"]
        self.area_encoder = bundle["area_encoder"]
        self.cfg = cfg
        self.manifest = bundle.get("manifest", {})

    @classmethod
    def load(cls, cfg: dict[str, Any], version: str = "production") -> "Categorizer":
        store = ModelStore(cfg["paths"]["model_dir"], cfg["paths"].get("sealed_dir"))
        enclave = cfg.get("enclave", {})
        if enclave.get("seal_models") and enclave.get("mode") == "nitro":
            bundle = store.load_sealed(version, enclave["local_key_path"])
        else:
            try:
                bundle = store.load_bundle(version)
            except FileNotFoundError:
                # Fallback to sealed local bundle
                bundle = store.load_sealed(version, enclave["local_key_path"])
        return cls(bundle, cfg)

    def predict_one(self, summary: str, description: str = "") -> dict[str, Any]:
        text = combine_summary_description(
            summary,
            description,
            strip_html_tags=bool(self.cfg["features"].get("strip_html", True)),
        )
        return self.predict_texts([text])[0]

    def predict_texts(self, texts: list[str]) -> list[dict[str, Any]]:
        X = self.vectorizer.transform(texts)
        thr = float(self.cfg["models"]["labels"].get("decision_threshold", 0.35))
        top_k = int(self.cfg["inference"].get("top_k_labels", 5))
        label_preds = predict_labels(
            self.labels_estimator, self.labels_mlb, X, threshold=thr, top_k=top_k
        )
        area_preds = predict_areas(self.area_estimator, self.area_encoder, X, top_k=3)
        out: list[dict[str, Any]] = []
        for labs, areas in zip(label_preds, area_preds):
            out.append(
                {
                    "labels": [x["label"] for x in labs],
                    "labels_scored": labs,
                    "impacted_area": areas[0]["area"] if areas else None,
                    "impacted_area_scored": areas,
                }
            )
        # Flag tickets that look like a new issue type (low confidence)
        nl = self.cfg.get("monitoring", {}).get("new_labels", {})
        return score_novelty(
            out,
            label_conf_threshold=float(nl.get("label_confidence_threshold", 0.35)),
            area_conf_threshold=float(nl.get("area_confidence_threshold", 0.40)),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict Jira labels + impacted area")
    parser.add_argument("--config", default=None)
    parser.add_argument("--version", default="production")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--description", default="")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    model = Categorizer.load(cfg, version=args.version)
    result = model.predict_one(args.summary, args.description)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
