"""CLI for surge detection and new-label suggestions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jira_categorizer.config import load_config
from jira_categorizer.data.ingest import load_jira_export
from jira_categorizer.infer.predict import Categorizer
from jira_categorizer.monitor.new_labels import suggest_new_labels, write_new_label_report
from jira_categorizer.monitor.surge import detect_category_surges, write_surge_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Jira category surge + new label monitor")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default=None, help="Ticket CSV/JSON path")
    parser.add_argument(
        "--mode",
        choices=("surge", "new-labels", "all"),
        default="all",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    data_path = args.data or cfg["paths"]["raw_data"]
    tickets = load_jira_export(data_path)
    reports = Path(cfg["paths"]["reports_dir"])
    reports.mkdir(parents=True, exist_ok=True)

    out: dict = {}
    if args.mode in {"surge", "all"}:
        surge = detect_category_surges(
            tickets,
            cfg=cfg,
            label_field=cfg["data"]["label_field"],
            area_field=cfg["data"]["area_field"],
        )
        write_surge_report(surge, reports / "category_surges.json")
        out["surge"] = {
            "status": surge.get("status"),
            "surge_count": surge.get("surge_count", 0),
            "top": (surge.get("surges") or [])[:5],
        }

    if args.mode in {"new-labels", "all"}:
        model = Categorizer.load(cfg, version="production")
        suggestions = suggest_new_labels(tickets, model, cfg=cfg)
        write_new_label_report(suggestions, reports / "new_label_suggestions.json")
        out["new_labels"] = {
            "status": suggestions.get("status"),
            "novel_ticket_count": suggestions.get("novel_ticket_count"),
            "suggestions": [
                {
                    "suggested_label": s["suggested_label"],
                    "cluster_size": s["cluster_size"],
                    "top_terms": s["top_terms"],
                }
                for s in (suggestions.get("suggestions") or [])
            ],
        }

    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
