#!/usr/bin/env python3
"""Fetch public Apache Jira issues for offline training (no auth)."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path


BASE = "https://issues.apache.org/jira/rest/api/2/search"
DEFAULT_JQL = (
    "project = KAFKA AND labels is not EMPTY AND component is not EMPTY "
    "ORDER BY created DESC"
)
FIELDS = "key,summary,description,labels,components,issuetype,status,created,updated"


def fetch_page(jql: str, start_at: int, max_results: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "jql": jql,
            "startAt": start_at,
            "maxResults": max_results,
            "fields": FIELDS,
        }
    )
    req = urllib.request.Request(
        f"{BASE}?{params}",
        headers={"Accept": "application/json", "User-Agent": "jira-categorizer/0.1"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_row(issue: dict) -> dict:
    fields = issue.get("fields") or {}
    desc = fields.get("description") or ""
    if isinstance(desc, dict):
        desc = json.dumps(desc)
    comps = [c.get("name", "") for c in (fields.get("components") or []) if c.get("name")]
    labels = fields.get("labels") or []
    return {
        "issue_key": issue.get("key"),
        "summary": fields.get("summary") or "",
        "description": desc,
        "labels": "|".join(labels),
        "impacted_area": "|".join(comps),
        "issuetype": (fields.get("issuetype") or {}).get("name", ""),
        "status": (fields.get("status") or {}).get("name", ""),
        "created": fields.get("created") or "",
        "updated": fields.get("updated") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jql", default=DEFAULT_JQL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--out",
        default="sample_data/public_jira/kafka_100.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    issues: list[dict] = []
    start = 0
    while len(issues) < args.limit:
        batch = fetch_page(args.jql, start, min(50, args.limit - len(issues)))
        chunk = batch.get("issues") or []
        if not chunk:
            break
        issues.extend(chunk)
        start += len(chunk)
        if start >= batch.get("total", 0):
            break
    issues = issues[: args.limit]
    rows = [to_row(i) for i in issues]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lab_c: Counter[str] = Counter()
    area_c: Counter[str] = Counter()
    for r in rows:
        for x in r["labels"].split("|"):
            if x:
                lab_c[x.lower()] += 1
        for x in r["impacted_area"].split("|"):
            if x:
                area_c[x.lower()] += 1

    meta = out.with_suffix(".SOURCE.md")
    meta.write_text(
        "\n".join(
            [
                "# Public Jira dataset",
                "",
                f"- Source: Apache Jira ({BASE.rsplit('/rest', 1)[0]})",
                f"- JQL: `{args.jql}`",
                f"- Count: {len(rows)}",
                "- Mapping: components -> impacted_area; labels -> labels",
                "- Timestamps: created, updated (for surge detection)",
                "",
                "## Top labels",
                *[f"- {k}: {v}" for k, v in lab_c.most_common(15)],
                "",
                "## Top impacted areas (components)",
                *[f"- {k}: {v}" for k, v in area_c.most_common(15)],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out} ({len(rows)} rows)")
    print(f"Wrote {meta}")


if __name__ == "__main__":
    main()
