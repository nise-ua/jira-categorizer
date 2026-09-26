"""Suggest new label categories when novel issue types appear.

Approach (no LLM):
1. Score tickets with the current categorizer.
2. Mark as novel when max label confidence and/or area confidence is low,
   or when predicted labels are empty/weak.
3. Cluster novel ticket texts (TF-IDF + MiniBatchKMeans).
4. Propose a slug from top TF-IDF terms + exemplar summaries.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from jira_categorizer.data.preprocess import combine_summary_description
from jira_categorizer.monitor.novelty import score_novelty

# Avoid circular import with infer.predict; type as Any at boundary.

_SLUG_RE = re.compile(r"[^a-z0-9]+")
_NOISE_TERMS = {
    "description",
    "summary",
    "issue",
    "jira",
    "ticket",
    "bug",
    "fix",
    "http",
    "https",
    "www",
    "com",
    "org",
    "null",
    "none",
    "test",
    "tests",
}


def _slugify_terms(terms: list[str], max_parts: int = 3) -> str:
    parts: list[str] = []
    for t in terms:
        tok = _SLUG_RE.sub("-", t.lower()).strip("-")
        if not tok or tok in parts:
            continue
        if len(tok) < 3 or tok.isdigit() or tok in _NOISE_TERMS:
            continue
        # drop pure version-like tokens (e.g. kip-1331 kept; bare 1331 dropped above)
        if all(p.isdigit() for p in tok.split("-")):
            continue
        parts.append(tok)
        if len(parts) >= max_parts:
            break
    return "-".join(parts) if parts else "new-topic"


def suggest_new_labels(
    tickets: pd.DataFrame,
    categorizer: Any,
    *,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Cluster novel tickets and suggest new label categories."""
    mon = cfg.get("monitoring", {}).get("new_labels", {})
    if not mon.get("enabled", True):
        return {"enabled": False, "suggestions": []}

    label_thr = float(mon.get("label_confidence_threshold", 0.35))
    area_thr = float(mon.get("area_confidence_threshold", 0.40))
    min_cluster = int(mon.get("min_cluster_size", 3))
    max_clusters = int(mon.get("max_clusters", 8))
    max_suggestions = int(mon.get("max_suggestions", 5))
    top_terms = int(mon.get("top_terms", 5))
    random_state = int(cfg.get("data", {}).get("random_state", 42))

    if tickets.empty:
        return {"enabled": True, "status": "skip", "reason": "no tickets", "suggestions": []}

    strip_html = bool(cfg.get("features", {}).get("strip_html", True))
    texts: list[str] = []
    keys: list[str] = []
    summaries: list[str] = []
    for _, row in tickets.iterrows():
        text = combine_summary_description(
            str(row.get("summary", "")),
            str(row.get("description", "")),
            strip_html_tags=strip_html,
        )
        if not text:
            continue
        texts.append(text)
        keys.append(str(row.get("issue_key", "")))
        summaries.append(str(row.get("summary", ""))[:160])

    if not texts:
        return {"enabled": True, "status": "skip", "reason": "no usable text", "suggestions": []}

    preds = categorizer.predict_texts(texts)
    scored = score_novelty(preds, label_conf_threshold=label_thr, area_conf_threshold=area_thr)

    novel_idx = [i for i, p in enumerate(scored) if p["novelty"]["is_novel"]]
    novel_rate = len(novel_idx) / max(len(texts), 1)

    if len(novel_idx) < min_cluster:
        return {
            "enabled": True,
            "status": "ok",
            "novel_ticket_count": len(novel_idx),
            "novel_rate": round(novel_rate, 4),
            "thresholds": {
                "label_confidence_threshold": label_thr,
                "area_confidence_threshold": area_thr,
                "min_cluster_size": min_cluster,
            },
            "suggestions": [],
            "note": "Not enough novel tickets to propose a new label category",
        }

    novel_texts = [texts[i] for i in novel_idx]
    # Dedicated short-word TF-IDF for interpretable term suggestions
    vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=5000,
        stop_words="english",
        lowercase=True,
    )
    X = vec.fit_transform(novel_texts)
    n_clusters = int(min(max_clusters, max(1, len(novel_idx) // min_cluster)))
    n_clusters = max(1, min(n_clusters, len(novel_idx)))

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        batch_size=min(256, max(len(novel_idx), 1)),
    )
    labels = km.fit_predict(X)
    feature_names = np.asarray(vec.get_feature_names_out())

    suggestions: list[dict[str, Any]] = []
    for cluster_id in range(n_clusters):
        members = np.where(labels == cluster_id)[0]
        if len(members) < min_cluster:
            continue
        # Top terms from cluster centroid
        center = km.cluster_centers_[cluster_id]
        top_idx = np.argsort(center)[::-1][:top_terms]
        terms = [str(feature_names[j]) for j in top_idx if center[j] > 0]
        suggested = _slugify_terms(terms)
        # Exemplars: closest to centroid
        member_X = X[members]
        # densify small rows for distance
        dists = np.linalg.norm(member_X.toarray() - center, axis=1)
        order = np.argsort(dists)[:3]
        exemplars = []
        for local_i in order:
            global_i = novel_idx[int(members[local_i])]
            exemplars.append(
                {
                    "issue_key": keys[global_i],
                    "summary": summaries[global_i],
                    "label_confidence": scored[global_i]["novelty"]["label_confidence"],
                    "area_confidence": scored[global_i]["novelty"]["area_confidence"],
                }
            )
        avg_label_conf = float(
            np.mean([scored[novel_idx[int(m)]]["novelty"]["label_confidence"] for m in members])
        )
        suggestions.append(
            {
                "suggested_label": suggested,
                "top_terms": terms,
                "cluster_size": int(len(members)),
                "avg_label_confidence": round(avg_label_conf, 4),
                "confidence": "high"
                if len(members) >= min_cluster * 2 and avg_label_conf < label_thr * 0.8
                else "medium",
                "exemplars": exemplars,
                "rationale": (
                    f"{len(members)} recent tickets cluster around terms {terms[:3]}; "
                    "existing label/area models assign low confidence — consider adding this label."
                ),
            }
        )

    suggestions.sort(key=lambda s: (s["cluster_size"], -s["avg_label_confidence"]), reverse=True)
    suggestions = suggestions[:max_suggestions]

    return {
        "enabled": True,
        "status": "suggestions" if suggestions else "ok",
        "novel_ticket_count": len(novel_idx),
        "novel_rate": round(novel_rate, 4),
        "n_clusters_fit": n_clusters,
        "thresholds": {
            "label_confidence_threshold": label_thr,
            "area_confidence_threshold": area_thr,
            "min_cluster_size": min_cluster,
        },
        "suggestions": suggestions,
    }


def write_new_label_report(result: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return path
