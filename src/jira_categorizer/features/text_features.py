"""Classical text feature pipeline: word + char TF-IDF (no embeddings / no LLM)."""

from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_text_vectorizer(cfg: dict[str, Any]) -> FeatureUnion:
    feat = cfg["features"]
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=tuple(feat["word_ngram_range"]),
        max_features=int(feat["max_word_features"]),
        min_df=int(feat["min_df"]),
        lowercase=bool(feat.get("lowercase", True)),
        sublinear_tf=True,
        strip_accents="unicode",
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=tuple(feat["char_ngram_range"]),
        max_features=int(feat["max_char_features"]),
        min_df=int(feat["min_df"]),
        lowercase=bool(feat.get("lowercase", True)),
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
