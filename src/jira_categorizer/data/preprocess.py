"""Text cleaning for summary/description (no LLM, deterministic)."""

from __future__ import annotations

import html
import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_JIRA_KEY_RE = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")
_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def clean_text(text: str, *, strip_html_tags: bool = True) -> str:
    if not text:
        return ""
    text = str(text)
    if strip_html_tags:
        text = strip_html(text)
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    # Keep Jira keys as tokens but normalize spacing around them
    text = _JIRA_KEY_RE.sub(lambda m: f" {m.group(0)} ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def combine_summary_description(
    summary: str,
    description: str,
    *,
    strip_html_tags: bool = True,
) -> str:
    s = clean_text(summary, strip_html_tags=strip_html_tags)
    d = clean_text(description, strip_html_tags=strip_html_tags)
    if s and d:
        return f"{s} {d}"
    return s or d
