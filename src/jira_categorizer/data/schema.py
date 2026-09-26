"""Ticket schema and target parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


LABEL_SEPARATORS = ("|", ";", ",")


@dataclass
class TicketRecord:
    issue_key: str
    summary: str
    description: str
    labels: list[str] = field(default_factory=list)
    impacted_areas: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        summary = (self.summary or "").strip()
        description = (self.description or "").strip()
        if summary and description:
            return f"{summary}\n\n{description}"
        return summary or description


def split_multi_value(raw: str | list | None) -> list[str]:
    """Parse Jira multi-value fields (labels / impacted area)."""
    if raw is None:
        return []
    if isinstance(raw, list):
        values = [str(v).strip() for v in raw]
    else:
        text = str(raw).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        # Prefer pipe, then semicolon, then comma when it looks like tags
        sep = None
        for candidate in LABEL_SEPARATORS:
            if candidate in text:
                sep = candidate
                break
        values = [p.strip() for p in text.split(sep)] if sep else [text]
    # Normalize: lowercase, collapse spaces, drop empties
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        norm = " ".join(v.lower().split())
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out
