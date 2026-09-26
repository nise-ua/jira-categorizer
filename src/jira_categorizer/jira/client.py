"""Minimal Jira REST client for fetch + label/component write-back.

Works anonymously against public Jira for reads. Writes require
JIRA_BASE_URL + JIRA_EMAIL + JIRA_API_TOKEN (or JIRA_USER + JIRA_PASSWORD).
When credentials are missing, updates are returned as dry-run / queued.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from base64 import b64encode
from typing import Any


class JiraClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        email: str | None = None,
        api_token: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self.base_url = (base_url or os.environ.get("JIRA_BASE_URL") or "").rstrip("/")
        self.email = email or os.environ.get("JIRA_EMAIL")
        self.api_token = api_token or os.environ.get("JIRA_API_TOKEN")
        self.user = user or os.environ.get("JIRA_USER")
        self.password = password or os.environ.get("JIRA_PASSWORD")

    @property
    def can_write(self) -> bool:
        if not self.base_url:
            return False
        return bool((self.email and self.api_token) or (self.user and self.password))

    def _auth_header(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "jira-enclave-categorizer/0.1"}
        if self.email and self.api_token:
            token = b64encode(f"{self.email}:{self.api_token}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        elif self.user and self.password:
            token = b64encode(f"{self.user}:{self.password}".encode()).decode()
            headers["Authorization"] = f"Basic {token}"
        return headers

    def search(self, jql: str, *, fields: list[str], limit: int = 100) -> list[dict[str, Any]]:
        if not self.base_url:
            raise RuntimeError("JIRA_BASE_URL is not configured")
        issues: list[dict[str, Any]] = []
        start = 0
        while len(issues) < limit:
            params = urllib.parse.urlencode(
                {
                    "jql": jql,
                    "startAt": start,
                    "maxResults": min(50, limit - len(issues)),
                    "fields": ",".join(fields),
                }
            )
            url = f"{self.base_url}/rest/api/2/search?{params}"
            req = urllib.request.Request(url, headers=self._auth_header())
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            chunk = payload.get("issues") or []
            if not chunk:
                break
            issues.extend(chunk)
            start += len(chunk)
            if start >= payload.get("total", 0):
                break
        return issues[:limit]

    def update_issue_fields(
        self,
        issue_key: str,
        *,
        labels: list[str] | None = None,
        components: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update labels and/or components. Components set by name list."""
        fields: dict[str, Any] = {}
        if labels is not None:
            fields["labels"] = labels
        if components is not None:
            fields["components"] = [{"name": c} for c in components if c]

        body = {"fields": fields}
        if not self.can_write:
            return {
                "ok": False,
                "dry_run": True,
                "issue_key": issue_key,
                "fields": fields,
                "reason": "Jira write credentials not configured; queued locally only",
            }

        url = f"{self.base_url}/rest/api/2/issue/{urllib.parse.quote(issue_key)}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=self._auth_header(), method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                # 204 No Content typical
                _ = resp.read()
            return {"ok": True, "dry_run": False, "issue_key": issue_key, "fields": fields}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "dry_run": False,
                "issue_key": issue_key,
                "fields": fields,
                "error": f"HTTP {exc.code}: {detail[:500]}",
            }
