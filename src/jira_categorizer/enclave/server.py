"""Inference + ops UI server (enclave-friendly HTTP; host may proxy vsock)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from jira_categorizer.analytics.forecast import build_forecast_actual
from jira_categorizer.analytics.timeseries import build_novelty_timeseries, build_timeseries
from jira_categorizer.analytics.triage import build_triage_queue
from jira_categorizer.config import load_config, project_root
from jira_categorizer.data.cache import TicketCache
from jira_categorizer.data.excel_import import import_to_cache, save_upload
from jira_categorizer.data.preprocess import combine_summary_description
from jira_categorizer.enclave.attestation import require_attestation
from jira_categorizer.feedback.store import FeedbackStore
from jira_categorizer.infer.predict import Categorizer
from jira_categorizer.jira.client import JiraClient
from jira_categorizer.monitor.new_labels import suggest_new_labels
from jira_categorizer.monitor.surge import detect_category_surges

app = FastAPI(title="Jira Enclave Categorizer", version="0.1.0")
_STATE: dict[str, Any] = {}

UI_DIR = project_root() / "ui"


class PredictRequest(BaseModel):
    summary: str = Field(..., min_length=1)
    description: str = ""
    issue_key: str | None = None


class BatchPredictRequest(BaseModel):
    tickets: list[PredictRequest]


class TriageSaveRequest(BaseModel):
    issue_key: str
    summary: str = ""
    description: str = ""
    created: str | None = None
    selected_labels: list[str] = Field(default_factory=list)
    selected_areas: list[str] = Field(default_factory=list)
    suggested_labels: list[str] = Field(default_factory=list)
    suggested_area: str | None = None
    push_to_jira: bool = True


class SourceModeRequest(BaseModel):
    mode: str = Field(..., description="cache | jira | spreadsheet")


class JiraConnectRequest(BaseModel):
    base_url: str
    jql: str = "order by created DESC"
    fetch_limit: int = 200
    email: str | None = None
    api_token: str | None = None


class TrainStartRequest(BaseModel):
    window: str = "1m"  # 1m | 1y | all
    params: dict[str, Any] = Field(default_factory=dict)


class ValidateDayRequest(BaseModel):
    day: str
    train_window: str = "1m"


def _ensure_state() -> None:
    if "cfg" not in _STATE:
        _STATE["cfg"] = load_config(os.environ.get("JIRA_CAT_CONFIG"))
    cfg = _STATE["cfg"]
    paths = cfg.setdefault("paths", {})
    for key, default in (
        ("cache_dir", "artifacts/cache"),
        ("feedback_dir", "artifacts/feedback"),
        ("uploads_dir", "artifacts/uploads"),
    ):
        if key not in paths:
            p = project_root() / default
            paths[key] = str(p)
        else:
            p = Path(paths[key])
            if not p.is_absolute():
                paths[key] = str(project_root() / p)
        Path(paths[key]).mkdir(parents=True, exist_ok=True)

    if "cache" not in _STATE:
        ttl = int(cfg.get("ui", {}).get("cache_ttl_seconds", 300))
        _STATE["cache"] = TicketCache(paths["cache_dir"], ttl_seconds=ttl)
    if "feedback" not in _STATE:
        _STATE["feedback"] = FeedbackStore(paths["feedback_dir"])
    if "jira" not in _STATE:
        jira_cfg = cfg.get("jira", {})
        _STATE["jira"] = JiraClient(
            jira_cfg.get("base_url") or os.environ.get("JIRA_BASE_URL"),
            email=jira_cfg.get("email") or os.environ.get("JIRA_EMAIL"),
            api_token=jira_cfg.get("api_token") or os.environ.get("JIRA_API_TOKEN"),
        )
    if "source_mode" not in _STATE:
        # Prefer cache/spreadsheet when offline; jira only when explicitly enabled
        if cfg.get("jira", {}).get("enabled"):
            _STATE["source_mode"] = "jira"
        else:
            _STATE["source_mode"] = "cache"
    if "model" not in _STATE:
        require_attestation(cfg)
        _STATE["model"] = Categorizer.load(cfg, version="production")
    if "train_job" not in _STATE:
        from jira_categorizer.train.console import TrainJobState

        _STATE["train_job"] = TrainJobState(paths["reports_dir"])


def _fetch_jira_df() -> Any:
    import pandas as pd

    cfg = _STATE["cfg"]
    client: JiraClient = _STATE["jira"]
    if not client.base_url:
        raise RuntimeError("Jira base_url is not configured")
    jira_cfg = cfg.get("jira", {})
    jql = jira_cfg.get("jql") or "order by created DESC"
    issues = client.search(
        jql,
        fields=[
            "summary",
            "description",
            "labels",
            "components",
            "issuetype",
            "status",
            "created",
            "updated",
        ],
        limit=int(jira_cfg.get("fetch_limit", 200)),
    )
    rows = []
    for issue in issues:
        f = issue.get("fields") or {}
        desc = f.get("description") or ""
        if isinstance(desc, dict):
            import json

            desc = json.dumps(desc)
        comps = [c.get("name", "") for c in (f.get("components") or []) if c.get("name")]
        rows.append(
            {
                "issue_key": issue.get("key"),
                "summary": f.get("summary") or "",
                "description": desc,
                "labels": "|".join(f.get("labels") or []),
                "impacted_area": "|".join(comps),
                "issuetype": (f.get("issuetype") or {}).get("name", ""),
                "status": (f.get("status") or {}).get("name", ""),
                "created": f.get("created") or "",
                "updated": f.get("updated") or "",
            }
        )
    return pd.DataFrame(rows), jql


def _tickets(force_refresh: bool = False):
    _ensure_state()
    cfg = _STATE["cfg"]
    cache: TicketCache = _STATE["cache"]
    mode = _STATE.get("source_mode", "cache")

    if mode == "jira":
        if force_refresh or not cache.is_fresh() or not str(cache._meta_read().get("source", "")).startswith("jira:"):
            try:
                df, jql = _fetch_jira_df()
                cache.save(df, source=f"jira:{jql}")
                return df
            except Exception as exc:
                # Fall back to cache/file when offline
                cached = cache.load()
                if cached is not None:
                    return cached
                return cache.get_or_load_file(cfg["paths"]["raw_data"], force=True)
        cached = cache.load()
        if cached is not None:
            return cached

    if mode == "spreadsheet":
        cached = cache.load()
        if cached is not None and str(cache._meta_read().get("source", "")).startswith("spreadsheet:"):
            return cached
        # No spreadsheet yet — fall through to bundled sample so UI still works
        return cache.get_or_load_file(cfg["paths"]["raw_data"], force=force_refresh)

    # cache mode: local file / last import
    return cache.get_or_load_file(cfg["paths"]["raw_data"], force=force_refresh)


@app.on_event("startup")
def _startup() -> None:
    _ensure_state()


@app.get("/health")
def health() -> dict[str, Any]:
    _ensure_state()
    model: Categorizer | None = _STATE.get("model")
    return {
        "status": "ok" if model else "loading",
        "model_version": (model.manifest.get("version") if model else None),
        "jira_writable": _STATE["jira"].can_write,
    }


@app.get("/attestation")
def attestation() -> dict[str, Any]:
    _ensure_state()
    return require_attestation(_STATE["cfg"]).to_dict()


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    _ensure_state()
    model: Categorizer = _STATE["model"]
    result = model.predict_one(req.summary, req.description)
    if req.issue_key:
        result["issue_key"] = req.issue_key
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest) -> dict[str, Any]:
    _ensure_state()
    if not req.tickets:
        raise HTTPException(status_code=400, detail="tickets required")
    model: Categorizer = _STATE["model"]
    texts = [
        combine_summary_description(
            t.summary,
            t.description,
            strip_html_tags=bool(_STATE["cfg"]["features"].get("strip_html", True)),
        )
        for t in req.tickets
    ]
    preds = model.predict_texts(texts)
    for i, t in enumerate(req.tickets):
        if t.issue_key:
            preds[i]["issue_key"] = t.issue_key
    return {"results": preds}


@app.get("/monitor/surges")
def monitor_surges() -> dict[str, Any]:
    _ensure_state()
    cfg = _STATE["cfg"]
    tickets = _tickets()
    return detect_category_surges(
        tickets,
        cfg=cfg,
        label_field=cfg["data"]["label_field"],
        area_field=cfg["data"]["area_field"],
    )


@app.get("/monitor/new-labels")
def monitor_new_labels() -> dict[str, Any]:
    _ensure_state()
    cfg = _STATE["cfg"]
    model: Categorizer = _STATE["model"]
    return suggest_new_labels(_tickets(), model, cfg=cfg)


@app.get("/api/meta")
def api_meta() -> dict[str, Any]:
    _ensure_state()
    cfg = _STATE["cfg"]
    tickets = _tickets()
    return {
        "project": cfg.get("project", {}),
        "rows": int(len(tickets)),
        "dimensions": ["labels", "impacted_area", "issuetype", "status"],
        "windows": ["1d", "1w", "1m", "3m"],
        "jira_writable": _STATE["jira"].can_write,
        "jira_configured": bool(_STATE["jira"].base_url),
        "source_mode": _STATE.get("source_mode", "cache"),
        "model_version": _STATE["model"].manifest.get("version"),
        "source": _STATE["cache"]._meta_read(),
        "offline_ok": True,
    }


@app.post("/api/source")
def api_set_source(req: SourceModeRequest) -> dict[str, Any]:
    _ensure_state()
    mode = req.mode.strip().lower()
    if mode not in {"cache", "jira", "spreadsheet"}:
        raise HTTPException(status_code=400, detail="mode must be cache|jira|spreadsheet")
    _STATE["source_mode"] = mode
    if mode == "jira":
        _STATE["cfg"].setdefault("jira", {})["enabled"] = True
    df = _tickets(force_refresh=(mode == "jira"))
    return {
        "ok": True,
        "source_mode": mode,
        "rows": int(len(df)),
        "source": _STATE["cache"]._meta_read(),
    }


@app.post("/api/jira/connect")
def api_jira_connect(req: JiraConnectRequest) -> dict[str, Any]:
    _ensure_state()
    cfg = _STATE["cfg"]
    cfg.setdefault("jira", {})
    cfg["jira"].update(
        {
            "enabled": True,
            "base_url": req.base_url.rstrip("/"),
            "jql": req.jql,
            "fetch_limit": req.fetch_limit,
            "email": req.email,
            "api_token": req.api_token,
        }
    )
    _STATE["jira"] = JiraClient(req.base_url, email=req.email, api_token=req.api_token)
    _STATE["source_mode"] = "jira"
    try:
        df, jql = _fetch_jira_df()
        _STATE["cache"].save(df, source=f"jira:{jql}")
        return {
            "ok": True,
            "rows": int(len(df)),
            "source_mode": "jira",
            "source": _STATE["cache"]._meta_read(),
            "jira_writable": _STATE["jira"].can_write,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Jira unreachable or rejected the request ({exc}). Use Excel import for offline mode.",
        ) from exc


@app.post("/api/import/spreadsheet")
async def api_import_spreadsheet(file: UploadFile = File(...)) -> dict[str, Any]:
    _ensure_state()
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty upload")
    try:
        path = save_upload(content, _STATE["cfg"]["paths"]["uploads_dir"], file.filename or "export.xlsx")
        info = import_to_cache(path, _STATE["cache"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _STATE["source_mode"] = "spreadsheet"
    # Also point training raw_data at imported CSV snapshot for retrain convenience
    snap = Path(_STATE["cfg"]["paths"]["cache_dir"]) / "tickets.csv"
    _STATE["cfg"]["paths"]["raw_data"] = str(snap)
    return {"ok": True, "source_mode": "spreadsheet", **info, "cache": _STATE["cache"]._meta_read()}


@app.post("/api/refresh")
def api_refresh() -> dict[str, Any]:
    _ensure_state()
    df = _tickets(force_refresh=True)
    return {
        "ok": True,
        "rows": int(len(df)),
        "source_mode": _STATE.get("source_mode"),
        "source": _STATE["cache"]._meta_read(),
    }


@app.get("/api/forecast")
def api_forecast(dimension: str = "labels") -> dict[str, Any]:
    """Today + tomorrow hourly forecast vs actual, with management summary."""
    _ensure_state()
    cfg = _STATE["cfg"]
    tickets = _tickets()
    model: Categorizer = _STATE["model"]
    texts = [
        combine_summary_description(
            str(r.get("summary", "")),
            str(r.get("description", "")),
            strip_html_tags=bool(cfg["features"].get("strip_html", True)),
        )
        for _, r in tickets.iterrows()
    ]
    preds = model.predict_texts(texts) if len(texts) else []
    surges = detect_category_surges(tickets, cfg=cfg)
    suggestions = suggest_new_labels(tickets, model, cfg=cfg)
    return build_forecast_actual(
        tickets,
        predictions=preds,
        surges=surges.get("surges") or [],
        suggested_labels=suggestions.get("suggestions") or [],
        dimension=dimension if dimension in {"labels", "impacted_area"} else "labels",
    )


@app.get("/api/timeseries")
def api_timeseries(dimension: str = "labels", window: str = "1m", top_n: int = 8) -> dict[str, Any]:
    _ensure_state()
    try:
        return build_timeseries(_tickets(), dimension=dimension, window=window, top_n=top_n)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/timeseries/novelty")
def api_timeseries_novelty(window: str = "1m") -> dict[str, Any]:
    _ensure_state()
    tickets = _tickets()
    model: Categorizer = _STATE["model"]
    texts = [
        combine_summary_description(
            str(r.get("summary", "")),
            str(r.get("description", "")),
            strip_html_tags=bool(_STATE["cfg"]["features"].get("strip_html", True)),
        )
        for _, r in tickets.iterrows()
    ]
    preds = model.predict_texts(texts) if len(texts) else []
    try:
        series = build_novelty_timeseries(tickets, preds, window=window)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    surges = detect_category_surges(tickets, cfg=_STATE["cfg"])
    suggestions = suggest_new_labels(tickets, model, cfg=_STATE["cfg"])
    series["surges"] = surges.get("surges") or []
    series["suggested_labels"] = [
        {"suggested_label": s["suggested_label"], "cluster_size": s["cluster_size"]}
        for s in (suggestions.get("suggestions") or [])
    ]
    return series


@app.get("/api/triage")
def api_triage(days: int = 1, limit: int = 40) -> dict[str, Any]:
    _ensure_state()
    cfg = _STATE["cfg"]
    model: Categorizer = _STATE["model"]
    tickets = _tickets()
    suggestions = suggest_new_labels(tickets, model, cfg=cfg)
    new_labs = [s["suggested_label"] for s in (suggestions.get("suggestions") or [])]
    queue = build_triage_queue(
        tickets,
        model,
        days=days,
        limit=limit,
        suggested_new_labels=new_labs,
    )
    queue["new_label_suggestions"] = suggestions.get("suggestions") or []
    return queue


@app.post("/api/triage/save")
def api_triage_save(req: TriageSaveRequest) -> dict[str, Any]:
    _ensure_state()
    suggested_set = set(req.suggested_labels or [])
    selected_set = set(req.selected_labels or [])
    disagreed = (selected_set != suggested_set) or (
        (req.suggested_area or None) != ((req.selected_areas[0] if req.selected_areas else None))
    )
    jira_result: dict[str, Any] = {"ok": False, "skipped": True}
    if req.push_to_jira:
        jira_result = _STATE["jira"].update_issue_fields(
            req.issue_key,
            labels=req.selected_labels,
            components=req.selected_areas[:1] if req.selected_areas else [],
        )

    record = {
        "issue_key": req.issue_key,
        "summary": req.summary,
        "description": req.description,
        "created": req.created,
        "suggested_labels": req.suggested_labels,
        "suggested_area": req.suggested_area,
        "selected_labels": req.selected_labels,
        "selected_areas": req.selected_areas,
        "disagreed": disagreed,
        "jira_result": jira_result,
    }
    saved = _STATE["feedback"].append(record)

    # Refresh local cache row so charts/triage reflect human labels immediately
    cache: TicketCache = _STATE["cache"]
    df = cache.load()
    if df is not None and not df.empty:
        mask = df["issue_key"].astype(str) == req.issue_key
        if mask.any():
            df.loc[mask, "labels"] = "|".join(req.selected_labels)
            df.loc[mask, "impacted_area"] = "|".join(req.selected_areas)
            cache.save(df, source="triage-feedback")

    return {
        "ok": True,
        "disagreed": disagreed,
        "feedback": saved,
        "jira": jira_result,
        "learning": "Correction stored for next retrain (artifacts/feedback/corrections.jsonl)",
    }


@app.post("/api/retrain-with-feedback")
def api_retrain_with_feedback() -> dict[str, Any]:
    """Merge feedback into training CSV and retrain production model."""
    _ensure_state()
    from jira_categorizer.train.train import train

    cfg = _STATE["cfg"]
    merged_path = Path(cfg["paths"]["feedback_dir"]) / "train_with_feedback.csv"
    merge_info = _STATE["feedback"].merge_into_training_csv(cfg["paths"]["raw_data"], merged_path)
    result = train(cfg, data_path=str(merged_path))
    _STATE["model"] = Categorizer.load(cfg, version="production")
    return {
        "ok": True,
        "merge": merge_info,
        "version": result["version"],
        "promoted": result["promoted"],
        "metrics": {
            "labels_f1_macro": result["metrics"].get("labels", {}).get("f1_macro"),
            "area_f1_macro": result["metrics"].get("area", {}).get("f1_macro"),
        },
    }


@app.get("/api/train/params")
def api_train_params() -> dict[str, Any]:
    _ensure_state()
    from jira_categorizer.train.console import current_params

    return {"params": current_params(_STATE["cfg"])}


@app.get("/api/train/status")
def api_train_status() -> dict[str, Any]:
    _ensure_state()
    return _STATE["train_job"].status()


@app.post("/api/train/start")
def api_train_start(req: TrainStartRequest, sync: bool = False) -> dict[str, Any]:
    _ensure_state()
    from jira_categorizer.train.console import apply_train_params, run_windowed_train

    job = _STATE["train_job"]
    tickets = _tickets()
    cfg = _STATE["cfg"]
    params = req.params or {}

    def work() -> None:
        result = run_windowed_train(
            cfg=cfg,
            tickets=tickets,
            window=req.window,
            params=params,
            job=job,
        )
        _STATE["cfg"] = apply_train_params(_STATE["cfg"], params)
        _STATE["model"] = Categorizer.load(_STATE["cfg"], version="production")
        job.log(f"Server reloaded production model {result.get('version')}")

    if sync:
        # Deterministic path for tests / short runs
        job.log_path.write_text("", encoding="utf-8")
        job._write(
            {
                "state": "running",
                "started_at": __import__("datetime").datetime.now(
                    __import__("datetime").timezone.utc
                ).isoformat(),
                "finished_at": None,
                "error": None,
                "result": None,
            }
        )
        try:
            work()
        except Exception as exc:
            job._write(
                {
                    "state": "failed",
                    "started_at": job.status().get("started_at"),
                    "finished_at": __import__("datetime").datetime.now(
                        __import__("datetime").timezone.utc
                    ).isoformat(),
                    "error": str(exc),
                    "result": None,
                }
            )
        return job.status()
    return job.start(work)


@app.get("/api/validate/days")
def api_validate_days(train_window: str = "1m") -> dict[str, Any]:
    _ensure_state()
    from jira_categorizer.train.console import available_validation_days

    days = available_validation_days(_tickets(), train_window)
    return {"train_window": train_window, "days": days}


@app.post("/api/validate/day")
def api_validate_day(req: ValidateDayRequest) -> dict[str, Any]:
    _ensure_state()
    from jira_categorizer.train.console import validate_on_day

    try:
        result = validate_on_day(
            cfg=_STATE["cfg"],
            tickets=_tickets(),
            day=req.day,
            categorizer=_STATE["model"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    # Persist last validation for UI refresh
    path = Path(_STATE["cfg"]["paths"]["reports_dir"]) / "last_validation.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


@app.get("/")
def ui_index():
    index = UI_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI not built")
    return FileResponse(index)


if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR / "static")), name="static")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve categorizer + ops UI")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _STATE["cfg"] = cfg
    os.environ["JIRA_CAT_CONFIG"] = str(args.config or (project_root() / "config" / "default.yaml"))
    host = args.host or cfg["inference"].get("host", "127.0.0.1")
    port = args.port or int(cfg["inference"].get("port", 8080))
    import uvicorn

    # Pass app object so _STATE/config from this process are used
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
