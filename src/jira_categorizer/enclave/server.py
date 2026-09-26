"""Inference HTTP server intended to run inside the enclave (host proxies vsock)."""

from __future__ import annotations

import argparse
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from jira_categorizer.config import load_config
from jira_categorizer.enclave.attestation import require_attestation
from jira_categorizer.infer.predict import Categorizer

app = FastAPI(title="Jira Enclave Categorizer", version="0.1.0")
_STATE: dict[str, Any] = {}


class PredictRequest(BaseModel):
    summary: str = Field(..., min_length=1)
    description: str = ""
    issue_key: str | None = None


class BatchPredictRequest(BaseModel):
    tickets: list[PredictRequest]


@app.on_event("startup")
def _startup() -> None:
    # Config path may be injected via env by the enclave boot script
    cfg = _STATE.get("cfg") or load_config()
    require_attestation(cfg)
    _STATE["cfg"] = cfg
    _STATE["model"] = Categorizer.load(cfg, version="production")


@app.get("/health")
def health() -> dict[str, Any]:
    model: Categorizer | None = _STATE.get("model")
    return {
        "status": "ok" if model else "loading",
        "model_version": (model.manifest.get("version") if model else None),
    }


@app.get("/attestation")
def attestation() -> dict[str, Any]:
    cfg = _STATE["cfg"]
    return require_attestation(cfg).to_dict()


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    model: Categorizer = _STATE["model"]
    result = model.predict_one(req.summary, req.description)
    if req.issue_key:
        result["issue_key"] = req.issue_key
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest) -> dict[str, Any]:
    if not req.tickets:
        raise HTTPException(status_code=400, detail="tickets required")
    model: Categorizer = _STATE["model"]
    texts = []
    from jira_categorizer.data.preprocess import combine_summary_description

    for t in req.tickets:
        texts.append(
            combine_summary_description(
                t.summary,
                t.description,
                strip_html_tags=bool(_STATE["cfg"]["features"].get("strip_html", True)),
            )
        )
    preds = model.predict_texts(texts)
    for i, t in enumerate(req.tickets):
        if t.issue_key:
            preds[i]["issue_key"] = t.issue_key
    return {"results": preds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve categorizer inside enclave")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    _STATE["cfg"] = cfg
    host = args.host or cfg["inference"]["host"]
    port = args.port or int(cfg["inference"]["port"])
    import uvicorn

    uvicorn.run(
        "jira_categorizer.enclave.server:app",
        host=host,
        port=port,
        factory=False,
    )


if __name__ == "__main__":
    main()
