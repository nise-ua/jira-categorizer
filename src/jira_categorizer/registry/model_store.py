"""Local model registry with optional enclave sealing."""

from __future__ import annotations

import io
import json
import pickle
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from jira_categorizer.enclave.crypto import seal_bytes, unseal_bytes


@dataclass
class ModelManifest:
    name: str
    version: str
    created_at: str
    metrics: dict[str, Any]
    label_classes: list[str]
    area_classes: list[str]
    config_snapshot: dict[str, Any]
    sealed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelStore:
    def __init__(self, model_dir: str | Path, sealed_dir: str | Path | None = None):
        self.model_dir = Path(model_dir)
        self.sealed_dir = Path(sealed_dir) if sealed_dir else self.model_dir / "sealed"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.sealed_dir.mkdir(parents=True, exist_ok=True)

    def _version_dir(self, version: str) -> Path:
        return self.model_dir / version

    def save_bundle(
        self,
        *,
        version: str,
        vectorizer,
        labels_estimator,
        labels_mlb,
        area_estimator,
        area_encoder,
        metrics: dict[str, Any],
        config: dict[str, Any],
        seal: bool = False,
        key_path: str | Path | None = None,
        promote_as: str | None = "production",
    ) -> ModelManifest:
        vdir = self._version_dir(version)
        if vdir.exists():
            shutil.rmtree(vdir)
        vdir.mkdir(parents=True)

        joblib.dump(vectorizer, vdir / "vectorizer.joblib")
        joblib.dump(labels_estimator, vdir / "labels_estimator.joblib")
        joblib.dump(labels_mlb, vdir / "labels_mlb.joblib")
        joblib.dump(area_estimator, vdir / "area_estimator.joblib")
        joblib.dump(area_encoder, vdir / "area_encoder.joblib")

        manifest = ModelManifest(
            name=config.get("mlops", {}).get("registry_name", "jira-categorizer"),
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics,
            label_classes=list(getattr(labels_mlb, "classes_", [])),
            area_classes=list(getattr(area_encoder, "classes_", [])),
            config_snapshot={
                "features": config.get("features"),
                "models": config.get("models"),
                "data": config.get("data"),
            },
            sealed=False,
        )
        (vdir / "manifest.json").write_text(
            json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
        )
        (vdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if seal and key_path:
            sealed_path = self.sealed_dir / f"{version}.seal"
            buf = io.BytesIO()
            joblib.dump(
                {
                    "vectorizer": vectorizer,
                    "labels_estimator": labels_estimator,
                    "labels_mlb": labels_mlb,
                    "area_estimator": area_estimator,
                    "area_encoder": area_encoder,
                    "manifest": manifest.to_dict(),
                },
                buf,
            )
            sealed_path.write_bytes(seal_bytes(buf.getvalue(), key_path))
            manifest.sealed = True
            (vdir / "manifest.json").write_text(
                json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
            )

        if promote_as:
            prod = self.model_dir / promote_as
            if prod.exists() or prod.is_symlink():
                if prod.is_symlink() or prod.is_file():
                    prod.unlink()
                else:
                    shutil.rmtree(prod)
            # Copy rather than symlink for enclave filesystem simplicity
            shutil.copytree(vdir, prod)

        return manifest

    def load_bundle(self, version: str = "production") -> dict[str, Any]:
        vdir = self._version_dir(version)
        if not vdir.exists():
            raise FileNotFoundError(f"Model version not found: {vdir}")
        return {
            "vectorizer": joblib.load(vdir / "vectorizer.joblib"),
            "labels_estimator": joblib.load(vdir / "labels_estimator.joblib"),
            "labels_mlb": joblib.load(vdir / "labels_mlb.joblib"),
            "area_estimator": joblib.load(vdir / "area_estimator.joblib"),
            "area_encoder": joblib.load(vdir / "area_encoder.joblib"),
            "manifest": json.loads((vdir / "manifest.json").read_text(encoding="utf-8")),
        }

    def load_sealed(self, version: str, key_path: str | Path) -> dict[str, Any]:
        sealed_path = self.sealed_dir / f"{version}.seal"
        if not sealed_path.exists():
            raise FileNotFoundError(f"Sealed model not found: {sealed_path}")
        raw = unseal_bytes(sealed_path.read_bytes(), key_path)
        try:
            return joblib.load(io.BytesIO(raw))
        except Exception:
            return pickle.loads(raw)

    def latest_metrics(self, version: str = "production") -> dict[str, Any]:
        path = self._version_dir(version) / "metrics.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))
