"""Résolution du modèle à servir (phase 03).

Ordre : Model Registry MLflow (`models:/previ-r2d2-<dossier>-h<horizon>@production`)
puis, en repli, `models/<dossier>/h<horizon>/` (modèle promu, versionné DVC).

Le repli local n'est PAS un mode dégradé : git + DVC restent la source de
vérité pour la reproductibilité exacte (cf. phase 01). Le registry sert la
lecture rapide quand un serveur MLflow est disponible.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from previ_r2d2.common import config
from previ_r2d2.model.tracking import mlflow_tracking

logger = logging.getLogger(__name__)

_CACHE_ROOT = Path(config.ROOT) / "outputs" / "serving_cache"
_lock = threading.Lock()


@dataclass(frozen=True)
class ResolvedModel:
    weights_dir: Path
    origin: str  # "registry" | "local"
    version: int | None
    kge_stacking: float | None


def _local(dossier: str, horizon: int) -> Path:
    return config.MODELS_DIR / dossier / f"h{horizon}"


def _local_meta(dossier: str, horizon: int) -> tuple[int | None, float | None]:
    version_path = _local(dossier, horizon) / "version.json"
    if not version_path.exists():
        return None, None
    data = json.loads(version_path.read_text(encoding="utf-8"))
    return data.get("version"), data.get("kge_stacking")


def _from_registry(dossier: str, horizon: int) -> ResolvedModel | None:
    if config.settings.api_model_source == "local" or not mlflow_tracking.enabled():
        return None
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(mlflow_tracking.tracking_uri())
        name = mlflow_tracking.registered_model_name(dossier, horizon)
        client = MlflowClient(tracking_uri=mlflow_tracking.tracking_uri())
        mv = client.get_model_version_by_alias(name, mlflow_tracking.PRODUCTION_ALIAS)

        dst = _CACHE_ROOT / f"{name}" / f"v{mv.version}"
        with _lock:
            if not (dst / "meta_config.json").exists():
                dst.mkdir(parents=True, exist_ok=True)
                mlflow.artifacts.download_artifacts(
                    artifact_uri=f"models:/{name}@{mlflow_tracking.PRODUCTION_ALIAS}",
                    dst_path=str(dst),
                )
        # download_artifacts recrée l'arbo `model/` : viser le dossier feuille.
        leaf = dst / "model" if (dst / "model" / "meta_config.json").exists() else dst
        kge = None
        results = leaf / "results.json"
        if results.exists():
            kge = json.loads(results.read_text(encoding="utf-8")).get("kge_stacking")
        return ResolvedModel(leaf, "registry", int(mv.version), kge)
    except Exception as exc:  # noqa: BLE001 — repli local sur toute erreur registry
        logger.warning("registry indisponible pour %s h%s (%s) — repli local", dossier, horizon, exc)
        return None


def registry_production_version(dossier: str, horizon: int) -> int | None:
    """N° de version portant l'alias `@production` au registry, sans rien
    télécharger. None si registry désactivé/injoignable ou alias absent."""
    if not mlflow_tracking.enabled():
        return None
    try:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=mlflow_tracking.tracking_uri())
        name = mlflow_tracking.registered_model_name(dossier, horizon)
        mv = client.get_model_version_by_alias(name, mlflow_tracking.PRODUCTION_ALIAS)
        return int(mv.version)
    except Exception:  # noqa: BLE001
        return None


def resolve(dossier: str, horizon: int) -> ResolvedModel:
    """Renvoie le modèle à servir. Lève FileNotFoundError si aucun des deux."""
    resolved = _from_registry(dossier, horizon)
    if resolved is not None:
        return resolved

    local = _local(dossier, horizon)
    if not (local / "meta_config.json").exists():
        raise FileNotFoundError(
            f"{dossier} h{horizon} : aucun modèle en production "
            f"(ni registry MLflow @production, ni {local}/meta_config.json)."
        )
    version, kge = _local_meta(dossier, horizon)
    return ResolvedModel(local, "local", version, kge)
