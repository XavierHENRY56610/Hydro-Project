"""Suivi d'expériences MLflow — activé uniquement si `MLFLOW_TRACKING_URI`
(ou `settings.mlflow_tracking_uri`) est défini. Sans lui, toutes les fonctions
sont des no-op : les tests et la CI tournent sans serveur MLflow.

Registry : un modèle enregistré par (dossier, horizon), nommé
`previ-r2d2-<dossier>-h<horizon>`. Chaque entraînement crée une nouvelle
version (sans alias). La promotion (train.py) pose l'alias `@production` sur
la version retenue et le retire de l'ancienne — c'est ce que l'API charge
(`models:/previ-r2d2-<dossier>-h<horizon>@production`).
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path

from previ_r2d2.common import config

REGISTERED_MODEL_FMT = "previ-r2d2-{dossier}-h{horizon}"
PRODUCTION_ALIAS = "production"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or config.settings.mlflow_tracking_uri


def enabled() -> bool:
    return bool(tracking_uri())


def registered_model_name(dossier: str, horizon: int) -> str:
    return REGISTERED_MODEL_FMT.format(dossier=dossier, horizon=horizon)


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


@contextlib.contextmanager
def training_run(dossier: str, horizon: int, meta_type: str, params: dict):
    """Contexte d'un run d'entraînement. `yield` le run_id, ou None si MLflow
    est désactivé."""
    if not enabled():
        yield None
        return

    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(f"previ-r2d2/{dossier}")
    with mlflow.start_run(run_name=f"{dossier}-h{horizon}-{meta_type}") as run:
        mlflow.set_tags(
            {
                "dossier": dossier,
                "horizon": str(horizon),
                "meta_type": meta_type,
                "git_sha": _git_sha(),
            }
        )
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        yield run.info.run_id


def _active() -> bool:
    if not enabled():
        return False
    import mlflow

    return mlflow.active_run() is not None


def log_results(results: dict) -> None:
    """Loggue les métriques : KGE global (lgbm/lstm/stacking), composantes
    stacking, KGE + RMSE par pas d'horizon, KGE par régime et par saison."""
    if not _active():
        return
    import mlflow

    metrics: dict[str, float] = {}
    for key in ("kge_lgbm", "kge_lstm", "kge_stacking"):
        if results.get(key) is not None:
            metrics[key] = float(results[key])

    for comp_key, val in (results.get("components", {}).get("Stacking", {}) or {}).items():
        if val is not None:
            metrics[f"stacking_{comp_key}"] = float(val)

    by_step = results.get("kge_by_step", {}) or {}
    for i, val in enumerate(by_step.get("stack") or [], start=1):
        if val is not None:
            metrics[f"kge_step_{i}"] = float(val)
    for i, val in enumerate(by_step.get("rmse_m3s_stack") or [], start=1):
        if val is not None:
            metrics[f"rmse_m3s_step_{i}"] = float(val)

    for regime, d in (results.get("kge_by_regime", {}).get("Stacking", {}) or {}).items():
        if d.get("kge") is not None:
            metrics[f"kge_regime_{regime}"] = float(d["kge"])
    for saison, d in (results.get("kge_by_season", {}).get("Stacking", {}) or {}).items():
        if d.get("kge") is not None:
            metrics[f"kge_season_{saison}"] = float(d["kge"])

    if metrics:
        mlflow.log_metrics(metrics)


def log_data_validation(result) -> None:
    """Trace le résultat de la validation data_preparation : tag + warnings en
    artefact texte. `result` = ValidationResult (peut être n'importe quel objet
    exposant `.ok` / `.warnings`)."""
    if not _active():
        return
    import json
    import tempfile

    import mlflow

    mlflow.set_tag("data_validation_ok", str(getattr(result, "ok", True)))
    warnings = list(getattr(result, "warnings", []) or [])
    mlflow.set_tag("data_validation_warnings", str(len(warnings)))
    if warnings:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"warnings": warnings}, fh, indent=2, ensure_ascii=False)
            tmp = fh.name
        mlflow.log_artifact(tmp, artifact_path="data_validation")


def log_artifacts(weights_dir: Path, outputs_dir: Path) -> None:
    """Loggue results.json / meta_config.json, les plots et les CSV de sortie."""
    if not _active():
        return
    import mlflow

    for name in ("results.json", "meta_config.json"):
        p = weights_dir / name
        if p.exists():
            mlflow.log_artifact(str(p))
    plots = weights_dir / "plots"
    if plots.is_dir():
        mlflow.log_artifacts(str(plots), artifact_path="plots")
    if outputs_dir.is_dir() and any(outputs_dir.iterdir()):
        mlflow.log_artifacts(str(outputs_dir), artifact_path="outputs")


def register_candidate(
    dossier: str, horizon: int, weights_dir: Path, run_id: str | None
) -> int | None:
    """Loggue le dossier modèle candidat comme artefact `model/` et l'enregistre
    au Model Registry (nouvelle version, sans alias). Retourne le n° de version."""
    if not _active() or run_id is None:
        return None
    import mlflow

    mlflow.log_artifacts(str(weights_dir), artifact_path="model")
    name = registered_model_name(dossier, horizon)
    version = mlflow.register_model(f"runs:/{run_id}/model", name)
    return int(version.version)


def promote_to_production(dossier: str, horizon: int, version: int | None) -> None:
    """Pose l'alias `@production` sur `version`, le retire de l'ancienne."""
    if not enabled() or version is None:
        return
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri())
    name = registered_model_name(dossier, horizon)
    with contextlib.suppress(Exception):
        current = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
        if current and int(current.version) != version:
            client.set_registered_model_tag(name, f"archived_v{current.version}", "true")
    client.set_registered_model_alias(name, PRODUCTION_ALIAS, str(version))
