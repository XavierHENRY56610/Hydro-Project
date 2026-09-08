"""Pipeline d'entraînement previ-R2-D2 porté en ZenML (bonus phase 08).

Démonstration « et si on avait pris ZenML » (cours optionnel Sprint 14) :
mêmes étapes que `orchestrator.run_training`, découpées en `@step` avec
versionnage natif des artefacts (`ArtifactConfig`) — ce que ZenML apporte de
plus que Prefect, et qui fait ici **doublon** avec MLflow Registry + DVC
(cf. docs/mlops/ORCHESTRATORS.md).

Non branché sur l'image principale. Pour l'exécuter :

    pip install "zenml[server]"
    zenml init && zenml stack set default
    python infrastructure/orchestration-alternatives/zenml/training_pipeline.py
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from zenml import ArtifactConfig, pipeline, step

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.data_loading import load_df
from previ_r2d2.model.tracking import mlflow_tracking
from previ_r2d2.preprocessing.data_preparation.validation import validate_data_preparation


@step
def load_step(dossier: str) -> Annotated[pd.DataFrame, "data_preparation"]:
    return load_df(dossier)


@step
def validate_step(df: pd.DataFrame, dossier: str) -> Annotated[bool, "validation_ok"]:
    result = validate_data_preparation(df, strict=True, context=dossier)
    return result.ok


@step
def train_step(
    dossier: str, horizon: int, validation_ok: bool
) -> Annotated[
    dict,
    ArtifactConfig(name="training_results", version=None, run_metadata={"framework": "zenml-demo"}),
]:
    from previ_r2d2.orchestration.flows import _run_script

    if not validation_ok:
        raise RuntimeError("validation KO")
    _run_script("train.py", "--dossier", dossier, "--horizon", str(horizon), "--force")
    version_path = config.MODELS_DIR / dossier / f"h{horizon}" / "version.json"
    import json

    return json.loads(version_path.read_text(encoding="utf-8")) if version_path.exists() else {}


@step
def register_step(results: dict, dossier: str, horizon: int) -> None:
    """Aligne l'alias @production MLflow (même geste que cron/scripts/train.py)."""
    mlflow_tracking.promote_to_production(dossier, horizon, results.get("mlflow_model_version"))


@pipeline
def training(dossier: str = "touzac_g2_G2", horizon: int = 8) -> None:
    df = load_step(dossier)
    ok = validate_step(df, dossier)
    results = train_step(dossier, horizon, ok)
    register_step(results, dossier, horizon)


if __name__ == "__main__":
    training()
