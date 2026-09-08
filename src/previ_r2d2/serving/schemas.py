"""Schémas Pydantic des requêtes / réponses de l'API (phase 03)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from previ_r2d2.model.pipeline.orchestrator import HORIZON_CFG

HORIZONS = sorted(HORIZON_CFG.keys())


class PredictRequest(BaseModel):
    """Corps de `POST /predict`."""

    dossier: str = Field(..., description="Nom exact du dossier centrale (centrales/).", examples=["touzac_g2_G2"])
    horizon: int = Field(..., description="Horizon de prévision en pas.", examples=HORIZONS)
    source: str = Field(
        "live",
        description="`live` : fenêtre assemblée dynamiquement (Hub'Eau + météo). "
        "`frozen` : relit data_preparation.csv, 100 % reproductible.",
    )


class DebitPoint(BaseModel):
    lead: int = Field(..., description="Rang du pas dans la série renvoyée (0 = heure de lancement).")
    q_stacking_m3s: float
    q_entrant_m3s: float


class PredictResponse(BaseModel):
    dossier: str
    horizon: int
    now: str = Field(..., description="Horodatage (ISO) de la dernière observation servant d'ancrage.")
    source: str
    served_version: int | None = Field(None, description="Version du modèle servie (registry ou version.json local).")
    served_from: str = Field(..., description="`registry` (MLflow @production) ou `local` (models/ DVC).")
    points: list[DebitPoint]


class ModelInfo(BaseModel):
    dossier: str
    horizon: int
    version: int | None
    kge_stacking: float | None
    promoted_at: str | None
    origin: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    mlflow: str = Field(..., description="`up`, `down` ou `disabled`.")
    models_available: int


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None
