"""API de service previ-R2-D2 (phase 03) — FastAPI.

Endpoints :
  POST /predict            — calcule une prévision (dossier, horizon)
  GET  /forecast/{dossier} — dernières prévisions archivées (prevision/enchere.json)
  GET  /models             — inventaire des modèles en production
  GET  /health             — sonde de vivacité (publique)
  GET  /metrics            — métriques Prometheus (publique)

Lancement : `uvicorn previ_r2d2.serving.api:app` (voir Dockerfile.api / compose).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.orchestrator import HORIZON_CFG
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction
from previ_r2d2.model.tracking import mlflow_tracking
from previ_r2d2.serving import metrics, model_registry
from previ_r2d2.serving.logging_config import configure_logging, request_id_var
from previ_r2d2.serving.schemas import (
    DebitPoint,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    PredictRequest,
    PredictResponse,
)
from previ_r2d2.serving.security import require_auth

logger = logging.getLogger("previ_r2d2.serving")

HORIZONS = sorted(HORIZON_CFG.keys())


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_logging()
    logger.info(
        "API prête",
        extra={"auth_required": bool(config.settings.api_auth_required),
               "mlflow": mlflow_tracking.tracking_uri() or "disabled"},
    )
    yield


app = FastAPI(
    title="previ-R2-D2 — API de prévision de débit",
    version="0.3.0",
    summary="Prévision hybride LightGBM + BiLSTM + stacking pour centrales hydroélectriques.",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Middleware : identifiant de requête, log d'accès, métriques
# --------------------------------------------------------------------------
@app.middleware("http")
async def observability(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    route = request.scope.get("route")
    path_label = getattr(route, "path", request.url.path)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("erreur non gérée", extra={"method": request.method, "path": path_label})
        metrics.REQUESTS.labels(request.method, path_label, "500").inc()
        metrics.LATENCY.labels(request.method, path_label).observe(elapsed)
        request_id_var.reset(token)
        return JSONResponse(
            status_code=500,
            content={"detail": "erreur interne", "request_id": rid},
            headers={"X-Request-ID": rid},
        )
    elapsed = time.perf_counter() - start
    metrics.REQUESTS.labels(request.method, path_label, str(response.status_code)).inc()
    metrics.LATENCY.labels(request.method, path_label).observe(elapsed)
    response.headers["X-Request-ID"] = rid
    logger.info(
        "requête traitée",
        extra={"method": request.method, "path": path_label,
               "status": response.status_code, "duration_ms": round(elapsed * 1000, 1)},
    )
    request_id_var.reset(token)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "request_id": request_id_var.get()},
        headers=exc.headers or {},
    )


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _load_bv_json(dossier: str) -> dict:
    path = config.CENTRALES_DIR / dossier / "bv.json"
    if not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"centrale inconnue : {dossier}")
    return json.loads(path.read_text(encoding="utf-8"))


def _discover_dossiers() -> list[str]:
    return sorted(p.parent.name for p in config.CENTRALES_DIR.glob("*/bv.json"))


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    if not mlflow_tracking.enabled():
        mlflow_state = "disabled"
    else:
        mlflow_state = "up"
        try:
            import urllib.request

            with urllib.request.urlopen(f"{mlflow_tracking.tracking_uri()}/health", timeout=2) as r:
                mlflow_state = "up" if r.status == 200 else "down"
        except Exception:  # noqa: BLE001
            mlflow_state = "down"
    n_models = sum(
        1 for _ in config.MODELS_DIR.glob("*/h*/meta_config.json")
    )
    return HealthResponse(status="ok", mlflow=mlflow_state, models_available=n_models)


@app.get("/metrics", tags=["ops"])
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)


@app.get("/models", response_model=ModelsResponse, tags=["models"])
def list_models(_: None = Depends(require_auth)) -> ModelsResponse:
    out: list[ModelInfo] = []
    for version_path in sorted(config.MODELS_DIR.glob("*/h*/version.json")):
        horizon_dir = version_path.parent
        dossier = horizon_dir.parent.name
        horizon = int(horizon_dir.name.removeprefix("h"))
        data = json.loads(version_path.read_text(encoding="utf-8"))
        reg_version = model_registry.registry_production_version(dossier, horizon)
        out.append(ModelInfo(
            dossier=dossier,
            horizon=horizon,
            version=data.get("version"),
            kge_stacking=data.get("kge_stacking"),
            promoted_at=data.get("promoted_at"),
            origin="registry" if reg_version is not None else "local",
        ))
    return ModelsResponse(models=out)


@app.get("/forecast/{dossier}", tags=["forecast"])
def latest_forecast(dossier: str, _: None = Depends(require_auth)) -> dict:
    base = config.CENTRALES_DIR / dossier
    if not base.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"centrale inconnue : {dossier}")
    out: dict = {"dossier": dossier}
    prevision = base / "prevision.json"
    enchere = base / "enchere.json"
    if prevision.exists():
        out["prevision"] = json.loads(prevision.read_text(encoding="utf-8"))
    if enchere.exists():
        out["enchere"] = json.loads(enchere.read_text(encoding="utf-8"))
    if "prevision" not in out and "enchere" not in out:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"aucune prévision archivée pour {dossier} (POST /predict d'abord)",
        )
    return out


@app.post("/predict", response_model=PredictResponse, tags=["forecast"])
def predict(req: PredictRequest, _: None = Depends(require_auth)) -> PredictResponse:
    if req.horizon not in HORIZON_CFG:
        raise HTTPException(422, f"horizon invalide : {req.horizon} (attendus : {HORIZONS})")
    if req.source not in ("live", "frozen"):
        raise HTTPException(422, f"source invalide : {req.source} (live | frozen)")

    bv_json = _load_bv_json(req.dossier)
    exutoire = bv_json.get("exutoire")
    if exutoire is None:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            f"bv.json de {req.dossier} sans clé 'exutoire'")

    try:
        resolved = model_registry.resolve(req.dossier, req.horizon)
    except FileNotFoundError as exc:
        metrics.PREDICTIONS.labels(req.dossier, str(req.horizon), "no_model").inc()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc

    now = pd.Timestamp.now().floor("h")
    try:
        result = run_prediction(
            req.dossier, req.horizon, exutoire, bv_json, now,
            source=req.source, weights_dir=resolved.weights_dir,
        )
    except FileNotFoundError as exc:
        metrics.PREDICTIONS.labels(req.dossier, str(req.horizon), "no_model").inc()
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        metrics.PREDICTIONS.labels(req.dossier, str(req.horizon), "error").inc()
        logger.exception("échec run_prediction", extra={"dossier": req.dossier, "horizon": req.horizon})
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            "échec du calcul de prévision") from exc

    metrics.PREDICTIONS.labels(req.dossier, str(req.horizon), "ok").inc()
    if resolved.kge_stacking is not None:
        metrics.MODEL_KGE.labels(req.dossier, str(req.horizon)).set(resolved.kge_stacking)

    points = [
        DebitPoint(lead=i, q_stacking_m3s=round(s, 3), q_entrant_m3s=round(e, 3))
        for i, (s, e) in enumerate(zip(result["q_stacking_m3s"], result["q_entrant_m3s"]))
    ]
    return PredictResponse(
        dossier=req.dossier,
        horizon=req.horizon,
        now=result["now"],
        source=req.source,
        served_version=resolved.version,
        served_from=resolved.origin,
        points=points,
    )
