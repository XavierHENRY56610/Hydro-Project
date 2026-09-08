"""Tests boîte noire de l'API de service (phase 03).

Le calcul de prévision (`run_prediction`) et la résolution du modèle
(`model_registry.resolve`) sont mockés : on teste ici le contrat HTTP —
routage, validation, codes d'erreur, X-Request-ID, auth, métriques — pas le
modèle (couvert par tests/model/pipeline/test_predict_orchestrator.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from previ_r2d2.common import config
from previ_r2d2.serving import api
from previ_r2d2.serving.model_registry import ResolvedModel


@pytest.fixture
def centrales(tmp_path, monkeypatch):
    centrales_dir = tmp_path / "centrales"
    models_dir = tmp_path / "models"
    (centrales_dir / "touzac_g2_G2").mkdir(parents=True)
    (centrales_dir / "touzac_g2_G2" / "bv.json").write_text(
        json.dumps({"exutoire": {"lat": 44.0, "lon": 1.2}}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)
    return tmp_path


@pytest.fixture
def client(centrales, monkeypatch):
    monkeypatch.setattr(config.settings, "api_auth_required", False)
    monkeypatch.setattr(config.settings, "api_jwt_secret", "")

    def fake_resolve(dossier, horizon):
        return ResolvedModel(Path("/fake"), "local", 3, 0.81)

    def fake_run_prediction(dossier, horizon, exutoire, bv_json, now, source="live", weights_dir=None):
        return {
            "centrale": dossier, "horizon": horizon, "now": "2026-09-07T12:00:00",
            "csv_path": "/tmp/x.csv",
            "q_stacking_m3s": [1.1, 1.2, 1.3][:_steps(horizon)],
            "q_entrant_m3s": [1.0, 1.1, 1.2][:_steps(horizon)],
        }

    monkeypatch.setattr(api.model_registry, "resolve", fake_resolve)
    monkeypatch.setattr(api, "run_prediction", fake_run_prediction)
    with TestClient(api.app) as c:
        yield c


def _steps(horizon: int) -> int:
    return {8: 8, 48: 2, 72: 3}[horizon]


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mlflow"] in {"up", "down", "disabled"}
    assert "X-Request-ID" in r.headers


def test_metrics_prometheus_format(client):
    client.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 72})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "previ_api_requests_total" in r.text
    assert "previ_api_predictions_total" in r.text


def test_predict_happy_path(client):
    r = client.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 72, "source": "frozen"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["dossier"] == "touzac_g2_G2"
    assert body["served_from"] == "local"
    assert body["served_version"] == 3
    assert len(body["points"]) == 3
    assert body["points"][0]["lead"] == 0


def test_predict_unknown_centrale_404(client):
    r = client.post("/predict", json={"dossier": "pas_une_centrale", "horizon": 72})
    assert r.status_code == 404
    assert "request_id" in r.json()


def test_predict_invalid_horizon_422(client):
    r = client.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 999})
    assert r.status_code == 422


def test_predict_invalid_source_422(client):
    r = client.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 72, "source": "bogus"})
    assert r.status_code == 422


def test_predict_no_model_404(client, monkeypatch):
    def raise_missing(dossier, horizon):
        raise FileNotFoundError(f"{dossier} h{horizon} : aucun modèle en production")

    monkeypatch.setattr(api.model_registry, "resolve", raise_missing)
    r = client.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 8})
    assert r.status_code == 404


def test_request_id_is_echoed(client):
    r = client.get("/health", headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"


def test_forecast_404_when_nothing_archived(client):
    r = client.get("/forecast/touzac_g2_G2")
    assert r.status_code == 404


def test_forecast_returns_prevision_json(client, centrales):
    prevision = {"centrale": "touzac_g2_G2", "is-current": True, "run": {"points": []}}
    (centrales / "centrales" / "touzac_g2_G2" / "prevision.json").write_text(
        json.dumps(prevision), encoding="utf-8"
    )
    r = client.get("/forecast/touzac_g2_G2")
    assert r.status_code == 200
    assert r.json()["prevision"]["centrale"] == "touzac_g2_G2"


def test_models_lists_local_versions(client, centrales):
    vdir = centrales / "models" / "touzac_g2_G2" / "h72"
    vdir.mkdir(parents=True)
    (vdir / "version.json").write_text(
        json.dumps({"version": 5, "kge_stacking": 0.77, "promoted_at": "2026-09-01T00:00:00"}),
        encoding="utf-8",
    )
    r = client.get("/models")
    assert r.status_code == 200
    models = r.json()["models"]
    assert len(models) == 1
    assert models[0]["version"] == 5
    assert models[0]["origin"] == "local"


def test_auth_required_rejects_missing_token(centrales, monkeypatch):
    monkeypatch.setattr(config.settings, "api_auth_required", True)
    monkeypatch.setattr(config.settings, "api_jwt_secret", "s3cr3t-key-of-32-bytes-minimum-len")
    with TestClient(api.app) as c:
        r = c.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 72})
        assert r.status_code == 401
        # /health reste public
        assert c.get("/health").status_code == 200


def test_auth_required_accepts_valid_token(centrales, monkeypatch):
    import jwt

    monkeypatch.setattr(config.settings, "api_auth_required", True)
    monkeypatch.setattr(config.settings, "api_jwt_secret", "s3cr3t-key-of-32-bytes-minimum-len")
    monkeypatch.setattr(api.model_registry, "resolve",
                        lambda d, h: ResolvedModel(Path("/fake"), "local", 1, 0.5))
    monkeypatch.setattr(api, "run_prediction",
                        lambda *a, **k: {"now": "2026-09-07T12:00:00",
                                         "q_stacking_m3s": [1.0, 1.0, 1.0], "q_entrant_m3s": [1.0, 1.0, 1.0]})
    token = jwt.encode({"sub": "sebastien"}, "s3cr3t-key-of-32-bytes-minimum-len", algorithm="HS256")
    with TestClient(api.app) as c:
        r = c.post("/predict", json={"dossier": "touzac_g2_G2", "horizon": 72},
                   headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200, r.text
