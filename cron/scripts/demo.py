#!/usr/bin/env python3
"""demo — bout-en-bout sur une centrale synthétique (phase 08).

Sans accès aux vraies données (DagsHub), fabrique une centrale `demo_centrale`
puis déroule : entraînement -> promotion (models/ + tag git + alias MLflow
@production) -> prédiction servie depuis le registry -> état de monitoring.

Idempotent : réécrit `demo_centrale` à chaque exécution. À lancer dans le
conteneur `trainer` (MLflow branché) :

    docker compose run --rm trainer python cron/scripts/demo.py
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.orchestrator import run_training
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction
from previ_r2d2.model.pipeline.promotion import evaluate_candidate_vs_production, promote_model
from previ_r2d2.model.tracking import mlflow_tracking

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("demo")

DOSSIER = "demo_centrale"
HORIZON = 72
EXUTOIRE = {"lat": 43.13, "lon": 0.92}
BV_JSON = {
    "exutoire": EXUTOIRE,
    "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
    "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
    "stations_hydrometriques": [],
    "transit_vers_centrale_h": {"DJF": 2, "MAM": 2, "JJA": 2, "SON": 2},
}


def _write_synthetic_centrale() -> None:
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv

    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=450, freq="D")
    df = pd.DataFrame(
        {
            "debit_m3s": np.clip(10 + np.cumsum(rng.normal(0, 0.2, 450)), 1, None),
            "latitude_S1": 43.1, "longitude_S1": 0.9,
            "temperature_S1": 280 + 5 * np.sin(2 * np.pi * np.arange(450) / 365),
            "precipitation_S1": np.cumsum(rng.uniform(0, 2, 450)),
            "niveau0_S1": 1500.0,
        },
        index=idx,
    )
    base = config.CENTRALES_DIR / DOSSIER
    write_data_preparation_csv(df, base / "data_preparation.csv")
    (base / "bv.json").write_text(json.dumps(BV_JSON), encoding="utf-8")

    ref = config.REFERENCE_DIR / "config-general.json"
    records = json.loads(ref.read_text(encoding="utf-8")) if ref.exists() else []
    records = [r for r in records if r.get("dossier") != DOSSIER]
    records.append({"dossier": DOSSIER, "flex_strategy": "DEFAULT", "facteur_debit": 0.95})
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("centrale synthétique %s écrite (%d lignes)", DOSSIER, len(df))


def main() -> int:
    logger.info("=== 1/4 · centrale synthétique ===")
    _write_synthetic_centrale()

    logger.info("=== 2/4 · entraînement (rapide) + enregistrement MLflow ===")
    candidate_dir = config.ROOT / "weights" / "hybrid_candidate" / DOSSIER / f"h{HORIZON}"
    results = run_training(
        DOSSIER, HORIZON, EXUTOIRE, BV_JSON,
        epochs=2, n_trials_lgbm=0, n_trials_final=0, weights_dir=candidate_dir, register=True,
    )
    logger.info("KGE stacking candidat : %.3f · version MLflow : %s",
                results["kge_stacking"], results.get("mlflow_model_version"))

    logger.info("=== 3/4 · promotion (models/ + tag git + alias @production) ===")
    decision = evaluate_candidate_vs_production(DOSSIER, HORIZON, results)
    for name in ("data_preparation.csv", "bv.json"):
        (candidate_dir / name).write_bytes((config.CENTRALES_DIR / DOSSIER / name).read_bytes())
    version = promote_model(DOSSIER, HORIZON, candidate_dir, results["kge_stacking"])
    mlflow_tracking.promote_to_production(DOSSIER, HORIZON, results.get("mlflow_model_version"))
    logger.info("promu v%d (décision : %s)", version, decision["decision"])

    logger.info("=== 4/4 · prédiction servie + état monitoring ===")
    from previ_r2d2.monitoring import state
    from previ_r2d2.serving import model_registry

    resolved = model_registry.resolve(DOSSIER, HORIZON)
    logger.info("modèle résolu : origine=%s version=%s", resolved.origin, resolved.version)
    # source="frozen" -> ancrer `now` sur la dernière ligne connue du CSV
    # (comme les tests e2e), pas sur l'heure murale.
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

    frozen_now = read_data_preparation_csv(
        config.CENTRALES_DIR / DOSSIER / "data_preparation.csv"
    ).index.max()
    pred = run_prediction(DOSSIER, HORIZON, EXUTOIRE, BV_JSON, frozen_now,
                          source="frozen", weights_dir=resolved.weights_dir)
    logger.info("prévision h%d : %s", HORIZON, [round(v, 2) for v in pred["q_entrant_m3s"]])
    state.update(DOSSIER, HORIZON, online_kge=results["kge_stacking"], drift_detected=0, drift_share=0.0)

    logger.info("=== démo OK ===")
    logger.info("UI : MLflow :5000 · Prefect :4200 · Grafana :3000 · API :8000/docs · nginx :8080")
    return 0


if __name__ == "__main__":
    sys.exit(main())
