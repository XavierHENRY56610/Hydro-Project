#!/usr/bin/env python3
"""train — stages DVC dvc/model/dvc.yaml:train_new et :train_monthly.

Deux cadences distinctes, toutes deux appelant `train_one` :
  - `run_new_dossiers` (quotidien) : uniquement les dossiers n'ayant ENCORE
    aucun modèle en production (nouvelle centrale) -- dès qu'ils atteignent
    12 mois d'historique, entraînés le jour même (pas d'attente jusqu'au
    prochain cycle mensuel).
  - `run_monthly_retrain` (mensuel) : tous les dossiers ayant DÉJÀ un modèle
    en production -- réentraîne inconditionnellement (l'invocation mensuelle
    EST l'échéance), promeut le nouveau modèle seulement s'il est meilleur
    (KGE Stacking) que le modèle en production sur le même holdout, cf.
    promotion.py. Versionne aussi data_preparation.csv + bv.json avec le
    modèle (même tag git) pour une reproductibilité exacte de chaque version
    promue.

Avant d'entraîner un dossier (dans les deux modes), rafraîchit son
data_preparation.csv -- un seul dossier à la fois, jamais toutes les
centrales (cf. refresh_data_preparation) : is_eligible_for_training/
history_span_days lisent ce fichier pour l'historique 12 mois, donc sans ce
rafraîchissement ciblé une toute nouvelle centrale ne l'aurait jamais
(fichier jamais généré = 0 jour d'historique pour toujours, blocage
permanent, pas juste un délai).

`run(dossier=, horizon=, force=)` reste disponible pour un test manuel ciblé
sur une seule centrale/horizon (cf. commandes de test rapide du skill).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path

from previ_r2d2.common import config
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.model.pipeline.eligibility import (
    MIN_HISTORY_DAYS,
    has_production_model,
    history_span_days,
    is_eligible_for_training,
)
from previ_r2d2.model.pipeline.orchestrator import HORIZON_CFG, run_training
from previ_r2d2.model.pipeline.promotion import evaluate_candidate_vs_production, promote_model
from previ_r2d2.model.tracking import mlflow_tracking

logger = logging.getLogger("train")

_BUILD_DATA_PREPARATION_PATH = Path(__file__).resolve().parent / "build-data-preparation.py"
_bdp_spec = importlib.util.spec_from_file_location("build_data_preparation_script", _BUILD_DATA_PREPARATION_PATH)
build_data_preparation_script = importlib.util.module_from_spec(_bdp_spec)
_bdp_spec.loader.exec_module(build_data_preparation_script)

# Défauts de production (plus exigeants que les défauts de run_training,
# calibrés pour un entraînement réel plutôt que pour des tests rapides).
DEFAULT_EPOCHS = 100
DEFAULT_N_TRIALS_LGBM = 100
DEFAULT_N_TRIALS_FINAL = 100


def discover_dossiers() -> list[str]:
    return sorted(p.parent.name for p in config.CENTRALES_DIR.glob("*/bv.json"))


def load_bv_json(dossier: str) -> dict:
    with open(config.CENTRALES_DIR / dossier / "bv.json") as fh:
        return json.load(fh)


def refresh_data_preparation(dossier: str) -> None:
    """Rafraîchit data_preparation.csv pour CE dossier uniquement (jamais
    toutes les centrales -- ce serait le stage DVC data_preparation complet,
    coûteux et pas nécessaire ici) avant de vérifier son éligibilité/de
    l'entraîner."""
    build_data_preparation_script.run(only_dossier=dossier)


def train_one(
    dossier: str,
    horizon: int,
    epochs: int = DEFAULT_EPOCHS,
    n_trials_lgbm: int = DEFAULT_N_TRIALS_LGBM,
    n_trials_final: int = DEFAULT_N_TRIALS_FINAL,
    **run_training_kwargs,
) -> str:
    """Entraîne un candidat, compare au modèle en production, promeut si
    meilleur (ou si premier entraînement). Retourne une ligne pour le digest."""
    bv_json = load_bv_json(dossier)
    exutoire = bv_json["exutoire"]
    candidate_dir = config.ROOT / "weights" / "hybrid_candidate" / dossier / f"h{horizon}"
    if candidate_dir.exists():
        shutil.rmtree(candidate_dir)

    results = run_training(
        dossier, horizon, exutoire, bv_json,
        epochs=epochs, n_trials_lgbm=n_trials_lgbm, n_trials_final=n_trials_final,
        weights_dir=candidate_dir, **run_training_kwargs,
    )
    decision = evaluate_candidate_vs_production(dossier, horizon, results)

    # Snapshot data_preparation.csv + bv.json dans candidate_dir avant promotion :
    # promote_model copie tout candidate_dir vers models/.../ puis le dvc-add/tag
    # d'un bloc -- ces 2 fichiers sont donc versionnés avec le même tag git que le
    # modèle, sans mécanisme DVC séparé (reproductibilité exacte : on sait quelles
    # données/config étaient actives pour une version de modèle donnée).
    shutil.copy2(config.CENTRALES_DIR / dossier / "data_preparation.csv", candidate_dir / "data_preparation.csv")
    shutil.copy2(config.CENTRALES_DIR / dossier / "bv.json", candidate_dir / "bv.json")

    if decision["decision"] in ("promote", "first_training"):
        version = promote_model(dossier, horizon, candidate_dir, results["kge_stacking"])
        mlflow_tracking.promote_to_production(dossier, horizon, results.get("mlflow_model_version"))
        summary = (
            f"{dossier} h{horizon} : PROMU v{version} "
            f"(kge_candidat={results['kge_stacking']}, kge_prod={decision['production_kge']})"
        )
    else:
        summary = (
            f"{dossier} h{horizon} : conservé "
            f"(kge_candidat={results['kge_stacking']}, kge_prod={decision['production_kge']})"
        )

    write_marker(f"train_{dossier}_h{horizon}")
    return summary


def run(dossier: str | None = None, horizon: int | None = None, force: bool = False) -> int:
    """Test manuel ciblé : une seule centrale/horizon (`force=True` ignore
    l'éligibilité, pour pouvoir tester même sans historique de 12 mois ou
    avant l'échéance de réentraînement). Rafraîchit aussi data_preparation
    pour ce dossier avant d'entraîner, comme les 2 modes automatisés."""
    dossiers = [dossier] if dossier is not None else discover_dossiers()
    horizons = [horizon] if horizon is not None else sorted(HORIZON_CFG.keys())
    summaries = []
    had_error = False
    for d in dossiers:
        try:
            refresh_data_preparation(d)
        except Exception as exc:
            logger.error("Échec rafraîchissement data_preparation %s : %s", d, exc, exc_info=True)
            summaries.append(f"{d} : ÉCHEC rafraîchissement data_preparation ({exc})")
            had_error = True
            continue
        for h in horizons:
            if not force and not is_eligible_for_training(d, h):
                continue
            try:
                summaries.append(train_one(d, h))
            except Exception as exc:
                logger.error("Échec entraînement %s h%s : %s", d, h, exc, exc_info=True)
                summaries.append(f"{d} h{h} : ÉCHEC ({exc})")
                had_error = True

    body = "\n".join(summaries) if summaries else "Aucune centrale éligible aujourd'hui."
    logger.info(body)
    write_marker("train")
    return 1 if had_error else 0


def run_new_dossiers() -> int:
    """Quotidien : pour chaque dossier, uniquement les horizons SANS modèle
    en production (pas "aucun modèle sur aucun horizon" -- un dossier déjà
    entraîné sur h8 mais pas encore sur h48/h72 doit continuer de proposer
    h48/h72 ici, sinon ces horizons ne seraient jamais entraînés une première
    fois : run_monthly_retrain ne traite que les horizons DÉJÀ en prod).
    Entraîne dès que 12 mois d'historique sont atteints, sans attendre le
    cycle mensuel."""
    horizons = sorted(HORIZON_CFG.keys())
    summaries = []
    had_error = False
    for d in discover_dossiers():
        pending_horizons = [h for h in horizons if not has_production_model(d, h)]
        if not pending_horizons:
            continue  # tous les horizons ont déjà un modèle -> relève uniquement du mensuel
        try:
            refresh_data_preparation(d)
        except Exception as exc:
            logger.error("Échec rafraîchissement data_preparation %s : %s", d, exc, exc_info=True)
            summaries.append(f"{d} : ÉCHEC rafraîchissement data_preparation ({exc})")
            had_error = True
            continue
        for h in pending_horizons:
            if history_span_days(d) < MIN_HISTORY_DAYS:
                continue
            try:
                summaries.append(train_one(d, h))
            except Exception as exc:
                logger.error("Échec entraînement %s h%s : %s", d, h, exc, exc_info=True)
                summaries.append(f"{d} h{h} : ÉCHEC ({exc})")
                had_error = True

    body = "\n".join(summaries) if summaries else "Aucune nouvelle centrale à entraîner aujourd'hui."
    logger.info(body)
    write_marker("train_new")
    return 1 if had_error else 0


def run_monthly_retrain() -> int:
    """Mensuel : tous les dossiers ayant déjà un modèle en production --
    réentraîne inconditionnellement (l'invocation mensuelle est l'échéance),
    train_one() compare ensuite le candidat au modèle en prod (KGE, même
    holdout) et ne promeut que s'il est meilleur (cf. promotion.py)."""
    horizons = sorted(HORIZON_CFG.keys())
    summaries = []
    had_error = False
    for d in discover_dossiers():
        trained_horizons = [h for h in horizons if has_production_model(d, h)]
        if not trained_horizons:
            continue  # jamais encore entraînée -> relève de run_new_dossiers
        try:
            refresh_data_preparation(d)
        except Exception as exc:
            logger.error("Échec rafraîchissement data_preparation %s : %s", d, exc, exc_info=True)
            summaries.append(f"{d} : ÉCHEC rafraîchissement data_preparation ({exc})")
            had_error = True
            continue
        for h in trained_horizons:
            try:
                summaries.append(train_one(d, h))
            except Exception as exc:
                logger.error("Échec entraînement %s h%s : %s", d, h, exc, exc_info=True)
                summaries.append(f"{d} h{h} : ÉCHEC ({exc})")
                had_error = True

    body = "\n".join(summaries) if summaries else "Aucune centrale à réentraîner ce mois-ci."
    logger.info(body)
    write_marker("train_monthly")
    return 1 if had_error else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["new", "monthly"],
        help="Mode automatisé (stages DVC train_new/train_monthly).",
    )
    parser.add_argument("--dossier", help="Test manuel ciblé sur cette centrale (ignore --mode).")
    parser.add_argument("--horizon", type=int, help="Test manuel ciblé sur cet horizon (8/48/72).")
    parser.add_argument(
        "--force", action="store_true",
        help="Ignorer l'éligibilité (utile avec --dossier/--horizon pour un test manuel).",
    )
    args = parser.parse_args(argv)
    if not args.dossier and not args.mode:
        parser.error("--mode {new,monthly} ou --dossier est requis")
    return args


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    if args.dossier:
        return run(dossier=args.dossier, horizon=args.horizon, force=args.force)
    if args.mode == "new":
        return run_new_dossiers()
    return run_monthly_retrain()


if __name__ == "__main__":
    sys.exit(main())
