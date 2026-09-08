"""Point d'entrée CLI -- miroir de run.py/app.py (Previ_v2), mais délègue
toute la logique ici (testable) plutôt que dans le script racine."""

from __future__ import annotations

import argparse
import json
import logging
import sys

import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.pipeline.orchestrator import HORIZON_CFG, run_training
from previ_r2d2.model.pipeline.predict_orchestrator import run_prediction

logger = logging.getLogger("previ-r2d2-cli")

HORIZONS = sorted(HORIZON_CFG.keys())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse les arguments CLI ; --horizon requis avec --dossier, pas avec --all-dossiers."""
    parser = argparse.ArgumentParser(description="previ-R2-D2 -- entraînement et prédiction du modèle hybride.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Entraîne le modèle hybride.")
    mode.add_argument("--predict", action="store_true", help="Prédiction opérationnelle.")

    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--dossier", type=str, default=None, metavar="NOM", help="Nom exact du dossier (centrales/).")
    scope.add_argument("--all-dossiers", action="store_true", help="Toutes les centrales x horizons 8/48/72.")

    parser.add_argument("--horizon", type=int, default=None, choices=HORIZONS, help="Horizon (requis avec --dossier).")
    parser.add_argument("--meta", type=str, default="ridge", choices=["ridge", "lgbm"], help="Meta-learner.")
    parser.add_argument("--epochs", type=int, default=40, metavar="N")
    parser.add_argument("--n-trials-lgbm", type=int, default=30, metavar="N")
    parser.add_argument("--n-trials-final", type=int, default=50, metavar="N")
    parser.add_argument("--force-lgbm", action="store_true")
    parser.add_argument("--force-lstm", action="store_true")

    args = parser.parse_args(argv)
    if args.dossier and args.horizon is None:
        parser.error("--horizon est requis avec --dossier")
    return args


def discover_dossiers() -> list[str]:
    """Liste triée des dossiers ayant un bv.json (sous centrales/)."""
    return sorted(p.parent.name for p in config.CENTRALES_DIR.glob("*/bv.json"))


def load_bv_json(dossier: str) -> dict:
    """Charge bv.json pour un dossier."""
    path = config.CENTRALES_DIR / dossier / "bv.json"
    with open(path) as fh:
        return json.load(fh)


def train_command(args: argparse.Namespace) -> int:
    """Entraîne un dossier+horizon ou tous les dossiers x tous les horizons ; continue après erreur, retourne 1 si au moins un échec."""
    if args.all_dossiers:
        pairs = [(d, h) for d in discover_dossiers() for h in HORIZONS]
    else:
        pairs = [(args.dossier, args.horizon)]

    had_error = False
    for dossier, horizon in pairs:
        try:
            bv_json = load_bv_json(dossier)
            exutoire = bv_json["exutoire"]
            logger.info("Entraînement %s h%s...", dossier, horizon)
            run_training(
                dossier, horizon, exutoire, bv_json,
                meta_type=args.meta, epochs=args.epochs,
                n_trials_lgbm=args.n_trials_lgbm, n_trials_final=args.n_trials_final,
                force_lgbm=args.force_lgbm, force_lstm=args.force_lstm,
                register=False,  # run.py = expés manuelles : run MLflow oui, registry non
            )
        except Exception as exc:
            logger.error("Échec %s h%s : %s", dossier, horizon, exc, exc_info=True)
            had_error = True
    return 1 if had_error else 0


def predict_command(args: argparse.Namespace) -> int:
    """Prédit un dossier+horizon ou tous les dossiers x tous les horizons ; continue après erreur, retourne 1 si au moins un échec."""
    now = pd.Timestamp.now().floor("h")
    if args.all_dossiers:
        pairs = [(d, h) for d in discover_dossiers() for h in HORIZONS]
    else:
        pairs = [(args.dossier, args.horizon)]

    had_error = False
    for dossier, horizon in pairs:
        try:
            bv_json = load_bv_json(dossier)
            exutoire = bv_json["exutoire"]
            logger.info("Prédiction %s h%s...", dossier, horizon)
            run_prediction(dossier, horizon, exutoire, bv_json, now)
        except Exception as exc:
            logger.error("Échec %s h%s : %s", dossier, horizon, exc, exc_info=True)
            had_error = True
    return 1 if had_error else 0


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée : configure le logging, parse les arguments, dispatche vers predict_command ou train_command."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args(argv)
    if args.predict:
        return predict_command(args)
    return train_command(args)


if __name__ == "__main__":
    sys.exit(main())
