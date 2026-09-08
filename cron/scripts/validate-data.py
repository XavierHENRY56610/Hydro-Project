#!/usr/bin/env python3
"""validate-data — stage DVC `validate` de dvc/preprocessing/dvc.yaml.

Valide `data_preparation.csv` de chaque centrale (ou d'un dossier avec
`--dossier`). Erreur bloquante -> code de sortie 1 (le pipeline s'arrête avant
l'entraînement). Les warnings sont journalisés mais ne bloquent pas.
"""

from __future__ import annotations

import argparse
import logging
import sys

from previ_r2d2.common import config
from previ_r2d2.common.dvc_markers import write as write_marker
from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv
from previ_r2d2.preprocessing.data_preparation.validation import validate_data_preparation

logger = logging.getLogger("validate-data")


def discover_dossiers() -> list[str]:
    return sorted(p.parent.name for p in config.CENTRALES_DIR.glob("*/data_preparation.csv"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Valide les data_preparation.csv.")
    parser.add_argument("--dossier", default=None, help="Un seul dossier (défaut : tous).")
    args = parser.parse_args(argv)

    dossiers = [args.dossier] if args.dossier else discover_dossiers()
    if not dossiers:
        logger.warning("Aucun data_preparation.csv trouvé.")
        write_marker("validate")
        return 0

    had_error = False
    for dossier in dossiers:
        df = read_data_preparation_csv(config.CENTRALES_DIR / dossier / "data_preparation.csv")
        result = validate_data_preparation(df, strict=False, context=dossier)
        for w in result.warnings:
            logger.warning("%s : %s", dossier, w)
        if result.ok:
            logger.info("%s : OK (%d lignes)", dossier, len(df))
        else:
            had_error = True
            for e in result.errors:
                logger.error("%s : %s", dossier, e)

    write_marker("validate")
    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
