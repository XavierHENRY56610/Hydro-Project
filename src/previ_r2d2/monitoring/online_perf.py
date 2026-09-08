"""Performance en ligne (phase 06).

Quand la vérité terrain arrive (débit observé, J+1 via Hub'Eau), on rejoue
les prévisions archivées et on calcule KGE / MAE / RMSE glissants par
(dossier, horizon). C'est le complément « en production » du KGE calculé au
moment de l'entraînement (jamais recalculé ensuite jusqu'ici).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.model.architectures.stacking import kge_components

logger = logging.getLogger(__name__)


@dataclass
class OnlineScore:
    n_points: int
    kge: float | None
    mae: float | None
    rmse: float | None

    def as_metrics(self) -> dict[str, float]:
        out: dict[str, float] = {"online_n_points": self.n_points}
        if self.kge is not None:
            out["online_kge"] = round(self.kge, 4)
        if self.mae is not None:
            out["online_mae"] = round(self.mae, 4)
        if self.rmse is not None:
            out["online_rmse"] = round(self.rmse, 4)
        return out


def score_forecast_vs_observed(predicted: pd.Series, observed: pd.Series) -> OnlineScore:
    """`predicted` / `observed` : séries indexées par datetime. Aligne sur
    l'intersection des index, ignore les NaN."""
    joined = pd.DataFrame({"pred": predicted, "obs": observed}).dropna()
    if len(joined) < 3:
        return OnlineScore(len(joined), None, None, None)
    y_pred = joined["pred"].to_numpy(dtype=float)
    y_true = joined["obs"].to_numpy(dtype=float)
    err = y_pred - y_true
    kge = kge_components(y_true, y_pred)["kge"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    return OnlineScore(len(joined), kge, mae, rmse)


def _forecast_points_from_archive(dossier: str, since: pd.Timestamp) -> pd.Series:
    """Concatène les points `debit.entrant` de toutes les `prevision_*.json`
    archivées depuis `since`. Index = `target_datetime`, valeur = débit prévu.
    Dernière prévision gagnante en cas de doublon de cible."""
    root = config.ARCHIVE_ROOT / dossier
    if not root.is_dir():
        return pd.Series(dtype=float)

    records: dict[pd.Timestamp, float] = {}
    for path in sorted(root.rglob("prevision_*.json")):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        gen = pd.to_datetime(data.get("generation-date", None), errors="coerce")
        if pd.isna(gen) or gen < since:
            continue
        for point in data.get("run", {}).get("points", []):
            ts = pd.to_datetime(point.get("target_datetime"), errors="coerce")
            entrant = point.get("debit", {}).get("entrant")
            if pd.notna(ts) and entrant is not None:
                records[ts] = float(entrant)
    if not records:
        return pd.Series(dtype=float)
    return pd.Series(records).sort_index()


def _observed_debit(dossier: str) -> pd.Series:
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import read_data_preparation_csv

    df = read_data_preparation_csv(config.CENTRALES_DIR / dossier / "data_preparation.csv")
    if df.empty or "debit_m3s" not in df.columns:
        return pd.Series(dtype=float)
    return df["debit_m3s"]


def evaluate_online(dossier: str, horizon: int, *, lookback_days: int = 30) -> OnlineScore:
    """Score glissant sur les `lookback_days` derniers jours pour (dossier, horizon)."""
    since = pd.Timestamp.now().normalize() - pd.Timedelta(days=lookback_days)
    predicted = _forecast_points_from_archive(dossier, since)
    observed = _observed_debit(dossier)
    if predicted.empty or observed.empty:
        logger.info("%s h%s : pas assez d'historique prévision/observé", dossier, horizon)
        return OnlineScore(0, None, None, None)
    return score_forecast_vs_observed(predicted, observed)
