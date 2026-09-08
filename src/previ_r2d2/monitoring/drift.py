"""Rapport de dérive des données (phase 06).

Référence = fenêtre d'entraînement du modèle promu (`data_preparation.csv`
figé avec le modèle, cf. train.py). Courant = N derniers jours.

`run_drift_report` tente Evidently (`DataDriftPreset`) et, si la lib est
absente ou son API a bougé, retombe sur un test de Kolmogorov-Smirnov par
colonne (scipy, toujours dispo). Le résultat (part de colonnes en dérive,
drapeau) alimente `state.json` puis les gauges Prometheus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_SHARE_THRESHOLD = 0.4  # part de colonnes en dérive -> dérive « dataset »


@dataclass
class DriftResult:
    n_columns: int
    n_drifted: int
    drift_share: float
    drift_detected: bool
    method: str  # "evidently" | "ks"

    def as_metrics(self) -> dict[str, float]:
        return {
            "drift_n_columns": self.n_columns,
            "drift_n_drifted": self.n_drifted,
            "drift_share": round(self.drift_share, 4),
            "drift_detected": int(self.drift_detected),
        }


def _numeric_common_columns(reference: pd.DataFrame, current: pd.DataFrame) -> list[str]:
    return [
        c for c in reference.columns
        if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])
    ]


def _ks_drift(reference: pd.DataFrame, current: pd.DataFrame) -> DriftResult:
    from scipy import stats

    cols = _numeric_common_columns(reference, current)
    tested, drifted = 0, 0
    for col in cols:
        ref = reference[col].dropna()
        cur = current[col].dropna()
        if len(ref) < 30 or len(cur) < 30:
            continue
        tested += 1
        _, p_value = stats.ks_2samp(ref, cur)
        if p_value < 0.05:
            drifted += 1
    share = drifted / tested if tested else 0.0
    return DriftResult(tested, drifted, share, share >= DRIFT_SHARE_THRESHOLD, "ks")


def _evidently_drift(reference: pd.DataFrame, current: pd.DataFrame, html_path: Path | None) -> DriftResult:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    cols = _numeric_common_columns(reference, current)
    definition = DataDefinition(numerical_columns=cols)
    ref = Dataset.from_pandas(reference[cols], data_definition=definition)
    cur = Dataset.from_pandas(current[cols], data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(cur, ref)

    if html_path is not None:
        try:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot.save_html(str(html_path))
        except Exception as exc:  # noqa: BLE001 — l'artefact HTML est un bonus
            logger.warning("rapport HTML Evidently non écrit : %s", exc)

    n_drifted, share = 0, 0.0
    for metric in snapshot.dict().get("metrics", []):
        mid = str(metric.get("metric_id", ""))
        value = metric.get("value")
        if mid.startswith("DriftedColumnsCount") and isinstance(value, dict):
            n_drifted = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
    return DriftResult(len(cols), n_drifted, share, share >= DRIFT_SHARE_THRESHOLD, "evidently")


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    html_path: Path | None = None,
) -> DriftResult:
    """Dérive de `current` vs `reference`. Evidently si dispo, sinon KS/scipy."""
    if reference.empty or current.empty:
        return DriftResult(0, 0, 0.0, False, "ks")
    try:
        return _evidently_drift(reference, current, html_path)
    except Exception as exc:  # noqa: BLE001 — repli KS sur absence/évolution d'API Evidently
        logger.warning("Evidently indisponible (%s) — repli test KS", exc)
        return _ks_drift(reference, current)
