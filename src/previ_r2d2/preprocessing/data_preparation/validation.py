"""Contrat de validation de `data_preparation.csv` (phase 02 MLOps).

Contrôle explicite (pas de pandera : colonnes dynamiques par centrale +
distinction erreurs/warnings) de ce qu'un entraînement suppose sans jamais le
vérifier aujourd'hui : index horaire trié et sans doublon, `debit_m3s` (cible)
présente et positive, colonnes débit/précipitation non négatives, pas de trou
de débit trop long, historique suffisant. Les colonnes météo restant figées
(acquisition FTP retirée), leur absence / leurs NaN sont tolérés.

`validate_data_preparation(df)` renvoie un `ValidationResult` ; en mode strict
il lève `DataValidationError` dès la première erreur bloquante. Les *warnings*
ne bloquent jamais (ex. trou de débit long mais historique par ailleurs sain).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

TARGET_COL = "debit_m3s"
MAX_DEBIT_GAP_HOURS = 168          # trou de débit > 1 semaine => warning
MIN_HISTORY_DAYS = 365            # < 12 mois => warning (l'éligibilité bloque déjà)
TEMPERATURE_KELVIN_RANGE = (180.0, 340.0)


class DataValidationError(ValueError):
    """Levée par `validate_data_preparation(..., strict=True)` sur erreur bloquante."""


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_failed(self, context: str = "") -> None:
        if not self.ok:
            prefix = f"{context} : " if context else ""
            raise DataValidationError(prefix + " ; ".join(self.errors))


def _is_debit_col(col: str) -> bool:
    return col == TARGET_COL or col.startswith("debit_amont")


def validate_data_preparation(
    df: pd.DataFrame,
    *,
    max_debit_gap_hours: int = MAX_DEBIT_GAP_HOURS,
    min_history_days: int = MIN_HISTORY_DAYS,
    strict: bool = True,
    context: str = "",
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or df.empty:
        result = ValidationResult(ok=False, errors=["data_preparation vide"])
        if strict:
            result.raise_if_failed(context)
        return result

    # --- index ---
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append(f"index de type {type(df.index).__name__}, DatetimeIndex attendu")
    else:
        if df.index.tz is not None:
            errors.append("index tz-aware (attendu : tz-naive UTC)")
        if df.index.has_duplicates:
            n = int(df.index.duplicated().sum())
            errors.append(f"{n} horodatage(s) dupliqué(s)")
        if not df.index.is_monotonic_increasing:
            errors.append("index non trié par ordre chronologique")

    # --- cible ---
    if TARGET_COL not in df.columns:
        errors.append(f"colonne cible '{TARGET_COL}' absente")
    else:
        debit = pd.to_numeric(df[TARGET_COL], errors="coerce")
        valid = debit.dropna()
        if valid.empty:
            errors.append(f"'{TARGET_COL}' entièrement vide")
        else:
            if (valid < 0).any():
                errors.append(f"'{TARGET_COL}' contient {int((valid < 0).sum())} valeur(s) négative(s)")
            if isinstance(df.index, pd.DatetimeIndex):
                covered = df.index[debit.notna().to_numpy()]
                if len(covered) >= 2:
                    largest_gap = covered.to_series().diff().max()
                    if largest_gap > pd.Timedelta(hours=max_debit_gap_hours):
                        warnings.append(
                            f"trou de débit de {largest_gap} (> {max_debit_gap_hours} h)"
                        )
                    span_days = (covered[-1] - covered[0]).days
                    if span_days < min_history_days:
                        warnings.append(f"historique de débit de {span_days} j (< {min_history_days} j)")

    # --- colonnes numériques : signe / plage ---
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        if _is_debit_col(col) and col != TARGET_COL and (series < 0).any():
            errors.append(f"'{col}' contient des valeurs négatives")
        if "precipitation" in col.lower() and (series < 0).any():
            errors.append(f"'{col}' (précipitation) contient des valeurs négatives")
        if "temperature" in col.lower():
            lo, hi = TEMPERATURE_KELVIN_RANGE
            if series.min() < lo or series.max() > hi:
                warnings.append(
                    f"'{col}' hors plage Kelvin plausible [{lo}, {hi}] "
                    f"(min={series.min():.1f}, max={series.max():.1f})"
                )

    result = ValidationResult(ok=not errors, errors=errors, warnings=warnings)
    if strict:
        result.raise_if_failed(context)
    return result
