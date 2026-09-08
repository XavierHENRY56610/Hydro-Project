from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.preprocessing.data_preparation.validation import (
    DataValidationError,
    validate_data_preparation,
)


def _valid_df(n=800):
    idx = pd.date_range("2023-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "debit_m3s": np.clip(10 + np.cumsum(rng.normal(0, 0.1, n)), 0.1, None),
            "debit_amont_O123": np.clip(5 + rng.normal(0, 1, n), 0, None),
            "temperature_pt1": 285.0 + 5 * np.sin(np.arange(n) / 100),
            "precipitation_pt1": np.cumsum(rng.uniform(0, 1, n)),
        },
        index=idx,
    )


def test_valid_df_passes():
    result = validate_data_preparation(_valid_df(), strict=False)
    assert result.ok
    assert result.errors == []


def test_empty_df_is_error():
    result = validate_data_preparation(pd.DataFrame(), strict=False)
    assert not result.ok


def test_strict_raises_on_error():
    with pytest.raises(DataValidationError):
        validate_data_preparation(pd.DataFrame(), strict=True, context="apas_G1_G4")


def test_missing_target_is_error():
    df = _valid_df().drop(columns=["debit_m3s"])
    result = validate_data_preparation(df, strict=False)
    assert not result.ok
    assert any("debit_m3s" in e for e in result.errors)


def test_negative_debit_is_error():
    df = _valid_df()
    df.iloc[3, df.columns.get_loc("debit_m3s")] = -1.0
    result = validate_data_preparation(df, strict=False)
    assert not result.ok
    assert any("négative" in e for e in result.errors)


def test_unsorted_index_is_error():
    df = _valid_df().iloc[::-1]
    result = validate_data_preparation(df, strict=False)
    assert not result.ok
    assert any("trié" in e for e in result.errors)


def test_duplicate_timestamps_is_error():
    df = _valid_df()
    df = pd.concat([df, df.iloc[[10]]]).sort_index()
    result = validate_data_preparation(df, strict=False)
    assert not result.ok
    assert any("dupliqué" in e for e in result.errors)


def test_tz_aware_index_is_error():
    df = _valid_df()
    df.index = df.index.tz_localize("UTC")
    result = validate_data_preparation(df, strict=False)
    assert not result.ok
    assert any("tz-aware" in e for e in result.errors)


def test_negative_precipitation_is_error():
    df = _valid_df()
    df.iloc[5, df.columns.get_loc("precipitation_pt1")] = -2.0
    result = validate_data_preparation(df, strict=False)
    assert not result.ok


def test_long_debit_gap_is_warning_not_error():
    df = _valid_df(400)
    df.loc[df.index[100:300], "debit_m3s"] = np.nan
    result = validate_data_preparation(df, strict=False, max_debit_gap_hours=48)
    assert result.ok
    assert any("trou de débit" in w for w in result.warnings)


def test_short_history_is_warning():
    result = validate_data_preparation(_valid_df(200), strict=False, min_history_days=365)
    assert result.ok
    assert any("historique" in w for w in result.warnings)


def test_implausible_temperature_is_warning():
    df = _valid_df()
    df["temperature_pt1"] = 25.0  # °C au lieu de K
    result = validate_data_preparation(df, strict=False)
    assert result.ok
    assert any("Kelvin" in w for w in result.warnings)
