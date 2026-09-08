from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from previ_r2d2.model.features.debit_autoregressif import compute_debit_autoregressif_features


def make_df(n_hours=200, cible_col="debit_m3s"):
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    values = 10 + np.cumsum(rng.normal(0, 0.3, size=n_hours))
    return pd.DataFrame({cible_col: values}, index=index)


def make_ramp_df(n_hours=500, cible_col="debit_m3s"):
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    return pd.DataFrame({cible_col: np.arange(n_hours, dtype=float)}, index=index)


def test_compute_debit_autoregressif_features_returns_unchanged_df_when_cible_col_absent():
    df = pd.DataFrame({"autre_colonne": [1.0, 2.0, 3.0]})

    result = compute_debit_autoregressif_features(df, horizon=8, steps_per_day=24)

    pd.testing.assert_frame_equal(result, df)


def test_compute_debit_autoregressif_features_adds_max_debit_vu_and_lag_columns():
    df = make_df()

    result = compute_debit_autoregressif_features(df, horizon=8, steps_per_day=24)

    assert "max_debit_vu" in result.columns
    # base_lags=[8,12,16,24,32], weekly_lags=[168,336] (24*7, 24*14) -- aucune collision
    for lag in [8, 12, 16, 24, 32, 168, 336]:
        assert f"debit_lag_{lag}h" in result.columns
    assert result["debit_lag_8h"].equals(df["debit_m3s"].shift(8))


def test_compute_debit_autoregressif_features_gradient_accel_and_trend_columns():
    df = make_df()

    result = compute_debit_autoregressif_features(df, horizon=8, steps_per_day=24)

    assert "debit_declin_vs_max" in result.columns
    # all_lags=[8,12,16,24,32,168,336] -> gradients sur paires consécutives
    for pair in ["8v12", "12v16", "16v24", "24v32", "32v168", "168v336"]:
        assert f"debit_gradient_{pair}" in result.columns
    # accélération : nom réel avec segment du milieu répété (fidèle à Previ_v2, pas une faute)
    for accel_name in ["8v12v12v16", "12v16v16v24", "16v24v24v32", "24v32v32v168", "32v168v168v336"]:
        assert f"debit_accel_{accel_name}" in result.columns
    assert "debit_trend_rel" in result.columns


def test_compute_debit_autoregressif_features_gradient_match_hand_computed_value():
    # debit_m3s = arange(n) -> debit_lag_Lh vaut (r-L) à la ligne r, donc gradient_l1v2 = l2-l1, constant
    df = make_ramp_df()

    result = compute_debit_autoregressif_features(df, horizon=8, steps_per_day=24)

    row = 400
    assert result["debit_gradient_8v12"].iloc[row] == pytest.approx(4.0)
    assert result["debit_gradient_16v24"].iloc[row] == pytest.approx(8.0)
    assert result["debit_gradient_168v336"].iloc[row] == pytest.approx(168.0)


def test_compute_debit_autoregressif_features_declin_accel_trend_match_hand_computed_values():
    # même rampe : lag8=392, lag12=388, lag16=384, lag24=376, lag32=368, lag168=232, lag336=64 à la ligne 400
    df = make_ramp_df()

    result = compute_debit_autoregressif_features(df, horizon=8, steps_per_day=24)

    row = 400
    assert result["debit_declin_vs_max"].iloc[row] == pytest.approx(392.0 / 400.000001)
    assert result["debit_accel_8v12v12v16"].iloc[row] == pytest.approx(0.0)
    assert result["debit_accel_24v32v32v168"].iloc[row] == pytest.approx(-128.0)
    assert result["debit_trend_rel"].iloc[row] == pytest.approx(4.0 / 392.001)


def test_compute_debit_autoregressif_features_recession_and_baseflow_match_hand_computed_values():
    # steps_per_day=1 pour que les fenêtres (7j, 30j, 365j) restent petites et
    # calculables à la main ; horizon=1 pour limiter le nombre de lags.
    # pic isolé à l'index 30 (après que q95_hist devienne non-NaN à index 29,
    # min_periods=30) pour que l'ewm du lag dépasse temporairement q95_hist et
    # exerce réellement le clip de debit_baseflow_7j.
    index = pd.date_range("2026-01-01", periods=45, freq="1D")
    values = pd.Series(
        [10.0] * 10 + [20.0] * 10 + [15.0] * 10 + [50.0] + [12.0] * 14, index=index
    )
    df = pd.DataFrame({"debit_m3s": values})

    result = compute_debit_autoregressif_features(df, horizon=1, steps_per_day=1)

    # all_lags = [1, 1(=1.5*1 tronqué), 2, 3, 4, 7, 14] -> dédupliqué = [1,2,3,4,7,14]
    # récession sur (all_lags[0],all_lags[1])=(1,2) et (all_lags[2],all_lags[3])=(3,4)
    assert "debit_recession_log_1h" in result.columns
    assert "debit_recession_log_3h" in result.columns
    expected_recession_1h = np.log1p(result["debit_lag_2h"]) - np.log1p(result["debit_lag_1h"])
    pd.testing.assert_series_equal(
        result["debit_recession_log_1h"], expected_recession_1h, check_names=False
    )

    assert "debit_baseflow_7j" in result.columns
    assert "debit_baseflow_30j" in result.columns
    spd = 1
    q95_hist_expected = df["debit_m3s"].rolling(spd * 365, min_periods=spd * 30).quantile(0.95)
    expected_baseflow_7j = result["debit_lag_1h"].ewm(span=spd * 7, min_periods=spd).mean().clip(upper=q95_hist_expected)
    pd.testing.assert_series_equal(
        result["debit_baseflow_7j"], expected_baseflow_7j, check_names=False
    )

    assert "debit_ratio_rapide_base" in result.columns
    expected_ratio = result["debit_baseflow_7j"] / (result["debit_baseflow_30j"] + 1e-6)
    pd.testing.assert_series_equal(
        result["debit_ratio_rapide_base"], expected_ratio, check_names=False
    )


def test_compute_debit_autoregressif_features_anomalie_and_ratio_seasonal_match_hand_computed_values():
    index = pd.date_range("2026-01-01", periods=40, freq="1D")
    values = pd.Series([10.0] * 10 + [20.0] * 10 + [15.0] * 10 + [12.0] * 10, index=index)
    df = pd.DataFrame({"debit_m3s": values})

    result = compute_debit_autoregressif_features(df, horizon=1, steps_per_day=1)

    q_recent = result["debit_lag_1h"]
    q_roll_mean = q_recent.rolling(30, min_periods=7).mean()
    q_roll_std = q_recent.rolling(30, min_periods=7).std()
    expected_zscore = (q_recent - q_roll_mean) / (q_roll_std + 1e-6)
    expected_ratio_seasonal = q_recent / (q_roll_mean + 1e-6)

    pd.testing.assert_series_equal(
        result["debit_anomalie_zscore"], expected_zscore, check_names=False
    )
    pd.testing.assert_series_equal(
        result["debit_ratio_seasonal"], expected_ratio_seasonal, check_names=False
    )
