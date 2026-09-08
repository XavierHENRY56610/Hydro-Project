from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.model.architectures.lightgbm.predict import predict_lgbm_full, predict_lgbm_future


def make_full_df(n_hours=400):
    index = pd.date_range("2026-01-01", periods=n_hours, freq="1h")
    rng = np.random.default_rng(0)
    precip_cumul = np.cumsum(rng.uniform(0, 0.5, size=n_hours))
    debit = 10 + np.cumsum(rng.normal(0, 0.1, size=n_hours))
    return pd.DataFrame(
        {
            "debit_m3s": debit,
            "latitude_S1": 43.1,
            "longitude_S1": 0.9,
            "temperature_S1": 280.0,
            "precipitation_S1": precip_cumul,
            "niveau0_S1": 1500.0,
        },
        index=index,
    )


EXUTOIRE = {"lat": 43.13, "lon": 0.92}
BV_PARAMS = {"altitude_bv": 300, "surface_km2": 100, "k_base": 1.0, "exposition": 1.0, "kc_unit": 1.0}


class StubModel:
    def predict(self, arr):
        return np.full(len(arr), 3.5)


def test_predict_lgbm_full_returns_all_nan_when_model_is_none():
    df = make_full_df()
    lgbm_result = {"model": None, "top_features": ["debit_m3s"]}

    preds = predict_lgbm_full(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert preds.shape == (len(df),)
    assert np.all(np.isnan(preds))


def test_predict_lgbm_full_returns_all_nan_when_no_top_features():
    df = make_full_df()
    lgbm_result = {"model": StubModel(), "top_features": []}

    preds = predict_lgbm_full(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert np.all(np.isnan(preds))


def test_predict_lgbm_full_returns_all_nan_when_no_feature_available():
    df = make_full_df()
    lgbm_result = {"model": StubModel(), "top_features": ["colonne_inexistante"]}

    preds = predict_lgbm_full(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert np.all(np.isnan(preds))


def test_predict_lgbm_full_aligns_predictions_on_correct_df_positions():
    df = make_full_df()
    from previ_r2d2.model.architectures.lightgbm.features import build_features

    X, _, date = build_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})
    top_features = list(X.columns[:3])
    lgbm_result = {"model": StubModel(), "top_features": top_features}

    preds = predict_lgbm_full(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert preds.shape == (len(df),)
    assert int(np.sum(~np.isnan(preds))) == len(date)
    assert np.array_equal(np.unique(preds[~np.isnan(preds)]), np.array([3.5]))
    idx_in_df = df.index.get_indexer(date)
    assert np.all(~np.isnan(preds[idx_in_df]))


def test_predict_lgbm_future_returns_all_nan_when_model_is_none():
    df = make_full_df()
    lgbm_result = {"model": None, "top_features": ["debit_m3s"]}

    preds, future_dates = predict_lgbm_future(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert preds.shape == (8,)
    assert np.all(np.isnan(preds))
    assert len(future_dates) == 8


def test_predict_lgbm_future_nominal_returns_correct_shape_and_dates():
    df = make_full_df()
    from previ_r2d2.model.architectures.lightgbm.features import build_future_features

    X, _, date = build_future_features(df, EXUTOIRE, BV_PARAMS, steps_per_day=24, horizon=8, transit_amont={})
    top_features = list(X.columns[:3])
    lgbm_result = {"model": StubModel(), "top_features": top_features}

    preds, future_dates = predict_lgbm_future(lgbm_result, df, EXUTOIRE, BV_PARAMS, 24, 8, {})

    assert preds.shape == (8,)
    assert not np.isnan(preds).any()
    assert list(future_dates) == list(date[-8:])
