from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from previ_r2d2.model.pipeline.artifacts import build_results, write_artifacts


def make_test_predictions(n=40):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2026-06-01", periods=n, freq="1D")
    yt_v = 10 + np.cumsum(rng.normal(0, 0.3, n))
    pl_v = yt_v + rng.normal(0, 0.5, n)
    pt_v = yt_v + rng.normal(0, 0.5, n)
    stk_v = yt_v + rng.normal(0, 0.3, n)
    q_now_v = yt_v + rng.normal(0, 0.2, n)
    df_sub = pd.DataFrame({"debit_m3s": yt_v}, index=dates)
    return yt_v, pl_v, pt_v, stk_v, q_now_v, df_sub


def test_build_results_has_expected_top_level_keys():
    yt_v, pl_v, pt_v, stk_v, q_now_v, df_sub = make_test_predictions()
    meta = Ridge(alpha=1.0).fit(np.random.default_rng(0).normal(0, 1, (40, 5)), np.random.default_rng(0).normal(0, 1, (40, 2)))
    y_test = np.column_stack([yt_v, yt_v])
    pred_lgbm_multi_test = np.column_stack([pl_v, pl_v])
    pred_lstm_test = np.column_stack([pt_v, pt_v])
    pred_stacking_multi = np.column_stack([stk_v, stk_v])

    results = build_results(
        "test_centrale", 72, "ridge", yt_v, pl_v, pt_v, stk_v, q_now_v, df_sub,
        y_test, pred_lgbm_multi_test, pred_lstm_test, pred_stacking_multi, meta, 2, [],
    )

    expected_keys = {
        "centrale", "horizon", "meta_type", "kge_lgbm", "kge_lstm", "kge_stacking",
        "components", "kge_by_step", "kge_by_regime", "kge_by_season", "quantiles_test", "interpretability",
    }
    assert expected_keys.issubset(results.keys())
    assert results["centrale"] == "test_centrale"
    assert results["horizon"] == 72


def test_write_artifacts_creates_expected_files(tmp_path):
    yt_v, pl_v, pt_v, stk_v, q_now_v, df_sub = make_test_predictions()
    meta = Ridge(alpha=1.0).fit(np.random.default_rng(0).normal(0, 1, (40, 5)), np.random.default_rng(0).normal(0, 1, (40, 2)))
    y_test = np.column_stack([yt_v, yt_v])
    pred_lgbm_multi_test = np.column_stack([pl_v, pl_v])
    pred_lstm_test = np.column_stack([pt_v, pt_v])
    pred_stacking_multi = np.column_stack([stk_v, stk_v])
    results = build_results(
        "test_centrale", 72, "ridge", yt_v, pl_v, pt_v, stk_v, q_now_v, df_sub,
        y_test, pred_lgbm_multi_test, pred_lstm_test, pred_stacking_multi, meta, 2, [],
    )
    meta_config = {"centrale": "test_centrale", "horizon": 72, "mlflow_run_id": None}
    weights_dir = tmp_path / "weights"
    outputs_dir = tmp_path / "outputs"
    weights_dir.mkdir()
    outputs_dir.mkdir()

    write_artifacts(results, meta_config, yt_v, pl_v, pt_v, stk_v, df_sub, "test_centrale", 72, weights_dir, outputs_dir)

    assert (weights_dir / "results.json").exists()
    assert (weights_dir / "meta_config.json").exists()
    csv_path = outputs_dir / "hybrid_test_test_centrale_72h.csv"
    assert csv_path.exists()

    with open(weights_dir / "results.json") as fh:
        loaded = json.load(fh)
    assert loaded["centrale"] == "test_centrale"

    df_csv = pd.read_csv(csv_path)
    assert list(df_csv.columns) == ["datetime", "q_obs_m3s", "q_lgbm_m3s", "q_lstm_m3s", "q_stacking_m3s"]
    assert len(df_csv) == len(yt_v)
