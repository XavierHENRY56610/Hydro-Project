from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.model.pipeline.plots import (
    plot_kge_by_step,
    plot_kge_radar,
    plot_ridge_coefficients,
    plot_test_comparison,
)


def make_test_comparison_fixture(n=40):
    rng = np.random.default_rng(0)
    horizon_steps = 2
    dates = pd.date_range("2026-06-01", periods=n, freq="1D")
    yt_v = 10 + np.cumsum(rng.normal(0, 0.3, n))
    pl_v = yt_v + rng.normal(0, 0.5, n)
    pt_v = yt_v + rng.normal(0, 0.5, n)
    stk_v = yt_v + rng.normal(0, 0.3, n)
    df_sub = pd.DataFrame({"debit_m3s": yt_v, "precipitation_S1": rng.uniform(0, 5, n)}, index=dates)
    df_test = pd.DataFrame(
        {"debit_m3s": rng.normal(10, 1, 200)}, index=pd.date_range("2026-01-01", periods=200, freq="1D")
    )
    valid = np.ones(n, dtype=bool)
    y_test_v = np.log1p(np.column_stack([yt_v, yt_v]))
    stk_all_v = np.column_stack([stk_v, stk_v])
    valid_pos = np.arange(160, 200)[:n]
    components = {
        "LightGBM seul": {"kge": 0.9, "r": 0.95, "alpha": 0.9, "beta": 1.0},
        "BiLSTM seul": {"kge": 0.85, "r": 0.9, "alpha": 0.95, "beta": 0.98},
        "Stacking": {"kge": 0.95, "r": 0.97, "alpha": 0.98, "beta": 1.02},
    }
    return df_sub, df_test, yt_v, pl_v, pt_v, stk_v, valid, y_test_v, stk_all_v, valid_pos, horizon_steps, components


def test_plot_test_comparison_creates_nonempty_png(tmp_path):
    df_sub, df_test, yt_v, pl_v, pt_v, stk_v, valid, y_test_v, stk_all_v, valid_pos, horizon_steps, components = (
        make_test_comparison_fixture()
    )
    output_path = tmp_path / "test.png"

    plot_test_comparison(
        df_sub, df_test, yt_v, pl_v, pt_v, stk_v, valid, y_test_v, stk_all_v, valid_pos,
        horizon_steps, "test_centrale", 48, components, output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def make_results_fixture():
    return {
        "centrale": "test_centrale",
        "horizon": 48,
        "kge_by_step": {"lgbm": [0.9, 0.85], "lstm": [0.88, 0.8], "stack": [0.95, 0.92], "rmse_m3s_stack": [0.1, 0.2]},
        "interpretability": {
            "ridge_coef": {
                "lgbm": {"mean_abs": 0.05, "max_abs": 0.1, "sign": "-"},
                "lstm": {"mean_abs": 0.1, "max_abs": 0.2, "sign": "+"},
                "ancre": {"mean_abs": 0.2, "max_abs": 0.3, "sign": "+"},
            }
        },
    }


def test_plot_kge_by_step_creates_nonempty_png(tmp_path):
    output_path = tmp_path / "kge_by_step.png"

    plot_kge_by_step(make_results_fixture(), output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridge_coefficients_creates_nonempty_png(tmp_path):
    output_path = tmp_path / "ridge.png"

    plot_ridge_coefficients(make_results_fixture(), output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_ridge_coefficients_skips_silently_when_no_ridge_coef(tmp_path):
    output_path = tmp_path / "no_ridge.png"
    results = {"centrale": "x", "horizon": 8, "interpretability": {}}

    plot_ridge_coefficients(results, output_path)

    assert not output_path.exists()


def test_plot_kge_radar_creates_nonempty_png(tmp_path):
    output_path = tmp_path / "radar.png"

    plot_kge_radar({"kge": 0.9, "r": 0.95, "alpha": 0.9, "beta": 1.0}, "LightGBM", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
