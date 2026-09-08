from __future__ import annotations

import json

import numpy as np
import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.monitoring import online_perf


def test_score_perfect_forecast():
    idx = pd.date_range("2026-01-01", periods=50, freq="h")
    obs = pd.Series(np.linspace(5, 15, 50), index=idx)
    score = online_perf.score_forecast_vs_observed(obs.copy(), obs.copy())
    assert score.n_points == 50
    assert score.kge > 0.99
    assert score.mae < 1e-6


def test_score_biased_forecast_has_lower_kge():
    idx = pd.date_range("2026-01-01", periods=50, freq="h")
    rng = np.random.default_rng(0)
    obs = pd.Series(10 + rng.normal(0, 1, 50), index=idx)
    biased = obs + 5
    score = online_perf.score_forecast_vs_observed(biased, obs)
    assert score.kge < 0.7  # bêta ~1.5 -> KGE ~0.5
    assert score.mae > 4


def test_score_too_few_points():
    idx = pd.date_range("2026-01-01", periods=2, freq="h")
    s = pd.Series([1.0, 2.0], index=idx)
    score = online_perf.score_forecast_vs_observed(s, s)
    assert score.n_points == 2
    assert score.kge is None


def test_evaluate_online_reads_archive_and_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "ARCHIVE_ROOT", tmp_path / "ARCHIVE")
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path / "centrales")

    # débit observé
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv

    idx = pd.date_range("2026-01-01", periods=200, freq="h")
    df = pd.DataFrame({"debit_m3s": np.linspace(8, 12, 200)}, index=idx)
    write_data_preparation_csv(df, tmp_path / "centrales" / "c1" / "data_preparation.csv")

    # une prévision archivée récente qui colle à l'observé
    gen = pd.Timestamp.now().floor("h")
    points = [
        {"target_datetime": idx[i].isoformat(), "debit": {"entrant": float(df["debit_m3s"].iloc[i])}}
        for i in range(20, 40)
    ]
    arch = tmp_path / "ARCHIVE" / "c1" / "2026" / "01" / "01"
    arch.mkdir(parents=True)
    (arch / "prevision_20260101_00h.json").write_text(
        json.dumps({"generation-date": gen.isoformat(), "run": {"points": points}}), encoding="utf-8"
    )

    score = online_perf.evaluate_online("c1", 8)
    assert score.n_points == 20
    assert score.kge > 0.9
