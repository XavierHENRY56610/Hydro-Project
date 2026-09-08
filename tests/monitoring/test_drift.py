from __future__ import annotations

import numpy as np
import pandas as pd

from previ_r2d2.monitoring import drift


def _frame(loc: float, n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "debit_m3s": np.clip(loc + rng.normal(0, 1, n), 0.1, None),
            "debit_amont_O1": np.clip(loc / 2 + rng.normal(0, 0.5, n), 0, None),
            "temperature_pt1": 285 + rng.normal(0, 2, n),
        },
        index=idx,
    )


def test_no_drift_when_same_distribution():
    ref = _frame(10.0, seed=1)
    cur = _frame(10.0, seed=2)
    result = drift.run_drift_report(ref, cur)
    assert not result.drift_detected
    assert result.drift_share < drift.DRIFT_SHARE_THRESHOLD


def test_drift_detected_on_shifted_distribution():
    ref = _frame(10.0, seed=1)
    cur = _frame(40.0, seed=2)  # toutes les colonnes décalées
    result = drift.run_drift_report(ref, cur)
    assert result.drift_detected
    assert result.drift_share >= drift.DRIFT_SHARE_THRESHOLD


def test_empty_inputs_return_no_drift():
    result = drift.run_drift_report(pd.DataFrame(), pd.DataFrame())
    assert result.n_columns == 0
    assert not result.drift_detected


def test_ks_fallback_used_when_evidently_forced_off(monkeypatch):
    monkeypatch.setattr(drift, "_evidently_drift", lambda *a, **k: (_ for _ in ()).throw(ImportError("nope")))
    result = drift.run_drift_report(_frame(10.0, seed=1), _frame(40.0, seed=2))
    assert result.method == "ks"
    assert result.drift_detected


def test_as_metrics_shape():
    m = drift.run_drift_report(_frame(10.0, seed=1), _frame(10.0, seed=2)).as_metrics()
    assert set(m) == {"drift_n_columns", "drift_n_drifted", "drift_share", "drift_detected"}
