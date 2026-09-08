from __future__ import annotations

import json

import numpy as np
import pandas as pd

from previ_r2d2.common import config
from previ_r2d2.preprocessing.data_preparation.dossier_window import build_dossier, to_hourly


def test_to_hourly_resamples_non_empty_series():
    index = pd.date_range("2026-01-01", periods=120, freq="15min")
    s = pd.Series(np.arange(120, dtype=float), index=index)

    result = to_hourly(s)

    assert result.index.freq is not None or len(result) == 30
    assert len(result) == 30


def test_to_hourly_returns_empty_series_unchanged():
    s = pd.Series(dtype=float)

    result = to_hourly(s)

    assert result.empty


def test_build_dossier_assembles_debit_amont_meteo(tmp_path, monkeypatch):
    centrales_dir = tmp_path / "centrales"
    nas_data_root = tmp_path / "nas_data"
    nas_meteo = tmp_path / "nas_meteo"
    (centrales_dir / "test_centrale").mkdir(parents=True)
    (nas_data_root / "test_centrale").mkdir(parents=True)

    monkeypatch.setattr(config, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(config, "NAS_DATA_ROOT", nas_data_root)
    monkeypatch.setattr(config, "NAS_METEO", nas_meteo)

    # débit : station de référence (index tz-naive -- read_debit_csv renvoie
    # toujours un index tz-naive après tz_localize(None), comparer une borne
    # tz-aware planterait sur df.index >= start)
    index = pd.date_range("2026-01-01", periods=48, freq="1h")
    debit_df = pd.DataFrame({"Date (TU)": index.strftime("%Y-%m-%dT%H:%M:%SZ"), "Valeur (en m³/s)": np.arange(48, dtype=float)})
    debit_path = centrales_dir / "test_centrale" / "O1234567890.csv"
    debit_df.to_csv(debit_path, sep=";", index=False)

    bv_json = {"stations_meteo_nwp": []}
    (centrales_dir / "test_centrale" / "bv.json").write_text(json.dumps(bv_json), encoding="utf-8")

    rec = {"dossier": "test_centrale", "flex_strategy": "DEFAULT", "station_vigicrue_reference": "O1234567890"}

    result = build_dossier(rec, index[0], index[-1])

    assert "debit_m3s" in result.columns
    assert not result.empty
