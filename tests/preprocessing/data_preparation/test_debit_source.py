from __future__ import annotations

import pytest

from previ_r2d2.common import config
from previ_r2d2.preprocessing.data_preparation.debit_source import debit_series


def test_debit_series_reads_debit_automate_csv_for_haute_chute(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    path = tmp_path / "melles" / "debit_automate.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-08T23:00:00Z;1.650\n", encoding="utf-8-sig"
    )
    rec = {"dossier": "melles", "flex_strategy": "HAUTE_CHUTE"}

    result = debit_series(rec)

    assert result.tolist() == pytest.approx([1.650])


def test_debit_series_reads_reference_station_csv_for_default(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    path = tmp_path / "touzac_g2_G2" / "O823153001.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Date (TU);Valeur (en m³/s)\n2026-07-08T23:00:00Z;4.200\n", encoding="utf-8-sig"
    )
    rec = {
        "dossier": "touzac_g2_G2",
        "flex_strategy": "DEFAULT",
        "station_vigicrue_reference": "O823153001",
    }

    result = debit_series(rec)

    assert result.tolist() == pytest.approx([4.200])


def test_debit_series_returns_empty_when_no_reference_station_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NAS_DATA_ROOT", tmp_path)
    rec = {"dossier": "incomplet", "flex_strategy": "DEFAULT", "station_vigicrue_reference": ""}

    result = debit_series(rec)

    assert result.empty


def test_debit_series_returns_empty_when_reference_csv_not_yet_local(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path)
    rec = {
        "dossier": "touzac_g2_G2",
        "flex_strategy": "DEFAULT",
        "station_vigicrue_reference": "O823153001",
    }

    result = debit_series(rec)

    assert result.empty
