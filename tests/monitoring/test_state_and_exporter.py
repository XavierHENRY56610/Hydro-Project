from __future__ import annotations

import json

from previ_r2d2.common import config
from previ_r2d2.monitoring import exporter, state


def test_state_update_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    state.update("apas_G1_G4", 8, online_kge=0.71, drift_detected=0)
    state.update("apas_G1_G4", 8, drift_share=0.12)  # fusion
    state.update("nancy_A", 48, online_kge=0.4)

    data = state.read_all()
    assert data["apas_G1_G4/h8"]["online_kge"] == 0.71  # posé au 1er appel, conservé
    assert data["apas_G1_G4/h8"]["drift_share"] == 0.12  # ajouté au 2e appel
    assert "updated_at" in data["apas_G1_G4/h8"]
    assert data["nancy_A/h48"]["online_kge"] == 0.4


def test_state_read_all_empty_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    assert state.read_all() == {}


def test_write_demo_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    state.update("apas_G1_G4", 8, online_kge=0.8)
    state.write_demo_alert()
    entry = state.read_all()["apas_G1_G4/h8"]
    assert entry["online_kge"] == -0.5
    assert entry["drift_detected"] == 1


def test_exporter_collector_emits_gauges(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(config, "CENTRALES_DIR", tmp_path / "centrales")

    vdir = tmp_path / "models" / "apas_G1_G4" / "h8"
    vdir.mkdir(parents=True)
    (vdir / "version.json").write_text(json.dumps({"version": 3, "kge_stacking": 0.82}), encoding="utf-8")
    (tmp_path / "centrales" / "apas_G1_G4").mkdir(parents=True)
    (tmp_path / "centrales" / "apas_G1_G4" / "prevision.json").write_text("{}", encoding="utf-8")
    state.update("apas_G1_G4", 8, online_kge=0.7, drift_detected=1, drift_share=0.6)

    families = {mf.name: mf for mf in exporter.PreviCollector().collect()}
    assert families["previ_model_kge"].samples[0].value == 0.82
    assert families["previ_model_version"].samples[0].value == 3.0
    assert families["previ_online_kge"].samples[0].value == 0.7
    assert families["previ_data_drift_detected"].samples[0].value == 1.0
    assert families["previ_hours_since_last_forecast"].samples[0].labels == {"dossier": "apas_G1_G4"}
