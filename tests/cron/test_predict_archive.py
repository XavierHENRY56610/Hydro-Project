from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "predict_archive_script", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "predict-archive.py"
)
predict_archive_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(predict_archive_script)


def test_run_skips_dossiers_without_production_model(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg_mod, "ARCHIVE_ROOT", tmp_path / "ARCHIVE")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    dossier_dir = tmp_path / "centrales" / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "bv.json").write_text("{}", encoding="utf-8")

    exit_code = predict_archive_script.run()

    assert exit_code == 0


def test_run_archives_then_predicts_when_model_exists(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg_mod, "ARCHIVE_ROOT", tmp_path / "ARCHIVE")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    dossier_dir = tmp_path / "centrales" / "apas_G1_G4"
    dossier_dir.mkdir(parents=True)
    (dossier_dir / "bv.json").write_text(json.dumps({"exutoire": {"lat": 43.1, "lon": 0.9}}), encoding="utf-8")
    (dossier_dir / "prevision.json").write_text('{"old": true}', encoding="utf-8")
    model_dir = tmp_path / "models" / "apas_G1_G4" / "h8"
    model_dir.mkdir(parents=True)
    (model_dir / "version.json").write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        predict_archive_script, "run_prediction",
        lambda dossier, horizon, exutoire, bv_json, now: calls.append((dossier, horizon)) or {
            "q_entrant_m3s": [1.0], "q_stacking_m3s": [1.0],
        },
    )

    exit_code = predict_archive_script.run()

    assert exit_code == 0
    assert ("apas_G1_G4", 8) in calls
    archived_files = list((tmp_path / "ARCHIVE" / "apas_G1_G4").rglob("*.json"))
    assert len(archived_files) == 1


def test_run_continues_after_one_dossier_fails(tmp_path, monkeypatch, caplog):
    """Une centrale dont la prédiction échoue (données corrompues, bug
    ponctuel...) ne doit jamais empêcher la prédiction des autres centrales
    ayant un modèle en production ce cycle horaire -- même isolation par
    (dossier, horizon) que train.py/onboarding-check.py::run()."""
    from previ_r2d2.common import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(cfg_mod, "ARCHIVE_ROOT", tmp_path / "ARCHIVE")
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    for dossier in ("centrale_en_panne", "centrale_ok"):
        dossier_dir = tmp_path / "centrales" / dossier
        dossier_dir.mkdir(parents=True)
        (dossier_dir / "bv.json").write_text(json.dumps({"exutoire": {"lat": 43.1, "lon": 0.9}}), encoding="utf-8")
        model_dir = tmp_path / "models" / dossier / "h8"
        model_dir.mkdir(parents=True)
        (model_dir / "version.json").write_text("{}", encoding="utf-8")

    calls = []

    def fake_run_prediction(dossier, horizon, exutoire, bv_json, now):
        if dossier == "centrale_en_panne":
            raise ValueError("données corrompues")
        calls.append((dossier, horizon))
        return {"q_entrant_m3s": [1.0], "q_stacking_m3s": [1.0]}

    monkeypatch.setattr(predict_archive_script, "run_prediction", fake_run_prediction)

    with caplog.at_level("INFO"):
        exit_code = predict_archive_script.run()

    assert exit_code == 1  # au moins un échec -> code de retour non-nul
    assert ("centrale_ok", 8) in calls
    assert "centrale_en_panne" in caplog.text and "ÉCHEC" in caplog.text
