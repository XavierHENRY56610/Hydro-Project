from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_spec = importlib.util.spec_from_file_location(
    "train_script", Path(__file__).resolve().parents[2] / "cron" / "scripts" / "train.py"
)
train_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_script)


def _patch_common(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", tmp_path / "centrales")
    monkeypatch.setattr(cfg_mod, "NAS_DATA_ROOT", tmp_path / "nas")
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")
    monkeypatch.setattr(train_script, "refresh_data_preparation", lambda dossier: None)


def test_run_skips_ineligible_dossiers_and_reports_empty_digest(tmp_path, monkeypatch, caplog):
    _patch_common(tmp_path, monkeypatch)

    (tmp_path / "centrales" / "apas_G1_G4").mkdir(parents=True)
    (tmp_path / "centrales" / "apas_G1_G4" / "bv.json").write_text("{}", encoding="utf-8")

    with caplog.at_level("INFO"):
        exit_code = train_script.run()

    assert exit_code == 0
    assert "Aucune centrale éligible" in caplog.text


def test_train_one_promotes_when_no_production_model_exists(tmp_path, monkeypatch):
    from previ_r2d2.common import config as cfg_mod
    from previ_r2d2.model.pipeline import promotion
    from previ_r2d2.preprocessing.data_preparation.data_preparation_csv import write_data_preparation_csv
    from tests.model.pipeline.test_orchestrator import make_synthetic_df

    centrales_dir = tmp_path / "centrales"
    monkeypatch.setattr(cfg_mod, "ROOT", tmp_path)
    monkeypatch.setattr(cfg_mod, "CENTRALES_DIR", centrales_dir)
    monkeypatch.setattr(cfg_mod, "MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(promotion.subprocess, "run", lambda *a, **k: None)
    from previ_r2d2.common import dvc_markers

    monkeypatch.setattr(dvc_markers, "MARKERS_DIR", tmp_path / "logs" / "dvc_markers")

    dossier_dir = centrales_dir / "test_centrale"
    dossier_dir.mkdir(parents=True)
    bv_json = {
        "exutoire": {"lat": 43.13, "lon": 0.92},
        "bassin_versant": {"altitude_moyenne_m": 300.0, "surface_km2": 100.0},
        "parametres_calage": {"K_base": 1.0, "exposition": 1.0, "kc_unit": 1.0},
        "stations_hydrometriques": [],
    }
    (dossier_dir / "bv.json").write_text(json.dumps(bv_json), encoding="utf-8")
    write_data_preparation_csv(make_synthetic_df(), dossier_dir / "data_preparation.csv")

    summary = train_script.train_one("test_centrale", 72, epochs=2, n_trials_lgbm=0, n_trials_final=0)

    assert "PROMU v1" in summary
    prod_dir = tmp_path / "models" / "test_centrale" / "h72"
    assert (prod_dir / "version.json").exists()
    assert (prod_dir / "data_preparation.csv").exists()
    assert (prod_dir / "bv.json").exists()


def test_run_continues_after_one_dossier_fails(tmp_path, monkeypatch, caplog):
    """Une centrale en échec (données corrompues, bug ponctuel...) ne doit
    jamais empêcher les autres centrales éligibles ce jour-là d'être
    entraînées -- cf. isolation par (dossier, horizon) dans `run()`."""
    _patch_common(tmp_path, monkeypatch)
    centrales_dir = tmp_path / "centrales"

    for dossier in ("centrale_en_panne", "centrale_ok"):
        (centrales_dir / dossier).mkdir(parents=True)
        (centrales_dir / dossier / "bv.json").write_text("{}", encoding="utf-8")

    def fake_train_one(dossier, horizon, **kwargs):
        if dossier == "centrale_en_panne":
            raise ValueError("données corrompues")
        return f"{dossier} h{horizon} : PROMU v1 (test)"

    monkeypatch.setattr(train_script, "train_one", fake_train_one)
    monkeypatch.setattr(train_script, "is_eligible_for_training", lambda d, h: True)

    with caplog.at_level("INFO"):
        exit_code = train_script.run()

    assert exit_code == 1  # au moins un échec -> code de retour non-nul
    assert "centrale_en_panne" in caplog.text and "ÉCHEC" in caplog.text
    assert "centrale_ok" in caplog.text and "PROMU v1" in caplog.text


def test_run_new_dossiers_skips_already_trained_and_trains_new_eligible(tmp_path, monkeypatch):
    _patch_common(tmp_path, monkeypatch)
    centrales_dir = tmp_path / "centrales"

    # "deja_entrainee" a déjà un modèle en prod sur h8 -> relève du mensuel, pas du quotidien.
    (centrales_dir / "deja_entrainee").mkdir(parents=True)
    (centrales_dir / "deja_entrainee" / "bv.json").write_text("{}", encoding="utf-8")
    prod_dir = tmp_path / "models" / "deja_entrainee" / "h8"
    prod_dir.mkdir(parents=True)
    (prod_dir / "version.json").write_text("{}", encoding="utf-8")

    # "nouvelle" n'a aucun modèle -> candidate, éligible si >= 12 mois d'historique.
    (centrales_dir / "nouvelle").mkdir(parents=True)
    (centrales_dir / "nouvelle" / "bv.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(train_script, "history_span_days", lambda d: 400.0 if d == "nouvelle" else 0.0)

    calls = []
    monkeypatch.setattr(
        train_script, "train_one",
        lambda dossier, horizon, **kwargs: calls.append((dossier, horizon)) or f"{dossier} h{horizon} : PROMU v1",
    )

    exit_code = train_script.run_new_dossiers()

    assert exit_code == 0
    assert ("nouvelle", 8) in calls
    assert not any(d == "deja_entrainee" for d, _ in calls)


def test_run_new_dossiers_still_proposes_untrained_horizons_of_partially_trained_dossier(tmp_path, monkeypatch):
    """Un dossier déjà entraîné sur h8 mais pas encore sur h48/h72 doit
    continuer de proposer h48/h72 ici -- run_monthly_retrain ne traite que
    les horizons DÉJÀ en prod, donc sans ce comportement h48/h72 ne seraient
    jamais entraînés une première fois (ni ici, ni là -- trou permanent)."""
    _patch_common(tmp_path, monkeypatch)
    centrales_dir = tmp_path / "centrales"

    (centrales_dir / "partiellement_entrainee").mkdir(parents=True)
    (centrales_dir / "partiellement_entrainee" / "bv.json").write_text("{}", encoding="utf-8")
    prod_dir = tmp_path / "models" / "partiellement_entrainee" / "h8"
    prod_dir.mkdir(parents=True)
    (prod_dir / "version.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(train_script, "history_span_days", lambda d: 400.0)
    calls = []
    monkeypatch.setattr(
        train_script, "train_one",
        lambda dossier, horizon, **kwargs: calls.append((dossier, horizon)) or f"{dossier} h{horizon} : PROMU v1",
    )

    exit_code = train_script.run_new_dossiers()

    assert exit_code == 0
    assert ("partiellement_entrainee", 48) in calls
    assert ("partiellement_entrainee", 72) in calls
    assert ("partiellement_entrainee", 8) not in calls  # h8 déjà en prod -> relève du mensuel


def test_run_monthly_retrain_skips_never_trained_and_retrains_existing(tmp_path, monkeypatch):
    _patch_common(tmp_path, monkeypatch)
    centrales_dir = tmp_path / "centrales"

    (centrales_dir / "deja_entrainee").mkdir(parents=True)
    (centrales_dir / "deja_entrainee" / "bv.json").write_text("{}", encoding="utf-8")
    prod_dir = tmp_path / "models" / "deja_entrainee" / "h8"
    prod_dir.mkdir(parents=True)
    (prod_dir / "version.json").write_text("{}", encoding="utf-8")

    (centrales_dir / "jamais_entrainee").mkdir(parents=True)
    (centrales_dir / "jamais_entrainee" / "bv.json").write_text("{}", encoding="utf-8")

    calls = []
    monkeypatch.setattr(
        train_script, "train_one",
        lambda dossier, horizon, **kwargs: calls.append((dossier, horizon)) or f"{dossier} h{horizon} : conservé",
    )

    exit_code = train_script.run_monthly_retrain()

    assert exit_code == 0
    assert calls == [("deja_entrainee", 8)]  # seul horizon déjà en prod pour ce dossier


def test_run_new_dossiers_continues_after_data_preparation_failure(tmp_path, monkeypatch):
    """Un échec de rafraîchissement data_preparation pour une centrale ne
    doit pas empêcher les autres d'être traitées."""
    _patch_common(tmp_path, monkeypatch)
    centrales_dir = tmp_path / "centrales"

    for dossier in ("casse", "ok"):
        (centrales_dir / dossier).mkdir(parents=True)
        (centrales_dir / dossier / "bv.json").write_text("{}", encoding="utf-8")

    def fake_refresh(dossier):
        if dossier == "casse":
            raise RuntimeError("panne réseau NAS")

    monkeypatch.setattr(train_script, "refresh_data_preparation", fake_refresh)
    monkeypatch.setattr(train_script, "history_span_days", lambda d: 400.0)
    calls = []
    monkeypatch.setattr(
        train_script, "train_one",
        lambda dossier, horizon, **kwargs: calls.append((dossier, horizon)) or f"{dossier} h{horizon} : PROMU v1",
    )

    exit_code = train_script.run_new_dossiers()

    assert exit_code == 1
    assert ("ok", 8) in calls
    assert not any(d == "casse" for d, _ in calls)
