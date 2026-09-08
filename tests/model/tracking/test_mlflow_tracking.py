"""Le suivi MLflow doit être un no-op complet quand MLFLOW_TRACKING_URI n'est
pas défini (tests, CI) — aucun import mlflow, aucune erreur."""

from __future__ import annotations

from pathlib import Path

from previ_r2d2.model.tracking import mlflow_tracking


def _disable(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(mlflow_tracking.config.settings, "mlflow_tracking_uri", "")


def test_disabled_when_no_uri(monkeypatch):
    _disable(monkeypatch)
    assert mlflow_tracking.enabled() is False


def test_training_run_yields_none_when_disabled(monkeypatch):
    _disable(monkeypatch)
    with mlflow_tracking.training_run("apas_G1_G4", 8, "ridge", {"epochs": 40}) as run_id:
        assert run_id is None


def test_log_and_register_are_noop_when_disabled(monkeypatch, tmp_path):
    _disable(monkeypatch)
    mlflow_tracking.log_results({"kge_stacking": 0.8})
    mlflow_tracking.log_artifacts(tmp_path, tmp_path)
    assert mlflow_tracking.register_candidate("apas_G1_G4", 8, tmp_path, None) is None
    mlflow_tracking.promote_to_production("apas_G1_G4", 8, None)  # ne lève pas


def test_enabled_when_uri_set(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    assert mlflow_tracking.enabled() is True


def test_registered_model_name():
    assert mlflow_tracking.registered_model_name("touzac_g2_G2", 8) == "previ-r2d2-touzac_g2_G2-h8"


def test_git_sha_returns_string():
    sha = mlflow_tracking._git_sha()
    assert isinstance(sha, str) and sha


def test_module_does_not_import_mlflow_at_load():
    """mlflow n'est importé que dans les fonctions (lazy) — le module se charge
    même sans mlflow installé (CI légère)."""
    src = Path(mlflow_tracking.__file__).read_text(encoding="utf-8")
    top_level = [
        line for line in src.splitlines()
        if line.startswith("import mlflow") or line.startswith("from mlflow")
    ]
    assert top_level == []
