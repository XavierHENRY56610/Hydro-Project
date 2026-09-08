"""Tests des flows Prefect (phase 05).

Les corps de tâche (`_run_script`, appels `subprocess`) sont mockés : on teste
l'orchestration — enchaînement des étapes, arrêt sur échec bloquant, garde du
push — pas les scripts `cron/` eux-mêmes (couverts par `tests/cron/`).
"""

from __future__ import annotations

import pytest
from prefect.testing.utilities import prefect_test_harness

from previ_r2d2.common import config
from previ_r2d2.orchestration import flows, notifications


@pytest.fixture(autouse=True, scope="module")
def _harness():
    with prefect_test_harness():
        yield


@pytest.fixture
def recorded(monkeypatch):
    calls: list[tuple[str, tuple[str, ...]]] = []
    monkeypatch.setattr(flows, "_run_script", lambda script, *args: calls.append((script, args)))
    monkeypatch.setattr(config.settings, "dagshub_token", "")
    return calls


def test_hourly_flow_runs_predict_archive(recorded):
    flows.hourly_forecast_flow()
    assert recorded == [("predict-archive.py", ())]


def test_daily_pipeline_runs_stages_in_order(recorded):
    flows.daily_pipeline_flow(push=False)
    assert [c[0] for c in recorded] == [
        "maj-data.py",
        "onboarding-check.py",
        "onboarding-bv.py",
        "build-data-preparation.py",
        "validate-data.py",
        "train.py",
    ]
    assert recorded[-1][1] == ("--mode", "new")


def test_daily_pipeline_push_skipped_without_token(recorded, monkeypatch):
    pushed: list = []
    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: pushed.append(a))
    flows.daily_pipeline_flow(push=True)
    assert pushed == []  # DAGSHUB_TOKEN vide -> pas de git/dvc push


def test_daily_pipeline_push_runs_with_token(recorded, monkeypatch):
    monkeypatch.setattr(config.settings, "dagshub_token", "tok")
    pushed: list = []
    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: pushed.append(a[0]))
    flows.daily_pipeline_flow(push=True)
    assert ["git", "push"] in pushed
    assert ["dvc", "push"] in pushed


def test_validation_failure_stops_before_training(monkeypatch):
    calls: list[str] = []

    def fake(script, *args):
        calls.append(script)
        if script == "validate-data.py":
            raise RuntimeError("données non conformes")

    monkeypatch.setattr(flows, "_run_script", fake)
    monkeypatch.setattr(config.settings, "dagshub_token", "")

    with pytest.raises(Exception):
        flows.daily_pipeline_flow(push=False)

    assert "validate-data.py" in calls
    assert "train.py" not in calls  # l'entraînement ne part pas


def test_monthly_retrain_trains_then_pushes(recorded, monkeypatch):
    monkeypatch.setattr(config.settings, "dagshub_token", "tok")
    pushed: list = []
    monkeypatch.setattr(flows.subprocess, "run", lambda *a, **k: pushed.append(a[0]))
    flows.monthly_retrain_flow(push=True)
    assert recorded == [("train.py", ("--mode", "monthly"))]
    assert ["git", "push"] in pushed


def test_notify_failure_noop_without_webhook(monkeypatch):
    monkeypatch.delenv("PREFECT_FAILURE_WEBHOOK", raising=False)
    # ne doit pas lever, même avec des objets nuls
    notifications.notify_failure(flow=None, flow_run=None, state=None)


def test_notify_failure_posts_to_webhook(monkeypatch):
    sent: list = []
    monkeypatch.setenv("PREFECT_FAILURE_WEBHOOK", "https://hooks.example/x")
    monkeypatch.setattr(
        notifications.urllib.request, "urlopen",
        lambda req, timeout=None: sent.append(req.data),
    )

    class _FR:
        name = "daily-pipeline/abc"

    notifications.notify_failure(flow=None, flow_run=_FR(), state=None)
    assert sent and b"chec" in sent[0]  # "échec" encodé
