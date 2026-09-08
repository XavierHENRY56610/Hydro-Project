"""Flows Prefect previ-R2-D2 (phase 05).

Trois cadences, alignées sur le README du projet :
  - **horaire**  `hourly_forecast_flow`  -> predict-archive
  - **quotidien** `daily_pipeline_flow`   -> maj débits -> onboarding -> BV ->
    data_preparation -> validation -> entraînement des nouvelles centrales ->
    push git + DVC
  - **mensuel**  `monthly_retrain_flow`   -> réentraînement + promotion
    conditionnelle -> push git + DVC

Chaque tâche exécute le script `cron/scripts/` correspondant en
sous-processus — exactement l'invocation des stages DVC, sans réécriture. Un
code retour non nul fait échouer la tâche (et déclenche ses retries).
"""

from __future__ import annotations

import logging
import subprocess
import sys

from prefect import flow, task

from previ_r2d2.common import config
from previ_r2d2.orchestration.notifications import notify_failure

logger = logging.getLogger(__name__)

_SCRIPTS = config.ROOT / "cron" / "scripts"


def _run_script(script: str, *args: str) -> None:
    """Exécute `cron/scripts/<script>` en sous-processus (cwd = racine projet)."""
    cmd = [sys.executable, str(_SCRIPTS / script), *args]
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=config.ROOT, check=True)


# --- Tâches (corps = scripts existants) ---------------------------------
@task(retries=3, retry_delay_seconds=60, task_run_name="maj-debits")
def refresh_debits() -> None:
    _run_script("maj-data.py")  # Hub'Eau tombe souvent -> 3 retries


@task(retries=1, task_run_name="onboarding-check")
def onboarding_check() -> None:
    _run_script("onboarding-check.py")


@task(retries=1, task_run_name="onboarding-bv")
def onboarding_bv() -> None:
    _run_script("onboarding-bv.py", "batch")


@task(retries=2, retry_delay_seconds=60, task_run_name="build-data-preparation")
def build_data_preparation() -> None:
    _run_script("build-data-preparation.py")


@task(task_run_name="validate-data")
def validate_data() -> None:
    _run_script("validate-data.py")  # erreur bloquante -> code 1 -> tâche KO


@task(retries=1, task_run_name="train-{mode}")
def train(mode: str) -> None:
    _run_script("train.py", "--mode", mode)


@task(retries=2, retry_delay_seconds=60, task_run_name="predict-archive")
def predict_archive() -> None:
    _run_script("predict-archive.py")


@task(retries=2, retry_delay_seconds=30, task_run_name="push-git-dvc")
def push_artifacts() -> None:
    """`git push` + `dvc push` (les commits/tags de promotion sont locaux).
    No-op si `DAGSHUB_TOKEN` absent (dev / CI)."""
    if not config.settings.dagshub_token:
        logger.warning("DAGSHUB_TOKEN absent — git push / dvc push ignorés.")
        return
    subprocess.run(["git", "push"], cwd=config.ROOT, check=True)
    subprocess.run(["dvc", "push"], cwd=config.ROOT, check=True)


# --- Flows -------------------------------------------------------------
@flow(name="hourly-forecast", on_failure=[notify_failure])
def hourly_forecast_flow() -> None:
    """Horaire : prévision + archivage pour chaque (dossier, horizon) en prod."""
    predict_archive()


@flow(name="daily-pipeline", on_failure=[notify_failure])
def daily_pipeline_flow(push: bool = True) -> None:
    """Quotidien : rafraîchit les données de bout en bout puis entraîne les
    centrales atteignant 12 mois d'historique (train --mode new)."""
    refresh_debits()
    onboarding_check()
    onboarding_bv()
    build_data_preparation()
    validate_data()
    train("new")
    if push:
        push_artifacts()


@flow(name="monthly-retrain", on_failure=[notify_failure])
def monthly_retrain_flow(push: bool = True) -> None:
    """Mensuel : réentraîne tous les modèles en production, promotion
    conditionnelle par KGE (cf. promotion.py)."""
    train("monthly")
    if push:
        push_artifacts()
