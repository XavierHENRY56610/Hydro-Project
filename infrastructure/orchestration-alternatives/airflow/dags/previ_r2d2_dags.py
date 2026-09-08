"""DAGs Airflow équivalents aux flows Prefect de previ-R2-D2 (bonus phase 08).

Démonstration « et si on avait pris Airflow » (cours obligatoire Sprint 12).
Chaque tâche lance le MÊME script `cron/scripts/` que le flow Prefect
correspondant, via BashOperator — aucune logique métier ici.

Non branché sur l'image principale : à monter dans un Airflow dédié
(`docker compose -f infrastructure/orchestration-alternatives/airflow/docker-compose.airflow.yml up`).
"""

from __future__ import annotations

import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {"retries": 1}
SCRIPTS = "/opt/previ/cron/scripts"  # cron/ monté dans le conteneur Airflow


def _bash(task_id: str, script: str, *, retries: int = 1) -> BashOperator:
    return BashOperator(
        task_id=task_id,
        bash_command=f"cd /opt/previ && python {SCRIPTS}/{script}",
        retries=retries,
    )


with DAG(
    dag_id="previ_hourly_forecast",
    schedule="7 * * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    tags=["previ-r2d2"],
):
    _bash("predict_archive", "predict-archive.py", retries=2)


with DAG(
    dag_id="previ_daily_pipeline",
    schedule="30 3 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    tags=["previ-r2d2"],
):
    maj = _bash("maj_debits", "maj-data.py", retries=3)
    onb = _bash("onboarding_check", "onboarding-check.py")
    bv = _bash("onboarding_bv", "onboarding-bv.py batch")
    dp = _bash("build_data_preparation", "build-data-preparation.py", retries=2)
    val = _bash("validate_data", "validate-data.py", retries=0)
    train = _bash("train_new", "train.py --mode new")
    push = BashOperator(
        task_id="push_git_dvc",
        bash_command='cd /opt/previ && [ -n "$DAGSHUB_TOKEN" ] && git push && dvc push || echo "skip push"',
    )
    maj >> onb >> bv >> dp >> val >> train >> push


with DAG(
    dag_id="previ_weekly_monitoring",
    schedule="0 5 * * 1",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    tags=["previ-r2d2"],
):
    BashOperator(
        task_id="weekly_monitoring",
        bash_command=(
            "cd /opt/previ && python -c "
            "'from previ_r2d2.orchestration.flows import weekly_monitoring_flow; "
            "weekly_monitoring_flow()'"
        ),
    )


with DAG(
    dag_id="previ_monthly_retrain",
    schedule="0 2 1 * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="Europe/Paris"),
    catchup=False,
    tags=["previ-r2d2"],
):
    retrain = _bash("train_monthly", "train.py --mode monthly")
    push = BashOperator(
        task_id="push_git_dvc",
        bash_command='cd /opt/previ && [ -n "$DAGSHUB_TOKEN" ] && git push && dvc push || echo "skip push"',
    )
    retrain >> push
