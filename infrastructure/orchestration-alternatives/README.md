# Orchestrateurs alternatifs (bonus phase 08)

Ce projet utilise **Prefect** (`src/previ_r2d2/orchestration/`). Ce dossier
contient des portages **de démonstration** vers les deux autres
orchestrateurs du cursus, pour la comparaison (`docs/mlops/ORCHESTRATORS.md`).
Aucun n'est branché sur l'image ni la CI.

## Airflow (`airflow/`) — cours obligatoire Sprint 12

```bash
docker compose build test          # (racine) construit previ-r2d2-base
docker compose -f infrastructure/orchestration-alternatives/airflow/docker-compose.airflow.yml up
# UI : http://localhost:8081  (admin / admin)
```

4 DAGs (`dags/previ_r2d2_dags.py`) : `previ_hourly_forecast`,
`previ_daily_pipeline`, `previ_weekly_monitoring`, `previ_monthly_retrain` —
chaque tâche est un `BashOperator` lançant le même `cron/scripts/` que le
flow Prefect équivalent.

## ZenML (`zenml/`) — cours optionnel Sprint 14

```bash
pip install "zenml[server]"
zenml init && zenml stack set default
python infrastructure/orchestration-alternatives/zenml/training_pipeline.py
zenml pipeline runs list
```

`training_pipeline.py` : `load → validate → train → register` en `@step`,
avec `ArtifactConfig` (versionnage natif d'artefacts — l'apport de ZenML,
redondant ici avec MLflow Registry + DVC).
