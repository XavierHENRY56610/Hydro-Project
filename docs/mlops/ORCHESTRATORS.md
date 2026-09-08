# Orchestrateurs — Prefect (retenu) vs Airflow vs ZenML

Les 3 sont au programme (Sprints 12 et 14). Ce projet **retient Prefect** ;
Airflow et ZenML sont fournis en démonstration dans
`infrastructure/orchestration-alternatives/`.

## Ce que chacun résout

| | Prefect | Airflow | ZenML |
|---|---|---|---|
| Nature | orchestrateur de workflows Python | orchestrateur de DAGs (standard entreprise) | framework de pipelines ML (steps + artefacts versionnés) |
| Unité | `@flow` / `@task` | `DAG` / `Operator` | `@pipeline` / `@step` |
| Planification | cron par déploiement (`serve()` ou work pools) | cron par DAG (scheduler) | via un orchestrateur branché (local, Airflow, Kubeflow…) |
| Versionnage d'artefacts | non (délégué à MLflow/DVC) | non | **oui, natif** (`ArtifactConfig`, lineage) |
| Retries / observabilité | natifs, UI | natifs, UI riche | dépend de l'orchestrateur |
| Poids | léger (pip) | lourd (webserver + scheduler + DB) | moyen |

## Pourquoi Prefect ici

1. Les tâches sont **déjà** des fonctions Python propres (`cron/scripts/`) —
   `@task` autour, zéro réécriture. Airflow imposerait des `Operator` et une
   structure DAG ; ZenML imposerait de découper en `@step` typés.
2. Cadences simples (horaire / quotidien / hebdo / mensuel) — `serve()` +
   cron suffit, pas besoin du scheduler + workers Airflow.
3. Versionnage d'artefacts : **déjà couvert** par MLflow Registry + DVC/tags
   git (cf. `MLOPS.md`, « doubles filets »). L'apport natif de ZenML fait
   doublon.
4. Empreinte : Prefect ajoute 2 conteneurs légers ; Airflow en ajoute 3-4
   dont un PostgreSQL dédié.

## Airflow serait meilleur si…

- l'équipe grandit et le standard « DAG » facilite l'onboarding ;
- les dépendances entre tâches deviennent complexes (branches, backfill,
  SLA) ;
- besoin d'un large catalogue de providers (BigQuery, Spark, dbt…).

DAGs équivalents : `infrastructure/orchestration-alternatives/airflow/dags/`
(BashOperator lançant les mêmes `cron/scripts/`). Compose dédié fourni.

## ZenML serait meilleur si…

- on voulait le **lineage** artefact-par-artefact et la comparaison de runs
  au niveau step (sans MLflow) ;
- on visait un déploiement multi-stack (local → cloud) piloté par la config.

Pipeline d'entraînement porté :
`infrastructure/orchestration-alternatives/zenml/training_pipeline.py`
(`@pipeline` enchaînant load → validate → train → evaluate → register, avec
`ArtifactConfig` versionné).

## Verdict

Pour **ce** projet (2 personnes, 3 centrales, cadences fixes, versionnage
déjà assuré) : **Prefect**. Airflow = sur-dimensionné, ZenML = redondant avec
MLflow/DVC. Les deux restent utiles à connaître et sont démontrés.
