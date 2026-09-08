# Phase 01 — Suivi d'expériences MLflow + Model Registry

## Ce que ça apporte

- **Tracking** : chaque entraînement (`run_training`) crée un run MLflow avec
  params, métriques (KGE global + par pas / régime / saison, RMSE, composantes
  stacking) et artefacts (7 plots, `results.json`, `meta_config.json`, CSV test).
- **Model Registry** : un modèle enregistré par `(dossier, horizon)`, nommé
  `previ-r2d2-<dossier>-h<horizon>`. Chaque entraînement automatisé
  (`cron/scripts/train.py`) crée une **nouvelle version**. La promotion pose
  l'alias `@production` sur la version retenue et l'enlève de l'ancienne.
- Le versionnement **DVC + tag git** de `models/<dossier>/h<horizon>/` est
  **conservé** : double filet. Registry = lecture rapide pour l'API (phase 03) ;
  DVC/git = reproductibilité exacte.

## Stack ajoutée (`docker-compose.yml`)

| Service | Rôle | Port |
|---|---|---|
| `mlflow` | serveur MLflow (`Dockerfile.mlflow` : image officielle + psycopg2 + boto3) | 5000 |
| `mlflow-db` | PostgreSQL — backend store du registry | — |
| `minio` | stockage S3 des artefacts | 9001 (console) |
| `minio-setup` | crée le bucket `mlflow-artifacts` au démarrage | — |
| `trainer` | branché : `MLFLOW_TRACKING_URI=http://mlflow:5000` + creds S3 | — |

`test` n'a **pas** `MLFLOW_TRACKING_URI` → le suivi est un no-op en CI / tests.

## Activation

Le suivi s'active dès que `MLFLOW_TRACKING_URI` est défini (l'environnement le
fait pour `trainer`). Sans lui : toutes les fonctions de
`previ_r2d2.model.tracking.mlflow_tracking` sont des no-op, `mlflow` n'est même
pas importé.

## Utilisation

```bash
make up                 # démarre mlflow + mlflow-db + minio (UI : http://localhost:5000)
make train DOSSIER=touzac_g2_G2 H=8      # entraîne -> run + version registry
```

- UI MLflow : http://localhost:5000 — expérience `previ-r2d2/<dossier>`, registry
- Console MinIO : http://localhost:9001 (`minioadmin` / `minioadmin`)

`run.py` (expés manuelles) loggue aussi ses runs mais **n'enregistre pas** au
registry (`register=False`).

## Fichiers touchés

- `src/previ_r2d2/model/tracking/mlflow_tracking.py` — **nouveau**, tout le suivi
- `src/previ_r2d2/model/pipeline/orchestrator.py` — `run_training` enveloppé dans
  un run MLflow ; `_run_training_body` = ancien corps inchangé
- `cron/scripts/train.py` — `train_one` : après `promote_model`, pose l'alias
  `@production` sur la version registry
- `src/previ_r2d2/cli.py` — `run.py` passe `register=False`
- `infrastructure/docker/{Dockerfile.mlflow, requirements.in}`, `docker-compose.yml`,
  `Makefile`, `.env.example`

## Limites connues (héritées) à traiter plus tard

- Le modèle est enregistré comme un **dossier d'artefacts** (`model/`), pas un
  flavor MLflow standard — l'API le téléchargera et le chargera avec
  `load_trained_models` (le modèle hybride LGBM+BiLSTM+Ridge n'a pas de flavor).
- Pas encore de nettoyage automatique des vieilles versions du registry.
