# Architecture — previ-R2-D2

Plateforme MLOps de bout en bout autour du modèle hybride de prévision de
débit (LightGBM + BiLSTM + stacking). Tout tourne en `docker compose` ;
l'API est aussi déployable en Kubernetes (Helm).

## Vue d'ensemble

```mermaid
flowchart TB
    subgraph SRC["Sources"]
      HUB["Hub'Eau / eaufrance<br/>débits"]
      NWP["Météo NWP (figée)"]
    end

    subgraph CORE["Cœur ML — previ_r2d2"]
      PREP["preprocessing<br/>débit · BV · data_preparation"]
      VAL["validation<br/>contrat data_preparation.csv"]
      MODEL["model<br/>LightGBM + BiLSTM + stacking"]
      PROMO["promotion.py<br/>KGE candidat vs prod"]
    end

    subgraph TRACK["Suivi & versioning"]
      MLF["MLflow<br/>runs + Model Registry @production"]
      PG[("PostgreSQL")]
      S3[("MinIO<br/>artefacts")]
      DVC[("DVC / DagsHub<br/>données + modèles + tags git")]
    end

    subgraph SERVE["Service"]
      API["API FastAPI<br/>/predict /forecast /models /health /metrics"]
      NGINX["Nginx<br/>reverse-proxy / Ingress"]
    end

    subgraph ORCH["Orchestration — Prefect"]
      HOURLY["hourly-forecast"]
      DAILY["daily-pipeline"]
      WEEKLY["weekly-monitoring"]
      MONTHLY["monthly-retrain"]
    end

    subgraph MON["Monitoring"]
      EXP["exporter métier :9200"]
      DRIFT["Evidently<br/>dérive hebdo"]
      PROM["Prometheus<br/>+ règles d'alerte"]
      GRAF["Grafana<br/>3 dashboards"]
    end

    HUB --> PREP
    NWP --> PREP
    PREP --> VAL --> MODEL --> PROMO
    MODEL -.->|params · métriques · artefacts| MLF
    PROMO -->|@production| MLF
    PROMO --> DVC
    MLF --- PG
    MLF --- S3
    API -->|models:/...@production, repli models/| MLF
    NGINX --> API

    HOURLY --> API
    DAILY --> PREP
    DAILY --> MODEL
    MONTHLY --> MODEL
    WEEKLY --> DRIFT
    WEEKLY --> EXP

    API --> PROM
    EXP --> PROM
    DRIFT --> EXP
    PROM --> GRAF
    PROM -->|alertes| GRAF
```

## Composants

| Domaine | Techno | Où |
|---|---|---|
| Cœur ML | LightGBM (Optuna+OOF) · BiLSTM (torch CPU, attention) · stacking Ridge | `src/previ_r2d2/model/` |
| Preprocessing | Hub'Eau, onboarding BV, `data_preparation.csv`, features ET0/neige/amont | `src/previ_r2d2/preprocessing/` |
| Validation données | contrat hand-rolled (erreurs/warnings), stage DVC `validate` | `src/previ_r2d2/preprocessing/data_preparation/validation.py` |
| Suivi d'expériences | MLflow (PostgreSQL + MinIO), Model Registry alias `@production` | `src/previ_r2d2/model/tracking/mlflow_tracking.py` |
| Versionnage données/modèles | DVC + remote DagsHub, tags git `<dossier>-h<h>-v<N>` | `dvc/`, `promotion.py` |
| API de service | FastAPI — logs JSON, `X-Request-ID`, JWT optionnel, `/metrics` Prometheus | `src/previ_r2d2/serving/` |
| Orchestration | Prefect 3 — 4 flows planifiés (`serve()`) | `src/previ_r2d2/orchestration/` |
| Monitoring | Evidently (dérive) · exporter Prometheus · Grafana · 6 alertes | `src/previ_r2d2/monitoring/`, `infrastructure/monitoring/` |
| Reverse-proxy | Nginx (compose) / Ingress nginx (k8s) | `infrastructure/nginx/`, chart Helm |
| Déploiement | chart Helm (Deployment + Service + Ingress + HPA + PVC) | `infrastructure/helm/previ-r2d2/` |
| CI/CD | GitHub Actions — lint + tests + couverture ; release image API sur tag | `.github/workflows/` |

## Cadences opérationnelles

| Fréquence | Flow Prefect | Effet |
|---|---|---|
| horaire (`hh:07`) | `hourly-forecast` | prévision + archivage pour chaque (dossier, horizon) en prod |
| quotidien (`03:30`) | `daily-pipeline` | rafraîchit débits → BV → data_preparation → validation → `train --mode new` → push git + DVC |
| hebdo (`lun 05:00`) | `weekly-monitoring` | dérive Evidently + KGE en ligne → `state.json` → gauges Prometheus |
| mensuel (`1er 02:00`) | `monthly-retrain` | réentraînement complet, promotion conditionnelle par KGE, push |

## Chargement du modèle par l'API

1. Model Registry MLflow — `models:/previ-r2d2-<dossier>-h<horizon>@production`
   (artefacts téléchargés + cachés dans `outputs/serving_cache/`)
2. Repli `models/<dossier>/h<horizon>/` — modèle promu, versionné DVC + tag git

Le repli n'est pas dégradé : git + DVC restent la source de vérité pour la
reproductibilité exacte. `API_MODEL_SOURCE` (`auto` | `registry` | `local`).

## Ports (compose)

| Service | Port | |
|---|---|---|
| nginx | 8080 | point d'entrée unique |
| api | 8000 | `/docs` Swagger |
| mlflow | 5000 | UI |
| minio | 9001 | console |
| prefect-server | 4200 | UI |
| prometheus | 9090 | |
| grafana | 3000 | admin/admin |
| monitoring-exporter | 9200 | `/metrics` métier |
| node-exporter | 9100 | |
