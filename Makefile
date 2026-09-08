# previ-R2-D2 — tout passe par Docker Compose (projet à deux).
.DEFAULT_GOAL := help
DC := docker compose

.PHONY: help build up down shell test test-slow pipeline train predict lint lock clean mlflow logs api prefect flow-run monitor fire-alert

help: ## liste les cibles
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build: ## (re)construit les images
	$(DC) build

up: ## démarre la stack MLflow (UI http://localhost:5000, MinIO http://localhost:9001)
	$(DC) up -d mlflow

mlflow: up ## alias de `up`

api: ## démarre l'API de service (http://localhost:8000/docs) + MLflow
	$(DC) up -d api

prefect: ## démarre l'orchestration Prefect (UI http://localhost:4200) + worker planifié
	$(DC) up -d prefect-worker

flow-run: ## déclenche un flow planifié maintenant — make flow-run FLOW=daily-pipeline
	$(DC) run --rm prefect-worker prefect deployment run "$(FLOW)/$(FLOW)"

monitor: ## démarre Prometheus (:9090) + Grafana (:3000) + exporter métier + node-exporter
	$(DC) up -d prometheus grafana node-exporter monitoring-exporter

fire-alert: ## force un KGE effondré + une dérive dans state.json (démo alerting)
	$(DC) run --rm monitoring-exporter python -m previ_r2d2.monitoring.exporter --fire-alert

down: ## arrête la stack
	$(DC) down

logs: ## suit les logs de la stack
	$(DC) logs -f

shell: ## bash interactif dans le conteneur trainer
	$(DC) run --rm trainer bash

test: ## suite rapide (pytest -m "not slow")
	$(DC) run --rm test

test-slow: ## tests d'intégration lents — ATTENTION : crée de vrais commits + tags
	$(DC) run --rm test python -m pytest -m slow -v tests/integration/

pipeline: ## dvc repro du pilier preprocessing (debit -> onboarding_check -> bv -> data_preparation)
	$(DC) run --rm trainer dvc repro dvc/preprocessing/dvc.yaml

train: ## entraînement ciblé — make train DOSSIER=touzac_g2_G2 H=8
	$(DC) run --rm trainer python cron/scripts/train.py --dossier $(DOSSIER) --horizon $(H) --force

predict: ## prédiction + archivage (toutes les centrales avec un modèle en prod)
	$(DC) run --rm trainer python cron/scripts/predict-archive.py

lint: ## ruff
	$(DC) run --rm trainer ruff check src tests cron

lock: ## regénère infrastructure/docker/requirements.lock (hashé) depuis requirements.in
	$(DC) run --rm trainer sh -c "pip install -q pip-tools && \
	  pip-compile --generate-hashes --allow-unsafe \
	  --output-file=infrastructure/docker/requirements.lock \
	  infrastructure/docker/requirements.in"

clean: ## supprime conteneurs + volumes
	$(DC) down -v
