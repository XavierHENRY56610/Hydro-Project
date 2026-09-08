# Phase 03 — API de service (FastAPI) + conteneurisation

## Ce que ça apporte

Une **porte d'entrée réseau** au modèle hybride. Jusqu'ici la prévision
n'existait qu'en CLI (`previ-r2d2 --predict`) et via le stage DVC horaire
`predict-archive`. L'API expose la même logique (`run_prediction`, déjà
testée) en HTTP REST + JSON, cadrée comme en production.

## Endpoints

| Méthode | Route | Rôle | Auth |
|---|---|---|---|
| `POST` | `/predict` | Calcule une prévision `(dossier, horizon)` — `source` `live` ou `frozen` | oui\* |
| `GET` | `/forecast/{dossier}` | Dernières prévisions archivées (`prevision.json` / `enchere.json`) | oui\* |
| `GET` | `/models` | Inventaire des modèles promus (version, KGE, origine registry/local) | oui\* |
| `GET` | `/health` | Sonde de vivacité + état MLflow + nb de modèles | **publique** |
| `GET` | `/metrics` | Exposition Prometheus | **publique** |
| `GET` | `/docs` | Swagger UI | publique |

\* seulement si `API_AUTH_REQUIRED=true` **et** `API_JWT_SECRET` renseigné.
Par défaut l'API est **ouverte** (démo / dev).

## Chargement du modèle (`serving/model_registry.py`)

Ordre de résolution, par `(dossier, horizon)` :

1. **Model Registry MLflow** — alias `models:/previ-r2d2-<dossier>-h<horizon>@production`.
   Les artefacts sont téléchargés une fois dans `outputs/serving_cache/` puis
   réutilisés.
2. **Repli `models/<dossier>/h<horizon>/`** — modèle promu, versionné DVC + tag git.

Le repli n'est pas un mode dégradé : git + DVC restent la source de vérité
pour la repro exacte (cf. phase 01). `API_MODEL_SOURCE` (`auto` | `registry` |
`local`) force le comportement.

`run_prediction` a reçu un paramètre `weights_dir` optionnel — l'API y passe
le dossier résolu, le reste du pipeline est inchangé.

## Cadrage production (cours FastAPI / LLM en Production)

- **Logs JSON structurés** (`serving/logging_config.py`) — une ligne JSON par
  log, sur stdout, avec `request_id`.
- **`X-Request-ID`** — repris de l'en-tête entrant ou généré (UUID), propagé
  via `ContextVar`, renvoyé dans la réponse, présent dans chaque log.
- **Erreurs sans stack trace côté client** — `{"detail": ..., "request_id": ...}`.
  Les 500 loggent la trace côté serveur uniquement.
- **Auth Bearer JWT optionnelle** (`serving/security.py`, HS256) — désactivée
  par défaut ; `/health` et `/metrics` restent publics même activée.
- **Métriques** (`serving/metrics.py`) — `previ_api_requests_total`,
  `previ_api_request_seconds`, `previ_api_predictions_total`,
  `previ_model_kge`. Le monitoring complet (dérive, KGE en ligne, Grafana)
  arrive en phase 06.

## Docker

- `infrastructure/docker/Dockerfile.api` — hérite de `previ-r2d2-base`
  (fastapi / uvicorn sont dans `requirements.in`), `CMD uvicorn ... :app`.
- Service `api` dans `docker-compose.yml` : port `8000`, `depends_on: mlflow`,
  healthcheck `curl /health`, mêmes bind-mounts que `trainer`.
- `make api` démarre l'API + MLflow. Swagger : http://localhost:8000/docs

## Utilisation

```bash
make api
curl -s localhost:8000/health | jq
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"dossier": "touzac_g2_G2", "horizon": 8, "source": "frozen"}' | jq
curl -s localhost:8000/models | jq
curl -s localhost:8000/metrics
```

## Tests

`tests/serving/test_api.py` — 13 tests boîte noire (`TestClient`) : contrat
HTTP, validation, 404 / 422 / 401, `X-Request-ID`, auth JWT, format
Prometheus. `run_prediction` et `resolve` sont mockés ; le modèle lui-même
reste couvert par `tests/model/pipeline/test_predict_orchestrator.py`.

## Ce qui reste pour plus tard

- Front Streamlit de démo (bonus, si le temps le permet — phase 08).
- Variante BentoML pour comparer les approches de packaging (bonus).
- Rate-limiting / quotas : hors périmètre projet perso.
