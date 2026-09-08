# MLOps — chaque brique et le cours dont elle vient

Projet fil rouge de la formation DataScientest (Data Scientist + AI
Engineering / MLOps). Chaque phase du plan s'adosse à un cours.

## Correspondance brique → cours

| Phase | Brique | Cours DataScientest | Statut cours | Choix |
|---|---|---|---|---|
| 00 | Socle Docker + `docker compose` + `pydantic-settings` | **Docker** (S10) | obligatoire | — |
| 00 | Config typée / secrets | Sécurisation des API (S10) | optionnel | — |
| 01 | MLflow tracking + Model Registry | **MLflow** (S11) | obligatoire | ✅ (vs Weights & Biases, optionnel) |
| 01 | Artefacts sur MinIO | MinIO (S11) | optionnel | ✅ |
| 01 | Backend PostgreSQL | SQL — base relationnelle (S12) | optionnel | ✅ |
| 00/01 | DVC + remote DagsHub | **DVC et DagsHub** (S11) | obligatoire | ✅ (déjà dans le projet) |
| 02 | Validation `data_preparation.csv` | Drift Monitoring / **Data quality** (S13) | obligatoire | contrat hand-rolled (colonnes variables par centrale) |
| 03 | API FastAPI + cadrage prod | LLM en Production (S10) · **BentoML** (S12) | optionnel / **obligatoire** | FastAPI ; **BentoML = à ajouter (phase 03b)** |
| 04 | CI/CD GitHub Actions + release GHCR | CI/CD (dans **Docker**, S10) | obligatoire | ✅ (vs Jenkins, optionnel) |
| 05 | Orchestration Prefect (flows planifiés) | **Airflow** (S12) · Prefect (S12) | **obligatoire** / optionnel béta | **Prefect** retenu ; **Airflow = bonus phase 08** |
| 06 | Dérive Evidently | **Drift Monitoring** (S13) | obligatoire | ✅ |
| 06 | Prometheus + Grafana + alertes | **Prometheus et Grafana MLOps** (S13) | obligatoire | ✅ |
| 07 | Kubernetes + Helm (chart API, HPA, Ingress) | **Kubernetes** (S14) | obligatoire | ✅ |
| 07 | Nginx reverse-proxy / Ingress | **Nginx** (S11) | obligatoire | ✅ (ajouté phase 07) |
| 08 | ZenML (pipeline + `ArtifactConfig`) | ZenML (S14) | optionnel | **bonus** (`infrastructure/orchestration-alternatives/zenml/`) |
| 08 | Comparaison orchestrateurs | transverse | — | `docs/mlops/ORCHESTRATORS.md` |

### Cours volontairement écartés

| Cours | Raison |
|---|---|
| Weights & Biases (S11, opt.) | MLflow retenu — Registry + intégration DVC/git déjà en place |
| Jenkins (S11, opt.) | GitHub Actions — le dépôt est sur GitHub |
| Nginx **comme serveur web statique** | pas de front lourd ; utilisé en reverse-proxy uniquement |
| Elasticsearch (S12, opt.) | pas de besoin de recherche / logs centralisés à ce stade |
| MongoDB (S12, opt.) | tout est fichiers + PostgreSQL (MLflow) |
| Kafka (S13, opt.) | volumétrie horaire faible, pas de streaming |
| Patterns Agentiques (S13, opt.) | hors périmètre prévision de débit |

## Doubles filets assumés

| Rôle | Outil A | Outil B | Pourquoi les deux |
|---|---|---|---|
| Repro d'un état | DVC (`dvc repro`) + tag git | — | référence unique |
| Lecture rapide d'un modèle | MLflow Registry `@production` | repli `models/` DVC | l'API lit vite le registry ; DVC/git garde la repro exacte |
| Planification | Prefect (`serve()`) | pipelines DVC | Prefect = opérationnel ; DVC = repro |
| Métriques modèle | exporter Prometheus | tags/artefacts MLflow | temps réel vs historique d'expériences |

## Rétro « v2 » — ce qu'on simplifierait

Pour un run réellement en production (2 personnes, 3 centrales, cadence
horaire), la stack de formation est surdimensionnée. Ordre de coupe :

1. **Retirer node-exporter + dashboard Infra** — 3 conteneurs qui tournent
   sur un poste, l'infra se surveille autrement (systemd, uptime kuma).
2. **Fusionner l'exporter métier dans l'API** (`/metrics` unique) via
   `prometheus_client` multiprocess — un service en moins.
3. **Choisir UN orchestrateur** : Prefect *ou* les pipelines DVC + un cron,
   pas les deux. Prefect si l'UI/retries/alerting servent vraiment.
4. **MinIO → filesystem local** pour les artefacts MLflow tant qu'on est
   mono-poste ; MinIO seulement si passage cloud.
5. **Kubernetes seulement si le trafic API le justifie** — `docker compose`
   + `restart: unless-stopped` suffit à 2 utilisateurs.
6. **Garder** : DVC/DagsHub (partage données), MLflow (Registry + historique),
   validation des données, Evidently hebdo, 2 dashboards Grafana (API +
   Modèle), CI.

Cible minimale défendable : **DVC + MLflow + FastAPI + Prefect + Grafana +
Evidently + CI**. Le reste est pédagogique.
