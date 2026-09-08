# Installation & démarrage

Projet à deux — **tout tourne dans Docker**. Aucun Python / conda à installer
en local.

## Prérequis

- **Docker** + **Docker Compose v2** (Docker Desktop, ou Docker Engine + `docker compose`).
- **git**.
- Un compte **DagsHub** avec un token perso (Settings → Tokens sur
  [dagshub.com](https://dagshub.com)).

## Démarrage

```bash
git clone https://github.com/XavierHENRY56610/Hydro-Project.git
cd Hydro-Project
git checkout import-projet        # (ou la branche de travail courante)

cp .env.example .env
# éditer .env : DAGSHUB_USER + DAGSHUB_TOKEN

make build                       # construit l'image de base (~5-10 min la 1re fois)
make up                          # démarre la stack
docker compose run --rm trainer dvc pull   # données + modèles (config-general.json,
                                           # shapefiles/, centrales/<dossier>/..., modèles)
make test                        # 319 tests -> tous verts
```

## Cibles Make

| Cible | Effet |
|---|---|
| `make build` | (re)construit l'image de base |
| `make up` / `make down` | démarre / arrête la stack |
| `make shell` | bash interactif dans le conteneur `trainer` |
| `make test` | suite rapide (`pytest -m "not slow"`) |
| `make test-slow` | tests d'intégration lents — **crée de vrais commits + tags** |
| `make pipeline` | `dvc repro dvc/preprocessing/dvc.yaml` |
| `make train DOSSIER=touzac_g2_G2 H=8` | entraînement ciblé |
| `make predict` | prédiction + archivage |
| `make lint` | `ruff` |
| `make lock` | regénère `infrastructure/docker/requirements.lock` |

## Où sont les choses

| | |
|---|---|
| Image commune | `infrastructure/docker/Dockerfile.base` |
| Services | `docker-compose.yml` (un service par phase MLOps) |
| Config typée | `src/previ_r2d2/common/config.py` — `Settings` (pydantic-settings) |
| Secrets | `.env` (gitignoré) — `secret_config.py` toujours prioritaire s'il existe |
| Données / modèles | DVC + remote DagsHub (`dvc pull` / `dvc push`) |

## Travail en équipe

- **Code + environnement** : partagés par git + Docker (image reproductible).
- **Données + modèles** : partagés par DVC / DagsHub.
- Après un entraînement qui promeut un modèle : `git push` **et** `dvc push`
  (le commit + tag de `promotion.py` sont locaux — cf. skill `hydro-projet`).
- Le token DagsHub de chacun vit dans son `.env` (jamais commité) ;
  l'`entrypoint.sh` le pousse dans `.dvc/config.local` au démarrage du conteneur.
