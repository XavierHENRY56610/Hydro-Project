# Installation & démarrage

Projet à deux — **tout tourne dans Docker**. Aucun Python / conda à installer
en local.

## Prérequis

- **Docker** + **Docker Compose v2** (Docker Desktop, ou Docker Engine + `docker compose`).
- **git**.
- Un compte **DagsHub** avec un token perso (Settings → Tokens sur
  [dagshub.com](https://dagshub.com)).
- **GNU Make** (optionnel) — absent de Git for Windows par défaut. Sans lui,
  utiliser directement les commandes `docker compose ...` de la table ci-dessous
  (ex. `docker compose run --rm test` au lieu de `make test`).

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
make test                        # 355 tests -> tous verts
make api                         # API de service -> http://127.0.0.1:8000/docs
```

## Cibles Make (et équivalent `docker compose`)

| Cible | `docker compose` équivalent | Effet |
|---|---|---|
| `make build` | `docker compose build` | (re)construit l'image de base |
| `make up` / `make down` | `docker compose up -d` / `down` | démarre / arrête la stack |
| `make shell` | `docker compose run --rm trainer bash` | bash interactif |
| `make test` | `docker compose run --rm test` | suite rapide (`pytest -m "not slow"`) |
| `make test-slow` | `docker compose run --rm test python -m pytest -m slow -v tests/integration/` | tests lents — **crée de vrais commits + tags** |
| `make pipeline` | `docker compose run --rm trainer dvc repro dvc/preprocessing/dvc.yaml` | pilier preprocessing |
| `make train DOSSIER=… H=…` | `docker compose run --rm trainer python cron/scripts/train.py --dossier … --horizon … --force` | entraînement ciblé |
| `make predict` | `docker compose run --rm trainer python cron/scripts/predict-archive.py` | prédiction + archivage |
| `make lint` | `docker compose run --rm trainer ruff check src tests cron` | `ruff` |
| `make api` | `docker compose up -d api` | API FastAPI (`:8000`, Swagger `/docs`) |
| `make lock` | (cf. Makefile) | regénère `requirements.lock` hashé |

## Intégration continue

`.github/workflows/ci.yml` tourne sur chaque PR et sur push `import-projet` :

- **`lint-test`** — `ruff` + `pytest -m "not slow"` + couverture (Python 3.11
  natif, torch CPU, cache pip).
- **`docker-build`** — reconstruit `Dockerfile.base` (cache GHA) pour garantir
  que l'image reste buildable quand `requirements.in` change.

`.github/workflows/release.yml` — sur tag `v*` : build + push de l'image API
vers `ghcr.io/<owner>/previ-r2d2-api:<version>` (+ `:latest`).

> **Protection de branche** (à activer une fois dans les *Settings* GitHub du
> dépôt) : sur `import-projet`, exiger que le job `lint-test` passe avant merge.

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
