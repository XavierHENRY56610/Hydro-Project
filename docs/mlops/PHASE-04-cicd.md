# Phase 04 — CI/CD (GitHub Actions)

> ⚠️ **Activation manuelle requise.** Les workflows sont livrés dans
> `infrastructure/ci/` (et non `.github/workflows/`) car le PAT de push n'a
> pas le scope `workflow`. Voir `infrastructure/ci/README.md` pour les
> déplacer en une commande.

## Ce que ça apporte

Jusqu'ici les tests se lançaient à la main (`make test`). Une PR qui casse un
test ou le lint pouvait être mergée sans que personne ne le voie. Cette phase
met un **filet automatique** sur chaque PR et publie l'image de l'API sur tag.

## `ci.yml` — sur chaque PR + push `import-projet` / `main`

| Job | Contenu |
|---|---|
| **`lint-test`** | Python 3.11 natif (cache pip sur `requirements.in` + `pyproject.toml`), install torch CPU + `requirements.in` + `pip install -e . --no-deps`, puis `ruff check` et `pytest -q -m "not slow"` avec couverture. Le taux de couverture est poussé dans le résumé du run. |
| **`docker-build`** | Reconstruit `Dockerfile.base` avec le cache GHA. But : détecter tôt qu'un changement de `requirements.in` casse le build de l'image, sans relancer toute la suite dans le conteneur (elle tourne déjà dans `lint-test`). |

- `concurrency` annule les runs obsolètes quand on pousse plusieurs fois sur
  la même branche.
- `MLFLOW_TRACKING_URI=""` → suivi MLflow en no-op (aucun serveur en CI).
- Les tests `slow` (`tests/integration/`, qui créent de vrais commits + tags)
  sont exclus par le `addopts` de `pyproject.toml` — **jamais** en CI.

### Pourquoi Python natif et pas le conteneur ?

Le job de CI teste le *code*. Le conteneur est testé (a) en local par les
devs, (b) par le job `docker-build`, (c) par `release.yml` au moment de
publier. Faire tourner `pytest` en natif est ~3× plus rapide et bien plus
simple à déboguer qu'un `docker compose run` en CI.

## `release.yml` — sur tag `v*`

`git tag v1.0.0 && git push --tags` →
`ghcr.io/<owner>/previ-r2d2-api:1.0.0` **et** `:latest` sur le GitHub
Container Registry.

- `docker build` classique (builder lié au daemon) pour que le
  `FROM previ-r2d2-base` de `Dockerfile.api` résolve l'image de base
  construite juste avant dans le job.
- `Dockerfile.api` accepte `ARG BASE_IMAGE` (défaut `previ-r2d2-base:latest`)
  pour rester paramétrable.
- Auth GHCR via le `GITHUB_TOKEN` du run (`permissions: packages: write`).

## Étape manuelle (une fois)

Dans **Settings → Branches** du dépôt GitHub : règle de protection sur
`import-projet` exigeant le job **`lint-test`** vert avant merge. Ne peut pas
se faire par fichier versionné (API admin uniquement).

## Ce qui reste pour plus tard

- Job `dvc status` / vérif que `dvc.lock` ne référence pas de stage supprimé —
  nécessite un token DagsHub en secret CI (à ajouter quand l'accès données
  sera débloqué).
- Publier aussi une image `trainer` / `pipeline` si un déploiement k8s des
  jobs le demande (phase 07).
