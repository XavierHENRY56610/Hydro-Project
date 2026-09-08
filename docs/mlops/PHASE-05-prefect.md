# Phase 05 — Orchestration Prefect

## Ce que ça apporte

Les 6 scripts `cron/scripts/` tournaient à la main (ou via `dvc repro`), aux
cadences décrites dans le README mais sans planificateur. Cette phase les
enrobe en flows Prefect **planifiés**, avec retries, visibilité (UI) et
notification d'échec — sans réécrire une ligne de logique métier.

## Les 3 flows (`src/previ_r2d2/orchestration/flows.py`)

| Flow | Cadence | Étapes |
|---|---|---|
| `hourly-forecast` | `7 * * * *` | `predict-archive` |
| `daily-pipeline` | `30 3 * * *` | `maj-data` → `onboarding-check` → `onboarding-bv` → `build-data-preparation` → `validate-data` → `train --mode new` → `git push` + `dvc push` |
| `monthly-retrain` | `0 2 1 * *` | `train --mode monthly` → `git push` + `dvc push` |

Chaque `@task` lance le script `cron/scripts/` en **sous-processus** (même
invocation que les stages DVC). Un code retour non nul fait échouer la tâche.

**Retries** ciblés : `maj-data` 3× (Hub'Eau tombe souvent), `build-data-preparation`
et `predict-archive` 2×, `train` 1×. `validate-data` : 0 retry — une donnée
non conforme doit arrêter le flow tout de suite, avant l'entraînement.

**`push` git + DVC** : tâche du flow (les commits/tags de `promotion.py` sont
locaux). No-op si `DAGSHUB_TOKEN` absent (dev / CI).

## Notification d'échec (`notifications.py`)

Remplace le digest mail retiré. `on_failure=[notify_failure]` sur chaque flow :
si `PREFECT_FAILURE_WEBHOOK` (URL Discord ou Slack entrante) est défini, un
flow qui échoue **après épuisement des retries** poste un message court.
Sans la variable : log local uniquement.

## Docker

| Service | Rôle |
|---|---|
| `prefect-server` | API + UI Prefect (http://localhost:4200), base SQLite dans le volume `prefect_data` |
| `prefect-worker` | `python -m previ_r2d2.orchestration.serve` — publie les 3 déploiements planifiés et exécute les runs (mêmes bind-mounts que `trainer`, `depends_on` mlflow) |

```bash
make prefect                          # démarre server + worker planifié
# UI : http://localhost:4200
make flow-run FLOW=daily-pipeline      # déclenche un flow maintenant
```

`serve.py` utilise `prefect.serve(...)` — pas de work pool à configurer, le
worker interroge le serveur et exécute les runs en sous-processus.

## Tests

`tests/orchestration/test_flows.py` — 8 tests sous `prefect_test_harness`
(serveur éphémère). Enchaînement des étapes, arrêt sur échec de
`validate-data` **avant** l'entraînement, garde du push selon
`DAGSHUB_TOKEN`, notification webhook. `_run_script` / `subprocess` mockés —
les scripts `cron/` restent couverts par `tests/cron/`.

## Rapport avec les pipelines DVC

Les `dvc/*.yaml` restent la référence pour la **reproductibilité** (`dvc repro`
rejoue un état exact). Prefect est la couche **planification + exécution
opérationnelle** par-dessus les mêmes scripts. Les deux coexistent
volontairement (double filet, cf. phase 01 pour MLflow vs DVC).

## Ce qui reste pour plus tard

- Après un `train` qui promeut : le flow pousse déjà (git + DVC) ; un
  `prefect deployment run` manuel reste possible pour rejouer.
- Bascule work pool + workers distribués : seulement si le déploiement k8s
  (phase 07) l'exige.
